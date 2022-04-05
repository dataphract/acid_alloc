//! A binary-buddy allocator.

#![no_std]
#![allow(unstable_name_collisions)]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(alloc_layout_extra)]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]
#![cfg_attr(feature = "int_log", feature(int_log))]
#![cfg_attr(not(feature = "sptr"), feature(strict_provenance))]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

mod bitmap;
mod polyfill;

#[cfg(test)]
mod tests;

use core::{
    alloc::Layout,
    cmp,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ptr::NonNull,
};

#[cfg(feature = "allocator_api")]
use core::alloc::{AllocError, Allocator};

#[cfg(all(feature = "allocator_api", any(feature = "alloc", test)))]
use alloc::alloc::Global;

#[cfg(feature = "sptr")]
use sptr::Strict;

use crate::{bitmap::Bitmap, polyfill::*};

fn round_up_pow2(x: usize) -> Option<usize> {
    match x {
        0 => None,
        1 => Some(1),
        x if x >= (1 << 63) => None,
        _ => Some(2usize.pow((x - 1).log2() as u32 + 1)),
    }
}

/// A link in a linked list of buddy blocks.
///
/// This type is meant to be embedded in the block itself.
#[repr(C)]
struct BuddyLink {
    // Rather than using a pointer, store only the address of the next link.
    // This avoids accidentally violating stacked borrows; the link "points to"
    // the next block, but by forgoing an actual pointer, it does not imply a
    // borrow.
    //
    // NOTE: Using this method, any actual pointer to a block must be acquired
    // via the allocator base pointer, and NOT by casting this address directly!
    next: Option<NonZeroUsize>,
}

impl BuddyLink {
    /// Initializes a `BuddyLink` at the pointed-to location.
    ///
    /// The address of the initialized `BuddyLink` is returned rather than the
    /// pointer. This indicates to the compiler that any effective mutable
    /// borrow of `ptr` has ended.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `ptr` must be valid for reads and writes for `size_of::<BuddyLink>()`
    ///   bytes.
    /// - If `next` is `Some(n)`, then `n` must be the address of an
    ///   initialized `BuddyLink` value.
    unsafe fn init(
        mut ptr: NonNull<MaybeUninit<BuddyLink>>,
        next: Option<NonZeroUsize>,
    ) -> NonZeroUsize {
        assert_eq!(ptr.as_ptr().align_offset(mem::align_of::<BuddyLink>()), 0);

        let addr = unsafe {
            let uninit_mut: &mut MaybeUninit<_> = ptr.as_mut();
            let ptr = uninit_mut.as_mut_ptr();
            ptr.write(BuddyLink { next });
            ptr.addr()
        };

        NonZeroUsize::new(addr).unwrap()
    }
}

struct BuddyLevel {
    block_size: usize,
    free_list: Option<NonZeroUsize>,
    buddies: Bitmap,
    splits: Option<Bitmap>,
}

impl BuddyLevel {
    /// Retrieves the index of the block which starts `block_ofs` bytes from the
    /// base.
    #[inline]
    fn index_of(&self, block_ofs: usize) -> usize {
        assert_eq!(block_ofs % self.block_size, 0);

        block_ofs.checked_div(self.block_size).unwrap()
    }

    /// Retrieves the index of the buddy bit for the block which starts
    /// `block_ofs` bytes from the base.
    #[inline]
    fn buddy_bit(&self, block_ofs: usize) -> usize {
        self.index_of(block_ofs).checked_div(2).unwrap()
    }

    /// Retrieves the offset of the buddy of the block which starts
    /// `block_ofs` bytes from the base.
    #[inline]
    fn buddy_ofs(&self, block_ofs: usize) -> usize {
        block_ofs ^ self.block_size
    }

    /// Pops a block from the free list.
    ///
    /// If the free list is empty, returns `None`.
    unsafe fn free_list_pop(&mut self, base: BasePtr) -> Option<NonZeroUsize> {
        let head = self.free_list.take()?;

        self.free_list = unsafe { base.link_mut(head).next };

        Some(head)
    }

    /// Pushes a block onto the free list.
    unsafe fn free_list_push(&mut self, base: BasePtr, block: NonZeroUsize) {
        assert_eq!(block.get() & (mem::align_of::<BuddyLink>() - 1), 0);

        let uninit = base.with_addr(block).cast::<MaybeUninit<BuddyLink>>();
        let link = unsafe { BuddyLink::init(uninit, self.free_list) };
        self.free_list = Some(link);
    }

    /// Removes the specified block from the free list.
    ///
    /// If the block is not present, returns `None`.
    unsafe fn free_list_find_remove(
        &mut self,
        base: BasePtr,
        block: NonZeroUsize,
    ) -> Option<NonZeroUsize> {
        let head = self.free_list?;

        if head == block {
            self.free_list = unsafe { base.link_mut(head).next.take() };

            return Some(head);
        }

        let mut cur = head;
        loop {
            let next = match unsafe { base.link_mut(cur).next } {
                Some(n) => n,

                // Block not found in free list.
                None => break,
            };

            let prev = cur;
            cur = next;
            assert!(prev != cur);

            if cur == block {
                unsafe {
                    // SAFETY: `prev` and `cur` must not be overlapping (which they shouldn't be).
                    let prev_mut = base.link_mut(prev);
                    let cur_mut = base.link_mut(cur);

                    prev_mut.next = cur_mut.next.take();
                }

                return Some(block);
            }
        }

        None
    }

    /// Allocates a block from the free list.
    ///
    /// The returned pointer has the provenance of `base`.
    unsafe fn allocate_one(&mut self, base: BasePtr) -> Option<NonNull<u8>> {
        let block = unsafe { self.free_list_pop(base)? };

        let ofs = base.offset_to(block);

        self.buddies.toggle(self.buddy_bit(ofs));

        Some(base.with_addr(block))
    }

    /// Assigns half a block from the level above this one.
    unsafe fn assign_half(&mut self, base: BasePtr, block: NonZeroUsize) {
        let ofs = base.offset_to(block);

        let buddy_bit = self.buddy_bit(ofs);
        assert!(!self.buddies.get(buddy_bit));
        self.buddies.set(buddy_bit, true);

        unsafe {
            self.free_list_push(base, block);
        }
    }

    fn free(&mut self, base: BasePtr, block: NonNull<u8>, coalesce: bool) -> Option<NonNull<u8>> {
        // Immediately drop and shadow the mutable pointer by converting it to
        // an address.  This indicates to the compiler that the base pointer has
        // sole access to the block.
        let block = block.addr();
        let block_ofs = base.offset_to(block);
        let buddy_bit = self.buddy_bit(block_ofs);
        self.buddies.toggle(buddy_bit);

        let split_bit = self.index_of(block_ofs);
        if let Some(splits) = self.splits.as_mut() {
            splits.set(split_bit, false);
        }

        if !coalesce || self.buddies.get(buddy_bit) {
            unsafe { self.free_list_push(base, block) };
            None
        } else {
            let buddy_ofs = self.buddy_ofs(block_ofs);
            let buddy =
                NonZeroUsize::new(base.ptr.addr().get().checked_add(buddy_ofs).unwrap()).unwrap();

            // Remove the buddy block from the free list.
            if unsafe { self.free_list_find_remove(base, buddy) }.is_none() {
                panic!("missing buddy in free list");
            }

            // Return the coalesced block.
            let coalesced_ofs = buddy_ofs & !self.block_size;
            Some(base.with_offset(coalesced_ofs).unwrap())
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct BasePtr {
    ptr: NonNull<u8>,
}

impl BasePtr {
    /// Calculates the offset from `self` to `block`.
    fn offset_to(self, block: NonZeroUsize) -> usize {
        block.get().checked_sub(self.ptr.addr().get()).unwrap()
    }

    /// Returns a mutable reference to the `BuddyLink` at `link`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `link` must be a properly aligned address for `BuddyLink` values.
    /// - The memory at `link` must contain a properly initialized `BuddyLink` value.
    /// - The memory at `link` must be unallocated.
    unsafe fn link_mut<'a>(self, link: NonZeroUsize) -> &'a mut BuddyLink {
        unsafe { self.ptr.with_addr(link).cast::<BuddyLink>().as_mut() }
    }

    /// Creates a new pointer with the given address.
    fn with_addr(self, addr: NonZeroUsize) -> NonNull<u8> {
        self.ptr.with_addr(addr)
    }

    /// Creates a new pointer with the given offset.
    fn with_offset(self, offset: usize) -> Option<NonNull<u8>> {
        let raw = self.ptr.addr().get().checked_add(offset)?;
        let addr = NonZeroUsize::new(raw)?;
        Some(self.ptr.with_addr(addr))
    }
}

/// Types which can be used to provide memory which backs an allocator.
///
/// This is a supertrait of [`Allocator`], and is implemented by the following types:
/// - The `Raw` marker type indicates that an allocator is not backed by another
///   allocator. This is the case when constructing the allocator from raw
///   pointers. Memory used by this allocator can be reclaimed using
///   `.into_raw_parts()`.
/// - The `Global` marker type indicates that an allocator is backed by the
///   global allocator. The allocator will free its memory on drop.
/// - Any type `A` which implements [`Allocator`] indicates that an allocator is
///   backed by an instance of `A`. The allocator will free its memory on drop.
///
/// [`Allocator`]: https://doc.rust-lang.org/stable/core/alloc/trait.Allocator.html
pub trait BackingAllocator: Sealed {
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// A marker type indicating that an allocator is backed by raw pointers.
pub struct Raw;
impl Sealed for Raw {}
impl BackingAllocator for Raw {
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
}

#[cfg(all(any(feature = "alloc", test), not(feature = "allocator_api")))]
/// The global memory allocator.
pub struct Global;

#[cfg(all(any(feature = "alloc", test), not(feature = "allocator_api")))]
impl Sealed for Global {}

#[cfg(all(any(feature = "alloc", test), not(feature = "allocator_api")))]
impl BackingAllocator for Global {
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

#[cfg(feature = "allocator_api")]
impl<A: Allocator> Sealed for A {}
#[cfg(feature = "allocator_api")]
impl<A: Allocator> BackingAllocator for A {
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Allocator::deallocate(self, ptr, layout) };
    }
}

/// A binary-buddy allocator.
///
/// This takes two const parameters:
/// - `PGSIZE` is the size of the smallest allocations the allocator can make.
/// - `ORDER` is the number of levels in the allocator.
pub struct BuddyAllocator<const ORDER: usize, const PGSIZE: usize, A: BackingAllocator> {
    /// Pointer to the region managed by this allocator.
    base: BasePtr,
    /// Pointer to the region that backs the bitmaps.
    ///
    /// This must not be used while the allocator exists; it is stored solely so
    /// that it may be returned in `into_raw_parts()`.
    metadata: NonNull<u8>,
    /// The number of zero-order blocks managed by this allocator.
    num_blocks: usize,
    levels: [BuddyLevel; ORDER],
    backing_allocator: A,
}

const fn sum_of_powers_of_2(max: u32) -> usize {
    2 * (2_usize.pow(max) - 1)
}

impl<const ORDER: usize, const PGSIZE: usize> BuddyAllocator<ORDER, PGSIZE, Raw> {
    /// Construct a new `BuddyAllocator` from raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `base` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::region_layout()`], and it must be valid for reads
    ///   and writes for the entire size indicated by that `Layout`.
    /// - `metadata` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::metadata_layout()`], and it must be valid for
    ///   reads and writes for the entire size indicated by that `Layout`.
    ///
    /// [`Layout`]: core::alloc::Layout
    pub unsafe fn new_raw(
        base: NonNull<u8>,
        num_blocks: usize,
        metadata: NonNull<u8>,
    ) -> BuddyAllocator<ORDER, PGSIZE, Raw> {
        unsafe {
            BuddyAllocatorParts::<ORDER, PGSIZE>::new(base, num_blocks, metadata)
                .with_backing_allocator(Raw)
        }
    }
}

#[cfg(all(not(feature = "allocator_api"), any(feature = "alloc", test)))]
impl<const ORDER: usize, const PGSIZE: usize> BuddyAllocator<ORDER, PGSIZE, Global> {
    pub fn new(num_blocks: usize) -> BuddyAllocator<ORDER, PGSIZE, Global> {
        let region_layout = Self::region_layout(num_blocks);
        let metadata_layout = Self::metadata_layout(num_blocks);

        unsafe {
            let region_ptr = {
                let raw = alloc::alloc::alloc(region_layout);
                NonNull::new(raw).unwrap()
            };

            let metadata_ptr = {
                let raw = alloc::alloc::alloc(metadata_layout);
                NonNull::new(raw).unwrap()
            };

            BuddyAllocatorParts::<ORDER, PGSIZE>::new(region_ptr, num_blocks, metadata_ptr)
                .with_backing_allocator(Global)
        }
    }
}

#[cfg(all(feature = "allocator_api", any(feature = "alloc", test)))]
impl<const ORDER: usize, const PGSIZE: usize> BuddyAllocator<ORDER, PGSIZE, Global> {
    pub fn new(num_blocks: usize) -> BuddyAllocator<ORDER, PGSIZE, Global> {
        BuddyAllocator::<ORDER, PGSIZE, Global>::new_in(Global, num_blocks)
            .expect("global allocation failed")
    }
}

#[cfg(feature = "allocator_api")]
impl<const ORDER: usize, const PGSIZE: usize, A: Allocator> BuddyAllocator<ORDER, PGSIZE, A> {
    pub fn new_in(
        allocator: A,
        num_blocks: usize,
    ) -> Result<BuddyAllocator<ORDER, PGSIZE, A>, AllocError> {
        let region_layout = Self::region_layout(num_blocks);
        let metadata_layout = Self::metadata_layout(num_blocks);

        let region = allocator.allocate(region_layout)?;
        let metadata = match allocator.allocate(metadata_layout) {
            Ok(m) => m,
            Err(e) => unsafe {
                // SAFETY: region was received as NonNull via Allocator::allocate
                let region_ptr = NonNull::new_unchecked(region.as_ptr() as *mut u8);
                allocator.deallocate(region_ptr, region_layout);
                return Err(e);
            },
        };

        unsafe {
            // SAFETY: both pointers were received as NonNull via Allocator::allocate
            let region_ptr = NonNull::new_unchecked(region.as_ptr() as *mut u8);
            let metadata_ptr = NonNull::new_unchecked(metadata.as_ptr() as *mut u8);

            Ok(
                BuddyAllocatorParts::<ORDER, PGSIZE>::new(region_ptr, num_blocks, metadata_ptr)
                    .with_backing_allocator(allocator),
            )
        }
    }
}

impl<const ORDER: usize, const PGSIZE: usize, A: BackingAllocator> Drop
    for BuddyAllocator<ORDER, PGSIZE, A>
{
    fn drop(&mut self) {
        let region = self.base.ptr;
        let metadata = self.metadata;
        let num_blocks = self.num_blocks;

        let region_layout = Self::region_layout(num_blocks);
        let metadata_layout = Self::metadata_layout(num_blocks);

        unsafe {
            self.backing_allocator.deallocate(region, region_layout);
            self.backing_allocator.deallocate(metadata, metadata_layout);
        }
    }
}

impl<const ORDER: usize, const PGSIZE: usize, A: BackingAllocator>
    BuddyAllocator<ORDER, PGSIZE, A>
{
    pub fn region_layout(num_blocks: usize) -> Layout {
        assert!(ORDER > 0);
        assert!(PGSIZE >= mem::size_of::<BuddyLink>() && PGSIZE.is_power_of_two());
        let order: u32 = ORDER.try_into().unwrap();

        let size = 2usize.pow(order - 1) * PGSIZE * num_blocks;
        let align = PGSIZE;

        Layout::from_size_align(size, align).unwrap()
    }

    /// Returns the layout requirements for the metadata region.
    pub fn metadata_layout(num_blocks: usize) -> Layout {
        assert!(ORDER > 0);
        assert!(PGSIZE >= mem::size_of::<BuddyLink>() && PGSIZE.is_power_of_two());
        let order: u32 = ORDER.try_into().unwrap();

        // Each level needs one buddy bit per pair of blocks.
        let num_pairs = (num_blocks + 1) / 2;

        // This is the layout required for the buddy bitmap of level 0.
        let buddy_l0_layout = Bitmap::map_layout(num_pairs);

        // Each subsequent level requires at most twice as much space as the
        // level 0 bitmap. It may require less if the number of level 0 blocks
        // is not a multiple of the bitmap block size, but for simplicity each
        // level is given exactly twice the space of the previous level.
        let (buddy_layout, _) = buddy_l0_layout
            .repeat(sum_of_powers_of_2(order - 1))
            .unwrap();

        let full_layout = if ORDER == 1 {
            // If ORDER is 1, then no split bitmap is required.
            buddy_layout
        } else {
            // Each level except level (ORDER - 1) needs one split bit per block.
            let split_l0_layout = Bitmap::map_layout(num_blocks);

            // Let K equal the size of a split bitmap for level 0. If ORDER is:
            // - 2, then 1 split bitmap is needed of size (2 - 1)K = K.
            // - 3, then 2 split bitmaps are needed of total size (3 - 1)K + (2 - 1)K = 3K.
            // - ...
            // - N, then 2 ^ (N - 2) split bitmaps are needed of total size
            //   (N - 1)K + (N - 2)K + ... + (2 - 1)K = 2 * (2 ^ (N - 1) - 1) * K
            //                                        = (sum from x = 1 to (N - 1) of 2^x) * K
            let split_pow = order - 1;

            let (split_layout, _) = split_l0_layout
                .repeat(sum_of_powers_of_2(split_pow))
                .unwrap();
            let (full_layout, _) = buddy_layout.extend(split_layout).unwrap();

            full_layout
        };

        full_layout
    }

    fn alloc_level(&self, size: usize) -> Option<usize> {
        let max_block_size = self.levels[0].block_size;
        if size > max_block_size {
            return None;
        }

        let alloc_size = cmp::max(round_up_pow2(size).unwrap(), PGSIZE);
        let level: usize = (max_block_size.log2() - alloc_size.log2())
            .try_into()
            .unwrap();

        Some(level)
    }

    fn min_free_level(&self, block_ofs: usize) -> usize {
        let max_block_size = self.levels[0].block_size;

        if block_ofs == 0 {
            return 0;
        }

        let max_size = 1 << block_ofs.trailing_zeros();
        if max_size > max_block_size {
            return 0;
        }

        assert!(max_size >= PGSIZE);

        (max_block_size.log2() - max_size.log2())
            .try_into()
            .unwrap()
    }

    pub unsafe fn allocate(&mut self, size: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }

        let target_level = self.alloc_level(size)?;

        // If there is a free block of the correct size, return it immediately.
        if let Some(block) = unsafe { self.levels[target_level].allocate_one(self.base) } {
            return Some(block);
        }

        // Otherwise, scan increasing block sizes until a free block is found.
        let (block, init_level) = (0..target_level).rev().find_map(|level| {
            unsafe { self.levels[level].allocate_one(self.base) }.map(|blk| (blk, level))
        })?;

        let block_ofs = self.base.offset_to(block.addr());

        // Once a free block is found, split it repeatedly to obtain a
        // suitably sized block.
        for level in init_level..target_level {
            // Split the block. The address of the front half does not change.
            let half_block_size = self.levels[level].block_size / 2;
            let back_half = NonZeroUsize::new(block.addr().get() + half_block_size).unwrap();

            // Mark the block as split.
            let split_bit = self.levels[level].index_of(block_ofs);
            if let Some(s) = self.levels[level].splits.as_mut() {
                s.set(split_bit, true);
            }

            // Add one half of the split block to the next level's free list.
            unsafe { self.levels[level + 1].assign_half(self.base, back_half) };
        }

        // The returned block inherits the provenance of the base pointer.
        Some(self.base.with_addr(block.addr()))
    }

    pub unsafe fn free(&mut self, block: NonNull<u8>) {
        // Some addresses can't come from earlier levels because their addresses
        // imply a smaller block size.
        let block_ofs = self.base.offset_to(block.addr());
        let min_level = self.min_free_level(block_ofs);

        let mut at_level = None;
        for level in min_level..self.levels.len() {
            if self.levels[level]
                .splits
                .as_ref()
                .map(|s| !s.get(self.levels[level].index_of(block_ofs)))
                .unwrap_or(true)
            {
                at_level = Some(level);
                break;
            }
        }

        let at_level = at_level.expect("no level found to free block");

        let mut block = Some(block);
        for level in (0..=at_level).rev() {
            match block.take() {
                Some(b) => {
                    block = self.levels[level].free(self.base, b, level != 0);
                }
                None => break,
            }
        }

        assert!(block.is_none(), "top level coalesced a block");
    }

    /// Decomposes the allocator into its raw components.
    ///
    /// The returned tuple contains the region pointer and the metadata pointer.
    ///
    /// # Safety
    ///
    /// All outstanding allocations are invalidated when this method is called;
    /// the returned region pointer becomes the sole owner of the region that
    /// was used to construct the allocator. As such, all allocations made from
    /// this allocator should be either freed or forgotten before calling this
    /// method.
    pub unsafe fn into_raw_parts(self) -> (NonNull<u8>, NonNull<u8>) {
        let BuddyAllocator { base, metadata, .. } = self;

        (base.ptr, metadata)
    }
}

/// Like a `BuddyAllocator`, but without a `Drop` impl or an associated
/// allocator.
///
/// This assists in tacking on the allocator type parameter because this struct can be
/// moved out of, while `BuddyAllocator` itself cannot.
struct BuddyAllocatorParts<const ORDER: usize, const PGSIZE: usize> {
    base: BasePtr,
    metadata: NonNull<u8>,
    num_blocks: usize,
    levels: [BuddyLevel; ORDER],
}

impl<const ORDER: usize, const PGSIZE: usize> BuddyAllocatorParts<ORDER, PGSIZE> {
    fn with_backing_allocator<A: BackingAllocator>(
        self,
        backing_allocator: A,
    ) -> BuddyAllocator<ORDER, PGSIZE, A> {
        let BuddyAllocatorParts {
            base,
            metadata,
            num_blocks,
            levels,
        } = self;

        BuddyAllocator {
            base,
            metadata,
            num_blocks,
            levels,
            backing_allocator,
        }
    }

    /// Construct a new `BuddyAllocatorParts` from raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `base` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::region_layout()`], and it must be valid for reads
    ///   and writes for the entire size indicated by that `Layout`.
    /// - `metadata` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::metadata_layout()`], and it must be valid for
    ///   reads and writes for the entire size indicated by that `Layout`.
    ///
    /// [`Layout`]: core::alloc::Layout
    pub unsafe fn new(
        base: NonNull<u8>,
        num_blocks: usize,
        metadata: NonNull<u8>,
    ) -> BuddyAllocatorParts<ORDER, PGSIZE> {
        assert!(ORDER > 0);
        assert!(PGSIZE >= mem::size_of::<BuddyLink>() && PGSIZE.is_power_of_two());
        let full_layout = BuddyAllocator::<ORDER, PGSIZE, Raw>::metadata_layout(num_blocks);

        // TODO: use MaybeUninit::uninit_array when not feature gated
        let mut levels: [MaybeUninit<BuddyLevel>; ORDER] = unsafe {
            // SAFETY: An uninitialized `[MaybeUninit<_>; _]` is valid.
            MaybeUninit::<[MaybeUninit<BuddyLevel>; ORDER]>::uninit().assume_init()
        };

        let mut meta_curs = metadata.as_ptr();

        for (li, level) in levels.iter_mut().enumerate() {
            let block_size = 2_usize.pow((ORDER - li) as u32 - 1) * PGSIZE;
            let block_factor = 2_usize.pow(li as u32);
            let num_blocks = block_factor * num_blocks;
            let num_pairs = num_blocks + 1 / 2;

            let buddy_size = Bitmap::map_layout(num_pairs).size();
            let buddy_bitmap = unsafe { Bitmap::new(num_pairs, meta_curs as *mut u64) };

            meta_curs = unsafe {
                meta_curs.offset(
                    buddy_size
                        .try_into()
                        .expect("buddy bitmap layout size overflows isize"),
                )
            };

            let split_bitmap = if li < ORDER - 1 {
                let split_size = Bitmap::map_layout(num_blocks).size();
                let split_bitmap = unsafe { Bitmap::new(num_blocks, meta_curs as *mut u64) };

                meta_curs = unsafe {
                    meta_curs.offset(
                        split_size
                            .try_into()
                            .expect("split bitmap layout size overflows isize"),
                    )
                };

                Some(split_bitmap)
            } else {
                None
            };

            unsafe {
                level.as_mut_ptr().write(BuddyLevel {
                    block_size,
                    free_list: None,
                    buddies: buddy_bitmap,
                    splits: split_bitmap,
                });
            }
        }

        assert!(
            unsafe { meta_curs.offset_from(metadata.as_ptr()) }
                < full_layout.size().try_into().unwrap()
        );

        let mut levels = unsafe {
            // TODO: This is the stdlib's implementation of
            // MaybeUninit::array_assume_init(). When that's stable, use it
            // instead.
            //
            // SAFETY:
            // - `levels` is fully initialized.
            // - `MaybeUninit<T>` and `T` have the same layout.
            // - `MaybeUninit<T>` won't drop `T`, so no double-frees.
            (&levels as *const _ as *const [BuddyLevel; ORDER]).read()
        };

        // Initialize the top-level free list by emplacing links in each block.
        let mut next_link = None;
        for block_idx in (0..num_blocks).rev() {
            let block_offset = block_idx * levels[0].block_size;

            let link = unsafe {
                // TODO: use NonZeroUsize::checked_add (this is usize::checked_add)
                BuddyLink::init(
                    base.map_addr(|b| {
                        let raw = b.get().checked_add(block_offset).unwrap();
                        NonZeroUsize::new(raw).unwrap()
                    })
                    .cast(),
                    next_link,
                )
            };
            next_link = Some(link);
        }

        levels[0].free_list = next_link;

        BuddyAllocatorParts {
            base: BasePtr { ptr: base },
            metadata,
            num_blocks,
            levels,
        }
    }
}

#[doc(hidden)]
mod private {
    pub trait Sealed {}
}
use private::Sealed;
