//! A binary-buddy memory allocator.

#![cfg(any(feature = "sptr", feature = "unstable"))]

use core::{
    alloc::Layout,
    cmp,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ptr::NonNull,
};

/// Declares and implements `Allocator` for wrappers around an allocator.
///
/// If the "unstable" feature is not enabled, this is a no-op.
macro_rules! declare_wrappers {
    ($($(#[$attr:meta])* $wrapper:ident uses $typename:ident via $method:path)*) => {
        $(
            #[doc = concat!("A `BuddyAllocator` wrapped by a `", stringify!($typename), "`.")]
            ///
            /// This type implements [`Allocator`].
            #[cfg(feature = "unstable")]
            $(#[$attr])*
            pub struct $wrapper<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator>
            {
                inner: $typename<BuddyAllocator<BLK_SIZE, LEVELS, A>>,
            }

            #[cfg(feature = "unstable")]
            impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> $wrapper<BLK_SIZE, LEVELS, A>
            {
                /// Returns a reference to the inner wrapper.
                pub fn inner(&self) -> &$typename<BuddyAllocator<BLK_SIZE, LEVELS, A>> {
                    &self.inner
                }
            }

            // SAFETY:
            //
            // See https://doc.rust-lang.org/nightly/core/alloc/trait.Allocator.html#safety.
            //
            // - Allocated blocks point to memory owned by the `BuddyAllocator` and are
            //   valid until it is dropped.
            // - `BuddyAllocator` is not `Clone`, and moving it does not invalidate
            //   allocated memory because that memory is behind a pointer.
            // - Any pointer to a currently allocated block is safe to deallocate.
            #[cfg(feature = "unstable")]
            #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
            unsafe impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> Allocator
                for $wrapper<BLK_SIZE, LEVELS, A>
            {
                fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                    $method(&self.inner).allocate(layout)
                }

                unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                    let _ = layout;

                    unsafe { $method(&self.inner).deallocate(ptr) }
                }
            }
        )*
    };
}

declare_wrappers! {
    RefCellBuddyAllocator uses RefCell via RefCell::borrow_mut
}

#[cfg(feature = "std")]
declare_wrappers! {
    #[cfg_attr(docs_rs, doc(cfg(feature = "std")))]
    MutexBuddyAllocator uses Mutex via Mutex::lock

    #[cfg_attr(docs_rs, doc(cfg(feature = "std")))]
    RwLockBuddyAllocator uses RwLock via RwLock::write
}

#[cfg(feature = "unstable")]
use core::alloc::{AllocError, Allocator};

#[cfg(not(feature = "unstable"))]
use crate::AllocError;

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
use alloc::alloc::Global;

use crate::{bitmap::Bitmap, BackingAllocator, BasePtr, BlockLink, Raw};

#[cfg(not(feature = "unstable"))]
use crate::polyfill::*;

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
use crate::Global;

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

    /// Pushes a block onto the free list.
    unsafe fn free_list_push(&mut self, base: BasePtr, block: NonZeroUsize) {
        assert_eq!(block.get() & (mem::align_of::<BlockLink>() - 1), 0);

        let new_head = block;

        if let Some(old_head) = self.free_list {
            let old_head_mut = unsafe { base.link_mut(old_head) };
            old_head_mut.prev = Some(new_head);
        }

        let old_head = self.free_list;

        // If `old_head` exists, it points back to `new_head`.

        unsafe {
            base.init_link_at(
                block,
                BlockLink {
                    next: old_head,
                    prev: None,
                },
            )
        };

        // `new_head` points forward to `old_head`.
        // `old_head` points back to `new_head`.
        self.free_list = Some(new_head);
    }

    /// Removes the specified block from the free list.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - The memory at `block` must be within the provenance of `base` and
    ///   valid for reads and writes for `size_of::<BlockLink>()` bytes.
    /// - `block` must be the address of an element of `self.free_list`.
    unsafe fn free_list_remove(&mut self, base: BasePtr, block: NonZeroUsize) {
        unsafe {
            let removed = base.link_mut(block);

            match removed.prev {
                // Link `prev` forward to `next`.
                Some(p) => base.link_mut(p).next = removed.next,

                // If there's no previous block, then `removed` is the head of
                // the free list.
                None => self.free_list = removed.next,
            }

            if let Some(n) = removed.next {
                // Link `next` back to `prev`.
                base.link_mut(n).prev = removed.prev;
            }
        }
    }

    /// Allocates a block from the free list.
    unsafe fn allocate(&mut self, base: BasePtr, align: usize) -> Option<NonZeroUsize> {
        let mut current = self.free_list;

        while let Some(cur) = current {
            if cur.get() % align == 0 {
                break;
            }

            current = unsafe { base.link_mut(cur).next };
        }

        // If current is `Some`, then a suitable block was found.
        let block = current?;

        unsafe { self.free_list_remove(base, block) };
        let ofs = base.offset_to(block);
        self.buddies.toggle(self.buddy_bit(ofs));

        Some(block)
    }

    /// Assigns a block to this level.
    unsafe fn assign(&mut self, base: BasePtr, block: NonZeroUsize) {
        let ofs = base.offset_to(block);

        let buddy_bit = self.buddy_bit(ofs);
        assert!(!self.buddies.get(buddy_bit));
        self.buddies.set(buddy_bit, true);

        unsafe { self.free_list_push(base, block) };
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
            unsafe { self.free_list_remove(base, buddy) };

            // Return the coalesced block.
            let coalesced_ofs = buddy_ofs & !self.block_size;
            Some(base.with_offset(coalesced_ofs).unwrap())
        }
    }
}

/// A binary-buddy allocator.
///
/// This takes two const parameters:
/// - `BLK_SIZE` is the size of the largest allocations the allocator can make.
/// - `LEVELS` is the number of levels in the allocator.
///
/// These parameters are subject to the following invariants:
/// - `BLK_SIZE` must be a power of two.
/// - `LEVELS` must be nonzero and less than `usize::BITS`.
/// - The minumum block size must be at least `2 * mem::size_of<usize>()`;  it
///   can be calculated with the formula `BLK_SIZE >> (LEVELS - 1)`.
///
/// Attempting to construct a `BuddyAllocator` whose const parameters violate
/// these invariants will result in a panic.
///
/// For example, the type of a buddy allocator which can allocate blocks of
/// sizes from 16 to 4096 bytes would be:
///
/// ```
/// use acid_alloc::BuddyAllocator;
///
/// // Minimum block size == BLK_SIZE >> (LEVELS - 1)
/// //                 16 ==     4096 >> (     9 - 1)
/// type CustomBuddyAllocator<A> = BuddyAllocator<4096, 9, A>;
/// # fn main() {}
/// ```
pub struct BuddyAllocator<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> {
    /// Pointer to the region managed by this allocator.
    base: BasePtr,
    /// Pointer to the region that backs the bitmaps.
    ///
    /// This must not be used while the allocator exists; it is stored solely so
    /// that it may be returned in `into_raw_parts()`.
    metadata: NonNull<u8>,
    /// The number of level-zero blocks managed by this allocator.
    num_blocks: usize,
    levels: [BuddyLevel; LEVELS],
    backing_allocator: A,
}

impl<const BLK_SIZE: usize, const LEVELS: usize> BuddyAllocator<BLK_SIZE, LEVELS, Raw> {
    /// Construct a new `BuddyAllocator` from raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `region` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::region_layout(num_blocks)`], and it must be valid
    ///   for reads and writes for the entire size indicated by that `Layout`.
    /// - `metadata` must be a pointer to a region that satisfies the [`Layout`]
    ///   returned by [`Self::metadata_layout(num_blocks)`], and it must be
    ///   valid for reads and writes for the entire size indicated by that
    ///   `Layout`.
    ///
    /// [`Self::region_layout(num_blocks)`]: Self::region_layout
    /// [`Self::metadata_layout(num_blocks)`]: Self::metadata_layout
    /// [`Layout`]: core::alloc::Layout
    pub unsafe fn new_raw(
        metadata: NonNull<u8>,
        region: NonNull<u8>,
        num_blocks: usize,
    ) -> BuddyAllocator<BLK_SIZE, LEVELS, Raw> {
        unsafe {
            BuddyAllocatorParts::<BLK_SIZE, LEVELS>::new(metadata, region, num_blocks)
                .with_backing_allocator(Raw)
        }
    }
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
impl<const BLK_SIZE: usize, const LEVELS: usize> BuddyAllocator<BLK_SIZE, LEVELS, Global> {
    /// Constructs a new `BuddyAllocator` backed by the global allocator.
    ///
    /// # Errors
    ///
    /// If allocation fails, this constructor invokes [`handle_alloc_error`].
    ///
    /// [`handle_alloc_error`]: alloc::alloc::handle_alloc_error
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn new(num_blocks: usize) -> BuddyAllocator<BLK_SIZE, LEVELS, Global> {
        let region_layout = Self::region_layout(num_blocks);
        let metadata_layout = Self::metadata_layout(num_blocks);

        unsafe {
            let region_ptr = {
                let raw = alloc::alloc::alloc(region_layout);
                NonNull::new(raw).unwrap_or_else(|| alloc::alloc::handle_alloc_error(region_layout))
            };

            let metadata_ptr = {
                let raw = alloc::alloc::alloc(metadata_layout);
                NonNull::new(raw).unwrap_or_else(|| {
                    alloc::alloc::dealloc(region_ptr.as_ptr(), region_layout);
                    alloc::alloc::handle_alloc_error(metadata_layout)
                })
            };

            BuddyAllocatorParts::<BLK_SIZE, LEVELS>::new(metadata_ptr, region_ptr, num_blocks)
                .with_backing_allocator(Global)
        }
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl<const BLK_SIZE: usize, const LEVELS: usize> BuddyAllocator<BLK_SIZE, LEVELS, Global> {
    /// Constructs a new `BuddyAllocator` backed by the global allocator.
    ///
    /// # Errors
    ///
    /// If allocation fails, this constructor invokes [`handle_alloc_error`].
    ///
    /// [`handle_alloc_error`]: alloc::alloc::handle_alloc_error
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn new(num_blocks: usize) -> BuddyAllocator<BLK_SIZE, LEVELS, Global> {
        BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new_in(num_blocks, Global)
            .expect("global allocation failed")
    }
}

#[cfg(feature = "unstable")]
impl<const BLK_SIZE: usize, const LEVELS: usize, A: Allocator> BuddyAllocator<BLK_SIZE, LEVELS, A> {
    /// Constructs a new `BuddyAllocator` backed by `allocator`.
    ///
    /// # Errors
    ///
    /// If allocation fails, returns `Err(AllocError)`.
    #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
    pub fn new_in(
        num_blocks: usize,
        allocator: A,
    ) -> Result<BuddyAllocator<BLK_SIZE, LEVELS, A>, AllocError> {
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
                BuddyAllocatorParts::<BLK_SIZE, LEVELS>::new(metadata_ptr, region_ptr, num_blocks)
                    .with_backing_allocator(allocator),
            )
        }
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> Drop
    for BuddyAllocator<BLK_SIZE, LEVELS, A>
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

impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator>
    BuddyAllocator<BLK_SIZE, LEVELS, A>
{
    fn assert_const_param_invariants() {
        Self::min_block_size();
    }

    fn min_block_size() -> usize {
        assert!(LEVELS > 0, "buddy allocator must have at least one level");
        assert!(
            BLK_SIZE.is_power_of_two(),
            "buddy allocator block size must be a power of two"
        );
        assert!(
            LEVELS < usize::BITS as usize,
            "buddy allocator cannot have more levels than bits in a usize"
        );

        let min_block_size = BLK_SIZE >> (LEVELS - 1);
        assert!(
            min_block_size >= mem::size_of::<BlockLink>(),
            "buddy allocator minimum block size must be at least mem::size_of::<BlockLink>() bytes"
        );

        min_block_size
    }

    /// Returns the layout requirements of the region managed by an allocator of
    /// this type.
    ///
    /// # Panics
    ///
    /// This function panics if the
    pub fn region_layout(num_blocks: usize) -> Layout {
        let min_block_size = Self::min_block_size();
        let levels: u32 = LEVELS.try_into().unwrap();

        let size = 2usize.pow(levels - 1) * min_block_size * num_blocks;
        let align = min_block_size;

        Layout::from_size_align(size, align).unwrap()
    }

    /// Returns the layout requirements of the metadata region for an allocator
    /// of this type.
    pub fn metadata_layout(num_blocks: usize) -> Layout {
        const fn sum_of_powers_of_2(max: u32) -> usize {
            2 * (2_usize.pow(max) - 1)
        }

        Self::assert_const_param_invariants();

        let levels: u32 = LEVELS.try_into().unwrap();

        // Each level needs one buddy bit per pair of blocks.
        let num_pairs = (num_blocks + 1) / 2;

        // This is the layout required for the buddy bitmap of level 0.
        let buddy_l0_layout = Bitmap::map_layout(num_pairs);

        // Each subsequent level requires at most twice as much space as the
        // level 0 bitmap. It may require less if the number of level 0 blocks
        // is not a multiple of the bitmap block size, but for simplicity each
        // level is given exactly twice the space of the previous level.
        let (buddy_layout, _) = buddy_l0_layout
            .repeat(sum_of_powers_of_2(levels - 1))
            .unwrap();

        if LEVELS == 1 {
            // There's only one level, so no split bitmap is required.
            return buddy_layout;
        }

        // Each level except level (LEVELS - 1) needs one split bit per block.
        let split_l0_layout = Bitmap::map_layout(num_blocks);

        // Let K equal the size of a split bitmap for level 0. If LEVELS is:
        // - 2, then 1 split bitmap is needed of size (2 - 1)K = K.
        // - 3, then 2 split bitmaps are needed of total size (3 - 1)K + (2 - 1)K = 3K.
        // - ...
        // - N, then 2 ^ (N - 2) split bitmaps are needed of total size:
        //
        //     (N - 1)K + ((N - 1) - 1)K + ... + (2 - 1)K
        //   = 2 * (2 ^ (N - 1) - 1) * K
        //   = (sum from x = 1 to (LEVELS - 1) of 2^x) * K
        let (split_layout, _) = split_l0_layout
            .repeat(sum_of_powers_of_2(levels - 1))
            .unwrap();
        let (full_layout, _) = buddy_layout.extend(split_layout).unwrap();

        full_layout
    }

    fn level_for(&self, size: usize) -> Option<usize> {
        fn round_up_pow2(x: usize) -> Option<usize> {
            match x {
                0 => None,
                1 => Some(1),
                x if x >= (1 << 63) => None,
                _ => Some(2usize.pow((x - 1).log2() as u32 + 1)),
            }
        }

        let min_block_size = Self::min_block_size();

        let max_block_size = self.levels[0].block_size;
        if size > max_block_size {
            return None;
        }

        let alloc_size = cmp::max(round_up_pow2(size).unwrap(), min_block_size);
        let level: usize = (max_block_size.log2() - alloc_size.log2())
            .try_into()
            .unwrap();

        Some(level)
    }

    fn min_free_level(&self, block_ofs: usize) -> usize {
        let min_block_size = Self::min_block_size();

        let max_block_size = self.levels[0].block_size;

        if block_ofs == 0 {
            return 0;
        }

        let max_size = 1 << block_ofs.trailing_zeros();
        if max_size > max_block_size {
            return 0;
        }

        assert!(max_size >= min_block_size);

        (max_block_size.log2() - max_size.log2())
            .try_into()
            .unwrap()
    }

    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`] which satisfies `layout`.
    ///
    /// The contents of the block are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a suitable block could not be allocated.
    ///
    /// [`NonNull<[u8]: NonNull
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Err(AllocError);
        }

        let target_level = self.level_for(layout.size()).ok_or(AllocError)?;

        // If there is a free block of the correct size, return it immediately.
        if let Some(block) =
            unsafe { self.levels[target_level].allocate(self.base, layout.align()) }
        {
            return Ok(self.base.with_addr_and_size(block, layout.size()));
        }

        // Otherwise, scan increasing block sizes until a free block is found.
        let (block, init_level) = (0..target_level)
            .rev()
            .find_map(|level| unsafe {
                self.levels[level]
                    .allocate(self.base, layout.align())
                    .map(|block| (block, level))
            })
            .ok_or(AllocError)?;

        let block_ofs = self.base.offset_to(block);

        // Split the block repeatedly to obtain a suitably sized block.
        for level in init_level..target_level {
            // Split the block. The address of the front half does not change.
            let half_block_size = self.levels[level].block_size / 2;
            let back_half = NonZeroUsize::new(block.get() + half_block_size).unwrap();

            // Mark the block as split.
            let split_bit = self.levels[level].index_of(block_ofs);
            if let Some(s) = self.levels[level].splits.as_mut() {
                s.set(split_bit, true);
            }

            // Add one half of the split block to the next level's free list.
            unsafe { self.levels[level + 1].assign(self.base, back_half) };
        }

        // The returned block inherits the provenance of the base pointer.
        Ok(self.base.with_addr_and_size(block, layout.size()))
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// This is equivalent to `Self::deallocate()`, but without the `Layout`
    /// parameter.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        // Some addresses can't come from earlier levels because their addresses
        // imply a smaller block size.
        let block_ofs = self.base.offset_to(ptr.addr());
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

        let mut block = Some(ptr);
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

#[cfg(all(feature = "unstable"))]
use core::cell::RefCell;

#[cfg(all(feature = "unstable", feature = "std"))]
use std::sync::{Mutex, RwLock};

/// Like a `BuddyAllocator`, but without a `Drop` impl or an associated
/// allocator.
///
/// This assists in tacking on the allocator type parameter because this struct can be
/// moved out of, while `BuddyAllocator` itself cannot.
struct BuddyAllocatorParts<const BLK_SIZE: usize, const LEVELS: usize> {
    base: BasePtr,
    metadata: NonNull<u8>,
    num_blocks: usize,
    levels: [BuddyLevel; LEVELS],
}

impl<const BLK_SIZE: usize, const LEVELS: usize> BuddyAllocatorParts<BLK_SIZE, LEVELS> {
    fn with_backing_allocator<A: BackingAllocator>(
        self,
        backing_allocator: A,
    ) -> BuddyAllocator<BLK_SIZE, LEVELS, A> {
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
        metadata: NonNull<u8>,
        base: NonNull<u8>,
        num_blocks: usize,
    ) -> BuddyAllocatorParts<BLK_SIZE, LEVELS> {
        let min_block_size = BuddyAllocator::<BLK_SIZE, LEVELS, Raw>::min_block_size();

        let full_layout = BuddyAllocator::<BLK_SIZE, LEVELS, Raw>::metadata_layout(num_blocks);

        // TODO: use MaybeUninit::uninit_array when not feature gated
        let mut levels: [MaybeUninit<BuddyLevel>; LEVELS] = unsafe {
            // SAFETY: An uninitialized `[MaybeUninit<_>; _]` is valid.
            MaybeUninit::<[MaybeUninit<BuddyLevel>; LEVELS]>::uninit().assume_init()
        };

        let mut meta_curs = metadata.as_ptr();

        for (li, level) in levels.iter_mut().enumerate() {
            let block_size = 2_usize.pow((LEVELS - li) as u32 - 1) * min_block_size;
            let block_factor = 2_usize.pow(li as u32);
            let num_blocks = block_factor * num_blocks;
            let num_pairs = (num_blocks + 1) / 2;

            let buddy_size = Bitmap::map_layout(num_pairs).size();
            let buddy_bitmap = unsafe { Bitmap::new(num_pairs, meta_curs as *mut u64) };

            meta_curs = unsafe {
                meta_curs.offset(
                    buddy_size
                        .try_into()
                        .expect("buddy bitmap layout size overflows isize"),
                )
            };

            let split_bitmap = if li < LEVELS - 1 {
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
            // When `MaybeUninit::array_assume_init()` is stable, use that
            // instead.
            //
            // SAFETY:
            // - `levels` is fully initialized.
            // - `MaybeUninit<T>` and `T` have the same layout.
            // - `MaybeUninit<T>` won't drop `T`, so no double-frees.
            (&levels as *const _ as *const [BuddyLevel; LEVELS]).read()
        };

        let base = BasePtr { ptr: base };

        // Initialize the top-level free list by emplacing links in each block.
        for block_idx in 0..num_blocks {
            let block_offset = block_idx.checked_mul(levels[0].block_size).unwrap();

            let block_addr =
                NonZeroUsize::new(base.ptr.addr().get().checked_add(block_offset).unwrap())
                    .unwrap();

            // All blocks except the first link to the previous block.
            let prev = (block_idx > 0).then(|| {
                let prev_addr = block_addr.get().checked_sub(levels[0].block_size).unwrap();
                NonZeroUsize::new(prev_addr).unwrap()
            });

            // All blocks except the last link to the next block.
            let next = (block_idx < num_blocks - 1).then(|| {
                let next_addr = block_addr.get().checked_add(levels[0].block_size).unwrap();
                NonZeroUsize::new(next_addr).unwrap()
            });

            unsafe {
                base.init_link_at(block_addr, BlockLink { prev, next });
            }
        }

        levels[0].free_list = (num_blocks > 0).then(|| base.ptr.addr());

        BuddyAllocatorParts {
            base,
            metadata,
            num_blocks,
            levels,
        }
    }
}
