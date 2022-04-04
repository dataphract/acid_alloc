#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(alloc_layout_extra)]
#![feature(int_log)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(strict_provenance)]

mod bitmap;

use core::{
    alloc::Layout,
    cmp,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ptr::NonNull,
};

use crate::bitmap::Bitmap;

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
    // This avoids violating stacked borrows; the link "points to" the next block,
    // but by forgoing a pointer, it does not imply a borrow.
    //
    // NOTE: Any pointer to a block must be acquired via the allocator base
    // pointer, and NOT by casting this address directly!
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

        unsafe {
            let uninit_mut: &mut MaybeUninit<_> = ptr.as_mut();
            uninit_mut.as_mut_ptr().write(BuddyLink { next });
            NonZeroUsize::new(uninit_mut.as_mut_ptr().addr()).unwrap()
        }
    }
}

/// Calculates the offset from `base` to `block`.
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
            let mut cur_mut = unsafe { base.with_addr(cur).cast::<BuddyLink>().as_mut() };
            let next = match cur_mut.next {
                Some(n) => n,

                // Block not found in free list.
                None => break,
            };

            let prev = cur;
            cur = next;
            assert!(prev != cur);

            cur_mut = unsafe { base.with_addr(cur).cast::<BuddyLink>().as_mut() };

            if cur == block {
                unsafe {
                    // SAFETY: `prev` and `cur` must not be overlapping (which they shouldn't be).
                    let prev_mut = base.with_addr(prev).cast::<BuddyLink>().as_mut();

                    prev_mut.next = cur_mut.next.take();
                }
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

pub struct BuddyAllocator<const ORDER: usize, const PGSIZE: usize> {
    /// Pointer to the region managed by this allocator.
    base: BasePtr,
    /// Pointer to the region that backs the bitmaps.
    ///
    /// This must not be used while the allocator exists; it is stored solely so
    /// that it may be returned in `into_raw_parts()`.
    metadata: *mut u8,
    /// The number of zero-order blocks managed by this allocator.
    num_blocks: usize,
    levels: [BuddyLevel; ORDER],
}

const fn sum_of_powers_of_2(max: u32) -> usize {
    2 * (2_usize.pow(max) - 1)
}

impl<const ORDER: usize, const PGSIZE: usize> BuddyAllocator<ORDER, PGSIZE> {
    pub fn region_layout(num_blocks: usize) -> Layout {
        assert!(ORDER > 0);
        assert!(PGSIZE >= mem::size_of::<BuddyLink>() && PGSIZE.count_ones() == 1);
        let order: u32 = ORDER.try_into().unwrap();

        let size = 2usize.pow(order - 1) * PGSIZE * num_blocks;
        let align = PGSIZE;

        Layout::from_size_align(size, align).unwrap()
    }

    /// Returns the layout requirements for the metadata region.
    pub fn metadata_layout(num_blocks: usize) -> Layout {
        assert!(ORDER > 0);
        assert!(PGSIZE >= mem::size_of::<BuddyLink>() && PGSIZE.count_ones() == 1);
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

    pub unsafe fn new(
        base: NonNull<u8>,
        num_blocks: usize,
        metadata: *mut u8,
    ) -> BuddyAllocator<ORDER, PGSIZE> {
        let full_layout = Self::metadata_layout(num_blocks);

        // TODO: use MaybeUninit::uninit_array when not feature gated
        let mut levels: [MaybeUninit<BuddyLevel>; ORDER] = MaybeUninit::uninit_array();

        let mut meta_curs = metadata;

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
            unsafe { meta_curs.offset_from(metadata) } < full_layout.size().try_into().unwrap()
        );

        let mut levels = unsafe { MaybeUninit::array_assume_init(levels) };

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

        BuddyAllocator {
            base: BasePtr { ptr: base },
            metadata,
            num_blocks,
            levels,
        }
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
    pub unsafe fn into_raw_parts(self) -> (NonNull<u8>, *mut u8) {
        let BuddyAllocator { base, metadata, .. } = self;

        (base.ptr, metadata)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::slice;
    use std::prelude::rust_2021::*;

    use super::*;

    fn new_with_global<const ORDER: usize, const PGSIZE: usize>(
        num_blocks: usize,
    ) -> BuddyAllocator<ORDER, PGSIZE> {
        let region_layout = BuddyAllocator::<ORDER, PGSIZE>::region_layout(num_blocks);
        let metadata_layout = BuddyAllocator::<ORDER, PGSIZE>::metadata_layout(num_blocks);

        unsafe {
            let region_ptr = {
                let raw = std::alloc::alloc(region_layout);
                NonNull::new(raw).unwrap()
            };

            let metadata_ptr = std::alloc::alloc(metadata_layout);

            BuddyAllocator::<ORDER, PGSIZE>::new(region_ptr, num_blocks, metadata_ptr)
        }
    }

    unsafe fn free_with_global<const ORDER: usize, const PGSIZE: usize>(
        allocator: BuddyAllocator<ORDER, PGSIZE>,
    ) {
        let num_blocks = allocator.num_blocks;

        let region_layout = BuddyAllocator::<ORDER, PGSIZE>::region_layout(num_blocks);
        let metadata_layout = BuddyAllocator::<ORDER, PGSIZE>::metadata_layout(num_blocks);

        unsafe {
            let (region, metadata) = allocator.into_raw_parts();
            std::alloc::dealloc(region.as_ptr(), region_layout);
            std::alloc::dealloc(metadata, metadata_layout);
        }
    }

    #[test]
    fn create_and_destroy() {
        // These parameters give a maximum block size of 1KiB and a total size of 8KiB.
        const ORDER: usize = 8;
        const PGSIZE: usize = 8;
        const NUM_BLOCKS: usize = 8;

        let allocator = new_with_global::<ORDER, PGSIZE>(NUM_BLOCKS);
        unsafe { free_with_global(allocator) };
    }

    #[test]
    fn alloc_write_and_free() {
        const ORDER: usize = 8;
        const PGSIZE: usize = 8;
        const NUM_BLOCKS: usize = 8;

        let mut allocator = new_with_global::<ORDER, PGSIZE>(NUM_BLOCKS);

        unsafe {
            let size = 64;
            let ptr: NonNull<u8> = allocator.allocate(size).unwrap();

            {
                // Do this in a separate scope so that the slice no longer
                // exists when ptr is freed
                let buf: &mut [u8] = slice::from_raw_parts_mut(ptr.as_ptr(), size);
                for (i, byte) in buf.iter_mut().enumerate() {
                    *byte = i as u8;
                }
            }

            allocator.free(ptr);
        }

        unsafe { free_with_global(allocator) };
    }

    #[test]
    fn coalesce_one() {
        // This configuration gives a 2-level buddy allocator with one
        // splittable top-level block.
        const ORDER: usize = 2;
        const PGSIZE: usize = 8;
        const NUM_BLOCKS: usize = 1;

        let mut allocator = new_with_global::<ORDER, PGSIZE>(NUM_BLOCKS);

        unsafe {
            // Allocate two minimum-size blocks to split the top block.
            let a = allocator.allocate(PGSIZE).unwrap();
            let b = allocator.allocate(PGSIZE).unwrap();

            // Free both blocks, coalescing them.
            allocator.free(a);
            allocator.free(b);

            // Allocate the entire region to ensure coalescing worked.
            let c = allocator.allocate(2 * PGSIZE).unwrap();
            allocator.free(c);

            // Same as above.
            let a = allocator.allocate(PGSIZE).unwrap();
            let b = allocator.allocate(PGSIZE).unwrap();

            // Free both blocks, this time in reverse order.
            allocator.free(a);
            allocator.free(b);

            let c = allocator.allocate(2 * PGSIZE).unwrap();
            allocator.free(c);
        }

        unsafe { free_with_global(allocator) };
    }

    #[test]
    fn coalesce_many() {
        const ORDER: usize = 4;
        const PGSIZE: usize = 8;
        const NUM_BLOCKS: usize = 8;

        let mut allocator = new_with_global::<ORDER, PGSIZE>(NUM_BLOCKS);

        for o in (0..ORDER).rev() {
            let alloc_size = 2usize.pow((ORDER - o - 1) as u32) * PGSIZE;
            let num_allocs = 2usize.pow(o as u32) * NUM_BLOCKS;

            let mut allocs = Vec::with_capacity(num_allocs);
            for i in 0..num_allocs {
                let ptr = unsafe { allocator.allocate(alloc_size).unwrap() };
                std::println!("alloced {i}");

                {
                    // Do this in a separate scope so that the slice no longer
                    // exists when ptr is freed
                    let buf: &mut [u8] =
                        unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), alloc_size) };
                    for (i, byte) in buf.iter_mut().enumerate() {
                        *byte = (i % 256) as u8;
                    }
                }

                allocs.push(ptr);
            }

            for alloc in allocs {
                unsafe {
                    allocator.free(alloc);
                }
            }
        }

        unsafe { free_with_global(allocator) };
    }
}
