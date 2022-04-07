extern crate std;

use core::{alloc::Layout, ptr::NonNull, slice};
use std::prelude::rust_2021::*;

use quickcheck::{Arbitrary, Gen, QuickCheck};

use crate::buddy::BuddyAllocator;

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
use crate::Global;

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
use alloc::alloc::Global;

enum AllocatorOpTag {
    Allocate,
    Free,
}

#[derive(Clone, Debug)]
enum AllocatorOp {
    /// Allocate a buffer that can hold `len` `u32` values.
    Allocate { len: usize },
    /// Free an existing allocation.
    ///
    /// Given `n` outstanding allocations, the allocation to free is at index
    /// `index % n`.
    Free { index: usize },
}

struct Allocation {
    id: u32,
    ptr: *mut u32,
    len: usize,
}

/// Limit on allocation size, expressed in bits.
const ALLOC_LIMIT_BITS: u8 = 16;

impl Arbitrary for AllocatorOp {
    fn arbitrary(g: &mut Gen) -> Self {
        match g
            .choose(&[AllocatorOpTag::Allocate, AllocatorOpTag::Free])
            .unwrap()
        {
            AllocatorOpTag::Allocate => AllocatorOp::Allocate {
                len: {
                    // Try to distribute allocations evenly between powers of two.
                    let exp = u8::arbitrary(g) % (ALLOC_LIMIT_BITS + 1);
                    usize::arbitrary(g) % 2_usize.pow(exp.into())
                },
            },
            AllocatorOpTag::Free => AllocatorOp::Free {
                index: usize::arbitrary(g),
            },
        }
    }
}

#[test]
fn allocations_are_mutually_exclusive() {
    // This config produces an allocator over 65536 bytes with block sizes from
    // 16 to 1024.
    const LEVELS: usize = 7;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const BLOCKS: usize = 16;

    fn prop<const BLK_SIZE: usize, const LEVELS: usize>(ops: Vec<AllocatorOp>) -> bool {
        let mut alloc = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(BLOCKS);

        let mut allocations = Vec::with_capacity(ops.len());

        for (id, op) in ops.into_iter().enumerate() {
            match op {
                AllocatorOp::Allocate { len } => {
                    let layout = Layout::array::<u32>(len).unwrap();

                    let ptr = unsafe {
                        let ptr = match alloc.allocate(layout) {
                            Ok(p) => p.as_ptr().cast(),
                            Err(_) => continue,
                        };

                        let slice: &mut [u32] = slice::from_raw_parts_mut(ptr, len);
                        slice.fill(id as u32);
                        drop(slice);
                        ptr
                    };

                    allocations.push(Allocation {
                        id: id as u32,
                        ptr,
                        len,
                    });
                }

                AllocatorOp::Free { index } => {
                    if allocations.is_empty() {
                        continue;
                    }

                    let index = index % allocations.len();
                    let a = allocations.swap_remove(index);

                    unsafe {
                        let slice: &[u32] = slice::from_raw_parts(a.ptr, a.len);
                        if slice.iter().copied().any(|elem| elem != a.id as u32) {
                            return false;
                        }
                    }

                    unsafe { alloc.deallocate(NonNull::new(a.ptr.cast()).unwrap()) };
                }
            }
        }

        true
    }

    let mut qc = QuickCheck::new();
    qc.quickcheck(prop::<BLK_SIZE, LEVELS> as fn(_) -> bool);
}

#[test]
#[should_panic]
fn zero_levels_panics() {
    let _ = BuddyAllocator::<256, 0, Global>::new(8);
}

#[test]
#[should_panic]
fn too_many_levels_panics() {
    const LEVELS: usize = usize::BITS as usize;
    let _ = BuddyAllocator::<256, LEVELS, Global>::new(8);
}

#[test]
#[should_panic]
fn non_power_of_two_block_size_panics() {
    let _ = BuddyAllocator::<255, 4, Global>::new(8);
}

#[test]
#[should_panic]
fn too_small_min_block_size_panics() {
    const LEVELS: usize = 8;
    const MIN_SIZE: usize = core::mem::size_of::<usize>() / 2;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let _ = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);
}

#[test]
fn create_and_destroy() {
    // These parameters give a maximum block size of 1KiB and a total size of 8KiB.
    const LEVELS: usize = 8;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);
    drop(allocator);
}

#[test]
fn alloc_empty() {
    const LEVELS: usize = 4;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let mut allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);

    let layout = Layout::from_size_align(0, 1).unwrap();
    allocator.allocate(layout).unwrap_err();
}

#[test]
fn alloc_min_size() {
    const LEVELS: usize = 4;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let mut allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);

    let layout = Layout::from_size_align(1, 1).unwrap();
    let a = allocator.allocate(layout).unwrap();
    let _b = allocator.allocate(layout).unwrap();
    let c = allocator.allocate(layout).unwrap();
    unsafe {
        allocator.deallocate(a.cast());
        allocator.deallocate(c.cast());
    }
}

#[test]
fn alloc_write_and_free() {
    const LEVELS: usize = 8;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let mut allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);

    unsafe {
        let layout = Layout::from_size_align(64, MIN_SIZE).unwrap();
        let ptr: NonNull<u8> = allocator.allocate(layout).unwrap().cast();

        {
            // Do this in a separate scope so that the slice no longer
            // exists when ptr is freed
            let buf: &mut [u8] = slice::from_raw_parts_mut(ptr.as_ptr(), layout.size());
            for (i, byte) in buf.iter_mut().enumerate() {
                *byte = i as u8;
            }
        }

        allocator.deallocate(ptr);
    }
}

#[test]
fn coalesce_one() {
    // This configuration gives a 2-level buddy allocator with one
    // splittable top-level block.
    const LEVELS: usize = 2;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 1;

    let mut allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);

    let full_layout = Layout::from_size_align(2 * MIN_SIZE, MIN_SIZE).unwrap();
    let half_layout = Layout::from_size_align(MIN_SIZE, MIN_SIZE).unwrap();

    unsafe {
        // Allocate two minimum-size blocks to split the top block.
        let a = allocator.allocate(half_layout).unwrap();
        let b = allocator.allocate(half_layout).unwrap();

        // Free both blocks, coalescing them.
        allocator.deallocate(a.cast());
        allocator.deallocate(b.cast());

        // Allocate the entire region to ensure coalescing worked.
        let c = allocator.allocate(full_layout).unwrap();
        allocator.deallocate(c.cast());

        // Same as above.
        let a = allocator.allocate(half_layout).unwrap();
        let b = allocator.allocate(half_layout).unwrap();

        // Free both blocks, this time in reverse order.
        allocator.deallocate(a.cast());
        allocator.deallocate(b.cast());

        let c = allocator.allocate(full_layout).unwrap();
        allocator.deallocate(c.cast());
    }
}

#[test]
fn coalesce_many() {
    const LEVELS: usize = 4;
    const MIN_SIZE: usize = 16;
    const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
    const NUM_BLOCKS: usize = 8;

    let mut allocator = BuddyAllocator::<BLK_SIZE, LEVELS, Global>::new(NUM_BLOCKS);

    for lvl in (0..LEVELS).rev() {
        let alloc_size = 2usize.pow((LEVELS - lvl - 1) as u32) * MIN_SIZE;
        let layout = Layout::from_size_align(alloc_size, MIN_SIZE).unwrap();
        let num_allocs = 2usize.pow(lvl as u32) * NUM_BLOCKS;

        let mut allocs = Vec::with_capacity(num_allocs);
        for _ in 0..num_allocs {
            let ptr = allocator.allocate(layout).unwrap();

            {
                // Do this in a separate scope so that the slice no longer
                // exists when ptr is freed
                let buf: &mut [u8] =
                    unsafe { slice::from_raw_parts_mut(ptr.as_ptr().cast(), layout.size()) };
                for (i, byte) in buf.iter_mut().enumerate() {
                    *byte = (i % 256) as u8;
                }
            }

            allocs.push(ptr);
        }

        for alloc in allocs {
            unsafe {
                allocator.deallocate(alloc.cast());
            }
        }
    }
}
