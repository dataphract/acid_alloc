#![deny(unsafe_op_in_unsafe_fn)]
#![no_main]
#![feature(allocator_api)]
#![feature(strict_provenance)]

use std::{ops::Range, ptr::NonNull};

use acid_alloc::{AllocInitError, Buddy, Raw};
use alloc_hater::{ArbLayout, Block, Blocks};
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

const BLK_SIZE: usize = 16384;
const LEVELS: usize = 8;

const MAX_BLOCKS: usize = 1024;

#[derive(Clone, Debug, Arbitrary)]
pub enum AllocatorOp {
    Allocate(ArbLayout),
    Deallocate(usize),
    AddRegion(usize),
}

#[derive(Clone, Debug)]
struct Args<const BLK_SIZE: usize, const LEVELS: usize> {
    num_blocks: usize,
    regions: Vec<Range<usize>>,
    ops: Vec<AllocatorOp>,
}

// Generates arbitrary, non-overlapping regions that can be added to an allocator.
fn regions<const BLK_SIZE: usize, const LEVELS: usize>(
    un: &mut Unstructured,
    num_blocks: usize,
) -> arbitrary::Result<Vec<Range<usize>>> {
    let min_block_size = Buddy::<BLK_SIZE, LEVELS, Raw>::min_block_size().unwrap();
    let num_atomic_blocks = (BLK_SIZE / min_block_size) * num_blocks;
    let num_regions = usize::arbitrary(un)? & num_atomic_blocks;

    let mut boundaries = Vec::with_capacity(num_regions);
    for _ in 0..num_regions {
        let bound = usize::arbitrary(un)? % num_regions;
        boundaries.push(bound);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    Ok(boundaries
        .windows(2)
        .map(|s| {
            let start = min_block_size.checked_mul(s[0]).unwrap();
            let end = min_block_size.checked_mul(s[1]).unwrap();
            start..end
        })
        .collect())
}

impl<const BLK_SIZE: usize, const LEVELS: usize> Arbitrary<'_> for Args<BLK_SIZE, LEVELS> {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<Args<BLK_SIZE, LEVELS>> {
        let num_blocks = usize::arbitrary(un)? % MAX_BLOCKS;
        let regions = regions::<BLK_SIZE, LEVELS>(un, num_blocks)?;

        let ops = Vec::arbitrary(un)?;

        Ok(Args {
            num_blocks,
            regions,
            ops,
        })
    }
}

fn create_buddy<const BLK_SIZE: usize, const LEVELS: usize>(
    num_blocks: usize,
) -> Result<Buddy<BLK_SIZE, LEVELS, Raw>, AllocInitError> {
    let metadata_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::metadata_layout(num_blocks)?;
    let region_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::region_layout(num_blocks)?;

    let metadata = NonNull::new(unsafe { std::alloc::alloc(metadata_layout) })
        .ok_or(AllocInitError::AllocFailed(metadata_layout))?;
    let region = NonNull::new(unsafe { std::alloc::alloc(region_layout) }).ok_or_else(|| {
        unsafe { std::alloc::dealloc(metadata.as_ptr(), metadata_layout) };
        AllocInitError::AllocFailed(region_layout)
    })?;

    unsafe { Buddy::new_raw_unpopulated(metadata, region, num_blocks) }
}

unsafe fn destroy_buddy<const BLK_SIZE: usize, const LEVELS: usize>(
    buddy: Buddy<BLK_SIZE, LEVELS, Raw>,
) {
    unsafe {
        let parts = buddy.into_raw_parts();
        std::alloc::dealloc(parts.metadata.as_ptr(), parts.metadata_layout);
        std::alloc::dealloc(parts.region.as_ptr(), parts.region_layout);
    }
}

fuzz_target!(|args: Args<BLK_SIZE, LEVELS>| {
    let Args {
        num_blocks,
        regions,
        ops,
    } = args;

    let mut buddy = match create_buddy::<BLK_SIZE, LEVELS>(num_blocks) {
        Ok(b) => b,
        Err(e) => return,
    };

    let base_addr = buddy.region().cast::<u8>().addr();
    let mut regions = regions
        .iter()
        .map(|range| {
            let start = base_addr.checked_add(range.start).unwrap();
            let end = base_addr.checked_add(range.end).unwrap();
            start..end
        })
        .collect::<Vec<_>>();

    let mut blocks = Blocks::new();

    let mut completed = Vec::new();

    for (op_id, op) in ops.into_iter().enumerate() {
        let op_id: u64 = op_id.try_into().unwrap();
        match op.clone() {
            AllocatorOp::Allocate(layout) => {
                let ptr = match buddy.allocate(layout.0) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                blocks.push(unsafe { Block::init(ptr, layout.0, op_id) });
            }

            AllocatorOp::Deallocate(idx) => {
                let mut block = match blocks.remove_modulo(idx) {
                    Some(b) => b,
                    None => continue,
                };

                unsafe {
                    if !block.verify() {
                        panic!("\nblock failed verification.\nnum blocks: {num_blocks}\ncompleted: {completed:?}\nfailed: {op:?}");
                    }

                    block.paint(op_id);
                }

                let (ptr, layout) = block.into_raw_parts();
                unsafe { buddy.deallocate(ptr.cast()) };
            }

            AllocatorOp::AddRegion(idx) => {
                let len = regions.len();

                if len == 0 {
                    continue;
                }

                let idx = idx % len;
                let region = regions.swap_remove(idx);
                unsafe { buddy.add_region(region) };
            }
        }

        completed.push(op);
    }

    unsafe { destroy_buddy(buddy) };
});
