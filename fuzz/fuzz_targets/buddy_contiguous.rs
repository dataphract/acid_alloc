#![no_main]
#![feature(allocator_api)]

use std::alloc::{Global, Layout};

use acid_alloc::Buddy;
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

const BLK_SIZE: usize = 64;
const LEVELS: usize = 8;

const MAX_BLOCKS: usize = 1024;
const MAX_ALIGN: usize = 4096;

// Arbitrary is not implemented for core::alloc::Layout :(
#[derive(Clone, Debug)]
struct FakeLayout {
    size: usize,
    align: usize,
}

impl Arbitrary<'_> for FakeLayout {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<FakeLayout> {
        let size = usize::arbitrary(un)?;

        // Select a random bit index and shift to obtain a power of two.
        let align_shift = u8::arbitrary(un)? % usize::BITS as u8;
        let align = 1 << align_shift;

        Ok(FakeLayout { size, align })
    }
}

#[derive(Clone, Debug, Arbitrary)]
enum BuddyOp {
    Allocate(FakeLayout),
    Deallocate(usize),
}

#[derive(Clone, Debug, Arbitrary)]
struct Args {
    num_blocks: usize,
    ops: Vec<BuddyOp>,
}

fuzz_target!(|args: Args| {
    let mut alloc: Buddy<BLK_SIZE, LEVELS, _> = match Buddy::try_new(args.num_blocks) {
        Ok(a) => a,
        Err(_) => return,
    };

    let mut outstanding = Vec::new();

    for op in args.ops {
        match op {
            BuddyOp::Allocate(fake_layout) => {
                let layout = Layout::from_size_align(fake_layout.size, fake_layout.align)
                    .expect("illegal layout values from FakeLayout");
                if let Ok(block) = alloc.allocate(layout) {
                    outstanding.push(block);
                }
            }

            BuddyOp::Deallocate(raw_idx) => {
                let idx = raw_idx % outstanding.len();
                let block = outstanding.swap_remove(idx);
                unsafe { alloc.deallocate(block.cast()) };
            }
        }
    }
});
