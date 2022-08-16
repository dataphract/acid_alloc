#![no_main]
#![feature(allocator_api)]

use acid_alloc_hater::BuddySubject;
use alloc_hater::AllocatorOp;
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

const BLK_SIZE: usize = 16384;
const LEVELS: usize = 8;

const MAX_BLOCKS: usize = 1024;

#[derive(Clone, Debug)]
struct Args {
    num_blocks: usize,
    ops: Vec<AllocatorOp>,
}

impl Arbitrary<'_> for Args {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<Args> {
        let num_blocks = usize::arbitrary(un)? % MAX_BLOCKS;
        let ops = Vec::arbitrary(un)?;

        Ok(Args { num_blocks, ops })
    }
}

fuzz_target!(|args: Args| {
    let buddy: BuddySubject<BLK_SIZE, LEVELS> = match BuddySubject::new(args.num_blocks) {
        Ok(a) => a,
        Err(_) => return,
    };

    let mut eval = alloc_hater::Evaluator::new(buddy);
    eval.evaluate(args.ops).unwrap();
});
