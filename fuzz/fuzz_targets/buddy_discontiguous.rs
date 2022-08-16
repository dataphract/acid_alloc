#![no_main]
#![feature(allocator_api)]

use std::ops::Range;

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
    gaps: Vec<Range<usize>>,
    ops: Vec<AllocatorOp>,
}

impl Arbitrary<'_> for Args {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<Args> {
        let num_blocks = usize::arbitrary(un)? % MAX_BLOCKS;

        let gaps = {
            let mut v = Vec::<usize>::arbitrary(un)?;

            v = v
                .into_iter()
                .map(|ofs| ofs % (BLK_SIZE * num_blocks))
                .take(usize::arbitrary(un)? % 2 * num_blocks)
                .collect();

            v.sort();

            v.chunks_exact(2).map(|pair| pair[0]..pair[1]).collect()
        };

        let ops = Vec::arbitrary(un)?;

        Ok(Args {
            num_blocks,
            gaps,
            ops,
        })
    }
}

fuzz_target!(|args: Args| {
    let buddy: BuddySubject<BLK_SIZE, LEVELS> =
        match acid_alloc_hater::BuddySubject::new_with_offset_gaps(args.num_blocks, args.gaps) {
            Ok(b) => b,
            Err(_) => return,
        };

    let mut eval = alloc_hater::Evaluator::new(buddy);
    eval.evaluate(args.ops).unwrap();
});
