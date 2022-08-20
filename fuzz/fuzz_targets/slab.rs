#![no_main]
use acid_alloc_hater::SlabSubject;
use alloc_hater::AllocatorOp;
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

const MAX_NUM_BLOCKS: usize = 1024;
const MAX_BLOCK_SIZE: usize = 1024;

#[derive(Clone, Debug)]
struct Args {
    block_size: usize,
    num_blocks: usize,
    ops: Vec<AllocatorOp>,
}

impl Arbitrary<'_> for Args {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<Args> {
        let block_size = usize::arbitrary(un)? % MAX_BLOCK_SIZE;
        let num_blocks = usize::arbitrary(un)? % MAX_NUM_BLOCKS;
        let ops = Vec::arbitrary(un)?;

        Ok(Args {
            block_size,
            num_blocks,
            ops,
        })
    }
}

fuzz_target!(|args: Args| {
    let Args {
        block_size,
        num_blocks,
        ops,
    } = args;

    let mut slab = match SlabSubject::new(args.block_size, args.num_blocks) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut eval = alloc_hater::Evaluator::new(slab);
    eval.evaluate(ops).unwrap();
});
