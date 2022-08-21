#![no_main]
use std::alloc::Layout;

use acid_alloc_hater::BumpSubject;
use alloc_hater::{AllocatorOp, ArbLayout};
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

const MAX_SIZE: usize = 64 * 1024;
const MAX_ALIGN_SHIFT: u8 = 12; // 4096 bytes

#[derive(Clone, Debug)]
struct Args {
    layout: Layout,
    ops: Vec<AllocatorOp>,
}

impl Arbitrary<'_> for Args {
    fn arbitrary(un: &mut Unstructured) -> arbitrary::Result<Args> {
        let size = usize::arbitrary(un)? % MAX_SIZE;
        let align_shift = u8::arbitrary(un)? % MAX_ALIGN_SHIFT;
        let align = 1_usize << align_shift;
        let layout = Layout::from_size_align(size, align).unwrap();
        let ops = Vec::arbitrary(un)?;

        Ok(Args { layout, ops })
    }
}

fuzz_target!(|args: Args| {
    let Args { layout, ops } = args;

    let mut bump = match BumpSubject::new(layout) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut eval = alloc_hater::Evaluator::new(bump);
    eval.evaluate(ops).unwrap();
});
