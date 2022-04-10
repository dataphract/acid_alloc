extern crate std;

use crate::{
    core::{alloc::Layout, cmp, fmt::Debug, ptr::NonNull, slice},
    slab::Slab,
    AllocError, AllocInitError, Buddy, Global,
};

#[cfg(not(feature = "unstable"))]
use crate::core::alloc::LayoutExt;

use alloc::{boxed::Box, vec::Vec};
use quickcheck::{Arbitrary, Gen, QuickCheck};

trait QcAllocator: Sized {
    type Params: Arbitrary + Debug;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError>;

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout);
}

#[derive(Clone, Debug)]
struct SlabParams {
    num_blocks: usize,
}

impl Arbitrary for SlabParams {
    fn arbitrary(g: &mut Gen) -> Self {
        SlabParams {
            num_blocks: usize::arbitrary(g) % g.size(),
        }
    }
}

impl<const BLK_SIZE: usize> QcAllocator for Slab<BLK_SIZE, Global> {
    type Params = SlabParams;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError> {
        Slab::try_new(params.num_blocks)
    }

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout) {
        unsafe { self.deallocate(ptr) }
    }
}

#[derive(Clone, Debug)]
struct BuddyParams {
    num_blocks: usize,
}

impl Arbitrary for BuddyParams {
    fn arbitrary(g: &mut Gen) -> Self {
        BuddyParams {
            num_blocks: cmp::max(usize::arbitrary(g) % g.size(), 1),
        }
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize> QcAllocator for Buddy<BLK_SIZE, LEVELS, Global> {
    type Params = BuddyParams;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError> {
        Buddy::try_new(params.num_blocks)
    }

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout) {
        unsafe { self.deallocate(ptr) }
    }
}

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

struct Allocation {
    id: u32,
    layout: Layout,
    ptr: NonNull<[u8]>,
    // Length with regard to the type passed to self.set_layout_of::<T>()
    len: usize,
}

struct AllocatorChecker<A: QcAllocator> {
    allocator: A,
    allocations: Vec<Allocation>,
    num_ops: u32,
    // Layout of single items.
    layout: Layout,
    post_allocate_hook: Option<Box<dyn Fn(u32, usize, &mut AllocResult) -> bool>>,
    pre_deallocate_hook: Option<Box<dyn Fn(&Allocation) -> bool>>,
}

type AllocResult = Result<NonNull<[u8]>, AllocError>;

impl<A: QcAllocator> AllocatorChecker<A> {
    fn new(params: A::Params, capacity: usize) -> Result<Self, AllocInitError> {
        Ok(AllocatorChecker {
            allocator: A::with_params(params)?,
            allocations: Vec::with_capacity(capacity),
            num_ops: 0,
            layout: Layout::new::<u8>(),
            post_allocate_hook: None,
            pre_deallocate_hook: None,
        })
    }

    fn set_layout_of<T>(&mut self) {
        self.layout = Layout::new::<T>();
    }

    fn set_post_allocate_hook(
        &mut self,
        hook: impl Fn(u32, usize, &mut AllocResult) -> bool + 'static,
    ) {
        self.post_allocate_hook = Some(Box::new(hook));
    }

    fn set_pre_deallocate_hook(&mut self, hook: impl Fn(&Allocation) -> bool + 'static) {
        self.pre_deallocate_hook = Some(Box::new(hook));
    }

    fn do_op(&mut self, op: AllocatorOp) -> bool {
        let op_id = self.num_ops;
        self.num_ops += 1;

        match op {
            AllocatorOp::Allocate { len } => {
                let layout = self.layout.repeat(len).unwrap().0;
                let mut res = self.allocator.allocate(layout);

                if !self
                    .post_allocate_hook
                    .as_ref()
                    .map(|f| f(op_id, len, &mut res))
                    .unwrap_or(true)
                {
                    return false;
                }

                match res {
                    Ok(ptr) => {
                        self.allocations.push(Allocation {
                            id: op_id,
                            layout,
                            ptr,
                            len,
                        });
                    }

                    // If the allocation should have succeeded, this is handled
                    // by post_allocate_hook
                    Err(AllocError) => (),
                }
            }

            AllocatorOp::Free { index } => {
                if self.allocations.is_empty() {
                    return true;
                }

                let index = index % self.allocations.len();
                let a = self.allocations.swap_remove(index);

                if !self
                    .pre_deallocate_hook
                    .as_ref()
                    .map(|f| f(&a))
                    .unwrap_or(true)
                {
                    return false;
                }

                unsafe { self.allocator.deallocate(a.ptr.cast::<u8>(), a.layout) };
            }
        }

        true
    }

    fn run(&mut self, ops: Vec<AllocatorOp>) -> bool {
        ops.into_iter().all(|op| self.do_op(op))
    }
}

// Miri is substantially slower to run property tests, so the number of test
// cases is reduced to keep the runtime in check.

#[cfg(not(miri))]
const MAX_TESTS: u64 = 100;

#[cfg(miri)]
const MAX_TESTS: u64 = 20;

fn allocations_are_mutually_exclusive<A>(params: A::Params, ops: Vec<AllocatorOp>) -> bool
where
    A: QcAllocator,
{
    let mut checker = AllocatorChecker::<A>::new(params, ops.capacity()).unwrap();
    checker.set_layout_of::<u32>();
    checker.set_post_allocate_hook(|id, len, res| {
        if let Ok(alloc) = res {
            let u32_ptr: NonNull<u32> = alloc.cast();
            let slice = unsafe { slice::from_raw_parts_mut(u32_ptr.as_ptr(), len) };
            slice.fill(id);
        }

        true
    });
    checker.set_pre_deallocate_hook(|alloc| {
        let u32_ptr: NonNull<u32> = alloc.ptr.cast();
        let slice = unsafe { slice::from_raw_parts(u32_ptr.as_ptr(), alloc.len) };
        slice.iter().copied().all(|elem| elem == alloc.id)
    });

    checker.run(ops)
}

#[test]
fn slab_allocations_are_mutually_exclusive() {
    let mut qc = QuickCheck::new().max_tests(MAX_TESTS);
    qc.quickcheck(allocations_are_mutually_exclusive::<Slab<8, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Slab<16, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Slab<32, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Slab<64, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Slab<128, Global>> as fn(_, _) -> bool);
}

#[test]
fn buddy_allocations_are_mutually_exclusive() {
    let mut qc = QuickCheck::new().max_tests(MAX_TESTS);
    qc.quickcheck(allocations_are_mutually_exclusive::<Buddy<16, 1, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Buddy<128, 2, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Buddy<1024, 4, Global>> as fn(_, _) -> bool);
    qc.quickcheck(allocations_are_mutually_exclusive::<Buddy<4096, 8, Global>> as fn(_, _) -> bool);
}
