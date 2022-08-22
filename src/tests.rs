#![cfg(test)]
extern crate std;

use core::{marker::PhantomData, mem, ops::Range, ptr};

use crate::{
    bump::Bump,
    core::{alloc::Layout, cmp, fmt::Debug, ptr::NonNull, slice},
    slab::Slab,
    AllocError, AllocInitError, Buddy, Global,
};

use alloc::{boxed::Box, vec::Vec};
use quickcheck::{Arbitrary, Gen, QuickCheck};

trait QcAllocator: Sized {
    type Params: Arbitrary + Debug;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError>;

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout);
}

// Slab =======================================================================

#[derive(Clone, Debug)]
struct SlabParams {
    block_size: usize,
    num_blocks: usize,
}

impl Arbitrary for SlabParams {
    fn arbitrary(g: &mut Gen) -> Self {
        SlabParams {
            block_size: cmp::max(mem::size_of::<usize>(), usize::arbitrary(g) % g.size()),
            num_blocks: usize::arbitrary(g) % g.size(),
        }
    }
}

impl QcAllocator for Slab<Global> {
    type Params = SlabParams;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError> {
        Slab::try_new(params.block_size, params.num_blocks)
    }

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout) {
        unsafe { self.deallocate(ptr) }
    }
}

// Buddy ======================================================================

#[derive(Clone, Debug)]
struct BuddyParams<const BLK_SIZE: usize> {
    num_blocks: usize,
    gaps: Vec<Range<usize>>,
}

impl<const BLK_SIZE: usize> Arbitrary for BuddyParams<BLK_SIZE> {
    fn arbitrary(g: &mut Gen) -> Self {
        //let num_blocks = cmp::max(usize::arbitrary(g) % 8, 1);
        let num_blocks = 2;

        let gaps = {
            let mut v = Vec::<usize>::arbitrary(g);

            v = v
                .into_iter()
                .map(|ofs| ofs % (BLK_SIZE * num_blocks))
                .take(usize::arbitrary(g) % 2 * num_blocks)
                .collect();

            v.sort();

            v.chunks_exact(2).map(|pair| pair[0]..pair[1]).collect()
        };

        BuddyParams { num_blocks, gaps }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let mut items = Vec::with_capacity(self.gaps.capacity() * 2);
        for i in 0..self.gaps.len() {
            items.push(BuddyParams {
                num_blocks: self.num_blocks,
                gaps: {
                    let mut v = self.gaps.clone();
                    v.remove(i);
                    v
                },
            });
            items.push(BuddyParams {
                num_blocks: self.num_blocks - 1,
                gaps: {
                    let mut v = self.gaps.clone();
                    v.remove(i);
                    v
                },
            });
        }

        Box::new(items.into_iter())
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize> QcAllocator for Buddy<BLK_SIZE, LEVELS, Global> {
    type Params = BuddyParams<BLK_SIZE>;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError> {
        Buddy::try_new_with_offset_gaps(params.num_blocks, params.gaps)
    }

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _: Layout) {
        unsafe { self.deallocate(ptr) }
    }
}

// Bump ======================================================================

#[derive(Clone, Debug)]
struct BumpParams {
    layout: Layout,
}

impl Arbitrary for BumpParams {
    fn arbitrary(g: &mut Gen) -> Self {
        BumpParams {
            layout: Layout::from_size_align(
                cmp::max(usize::arbitrary(g) % 8192, 1),
                1 << (usize::arbitrary(g) % 5),
            )
            .unwrap(),
        }
    }
}

impl QcAllocator for Bump<Global> {
    type Params = BumpParams;

    fn with_params(params: Self::Params) -> Result<Self, AllocInitError> {
        Bump::try_new(params.layout)
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
enum AllocatorOp<P: Arbitrary> {
    /// Allocate a buffer that can hold `len` `u32` values.
    Allocate { params: P },
    /// Free an existing allocation.
    ///
    /// Given `n` outstanding allocations, the allocation to free is at index
    /// `index % n`.
    Free { index: usize },
}

/// Limit on allocation size, expressed in bits.
const ALLOC_LIMIT_BITS: u8 = 16;

fn limited_size(g: &mut Gen) -> usize {
    let exp = u8::arbitrary(g) % (ALLOC_LIMIT_BITS + 1);
    usize::arbitrary(g) % 2_usize.pow(exp.into())
}

impl<P: Arbitrary> Arbitrary for AllocatorOp<P> {
    fn arbitrary(g: &mut Gen) -> Self {
        match g
            .choose(&[AllocatorOpTag::Allocate, AllocatorOpTag::Free])
            .unwrap()
        {
            AllocatorOpTag::Allocate => AllocatorOp::Allocate {
                params: P::arbitrary(g),
            },
            AllocatorOpTag::Free => AllocatorOp::Free {
                index: usize::arbitrary(g),
            },
        }
    }
}

type OpId = u32;

struct RawAllocation {
    id: OpId,
    ptr: NonNull<[u8]>,
    layout: Layout,
}

type AllocResult = Result<NonNull<[u8]>, AllocError>;

trait PropAllocation {
    type Params: Arbitrary;

    fn layout(params: &Self::Params) -> Layout;
    fn from_raw(params: &Self::Params, raw: RawAllocation) -> Self;
    fn into_raw(self) -> RawAllocation;
}

trait Prop {
    /// The allocator to test for this property.
    type Allocator: QcAllocator;

    type Allocation: PropAllocation;

    /// Examines the result of an allocation.
    fn post_allocate(
        op_id: OpId,
        params: &<Self::Allocation as PropAllocation>::Params,
        res: &mut AllocResult,
    ) -> bool {
        let _ = (op_id, params, res);
        true
    }

    fn pre_deallocate(allocation: &Self::Allocation) -> bool {
        let _ = allocation;
        true
    }

    fn check(
        params: <Self::Allocator as QcAllocator>::Params,
        ops: Vec<AllocatorOp<<Self::Allocation as PropAllocation>::Params>>,
    ) -> bool;
}

struct AllocatorChecker<P: Prop> {
    allocator: P::Allocator,
    allocations: Vec<P::Allocation>,
    num_ops: u32,
}

impl<P: Prop> AllocatorChecker<P> {
    fn new(
        params: <P::Allocator as QcAllocator>::Params,
        capacity: usize,
    ) -> Result<Self, AllocInitError> {
        Ok(AllocatorChecker {
            allocator: P::Allocator::with_params(params)?,
            allocations: Vec::with_capacity(capacity),
            num_ops: 0,
        })
    }

    fn do_op(&mut self, op: AllocatorOp<<P::Allocation as PropAllocation>::Params>) -> bool {
        let op_id = self.num_ops;
        self.num_ops += 1;

        match op {
            AllocatorOp::Allocate { params } => {
                let layout = P::Allocation::layout(&params);
                let mut res = self.allocator.allocate(layout);

                if !P::post_allocate(op_id, &params, &mut res) {
                    return false;
                }

                match res {
                    Ok(ptr) => {
                        self.allocations.push(P::Allocation::from_raw(
                            &params,
                            RawAllocation {
                                id: op_id,
                                ptr,
                                layout,
                            },
                        ));
                    }

                    // If the allocation should have succeeded, this is handled
                    // by post_allocate
                    Err(AllocError) => (),
                }
            }

            AllocatorOp::Free { index } => {
                if self.allocations.is_empty() {
                    return true;
                }

                let index = index % self.allocations.len();
                let a = self.allocations.swap_remove(index);

                if !P::pre_deallocate(&a) {
                    return false;
                }

                let a = a.into_raw();

                unsafe { self.allocator.deallocate(a.ptr.cast::<u8>(), a.layout) };
            }
        }

        true
    }

    fn run(&mut self, ops: Vec<AllocatorOp<<P::Allocation as PropAllocation>::Params>>) -> bool {
        if !ops.into_iter().all(|op| self.do_op(op)) {
            return false;
        }

        // Free any outstanding allocations.
        for alloc in self.allocations.drain(..) {
            let alloc = alloc.into_raw();
            unsafe {
                self.allocator
                    .deallocate(alloc.ptr.cast::<u8>(), alloc.layout)
            };
        }

        true
    }
}

// Miri is substantially slower to run property tests, so the number of test
// cases is reduced to keep the runtime in check.

#[cfg(not(miri))]
const MAX_TESTS: u64 = 100;

#[cfg(miri)]
const MAX_TESTS: u64 = 20;

struct MutuallyExclusive<A: QcAllocator> {
    phantom: PhantomData<A>,
}

struct MutuallyExclusiveAllocation {
    op_id: OpId,
    ptr: NonNull<[u32]>,
    layout: Layout,
}

#[derive(Clone, Debug)]
struct MutuallyExclusiveAllocationParams {
    len: usize,
}

impl Arbitrary for MutuallyExclusiveAllocationParams {
    fn arbitrary(g: &mut Gen) -> Self {
        MutuallyExclusiveAllocationParams {
            len: limited_size(g),
        }
    }
}

impl PropAllocation for MutuallyExclusiveAllocation {
    type Params = MutuallyExclusiveAllocationParams;

    fn layout(params: &Self::Params) -> Layout {
        Layout::array::<u32>(params.len).unwrap()
    }

    fn from_raw(params: &Self::Params, raw: RawAllocation) -> Self {
        MutuallyExclusiveAllocation {
            op_id: raw.id,
            ptr: NonNull::new(ptr::slice_from_raw_parts_mut(
                raw.ptr.as_ptr().cast(),
                params.len,
            ))
            .unwrap(),
            layout: raw.layout,
        }
    }

    fn into_raw(self) -> RawAllocation {
        // TODO: use size_of_val_raw when stable
        let num_bytes = mem::size_of::<u32>() * unsafe { self.ptr.as_ref().len() };

        let bytes = NonNull::new(ptr::slice_from_raw_parts_mut(
            self.ptr.cast().as_ptr(),
            num_bytes,
        ))
        .unwrap();

        RawAllocation {
            id: self.op_id,
            ptr: bytes,
            layout: self.layout,
        }
    }
}

impl<A: QcAllocator> Prop for MutuallyExclusive<A> {
    type Allocator = A;

    type Allocation = MutuallyExclusiveAllocation;

    fn check(
        params: A::Params,
        ops: Vec<AllocatorOp<<MutuallyExclusiveAllocation as PropAllocation>::Params>>,
    ) -> bool {
        let mut checker: AllocatorChecker<MutuallyExclusive<A>> =
            AllocatorChecker::new(params, ops.capacity()).unwrap();
        checker.run(ops)
    }

    fn post_allocate(
        op_id: OpId,
        params: &MutuallyExclusiveAllocationParams,
        res: &mut AllocResult,
    ) -> bool {
        if let Ok(alloc) = res {
            let u32_ptr: NonNull<u32> = alloc.cast();
            let slice = unsafe { slice::from_raw_parts_mut(u32_ptr.as_ptr(), params.len) };
            slice.fill(op_id);
        }

        true
    }

    fn pre_deallocate(allocation: &Self::Allocation) -> bool {
        let slice = unsafe { allocation.ptr.as_ref() };
        slice.iter().copied().all(|elem| elem == allocation.op_id)
    }
}

fn check<P: Prop>(
    params: <P::Allocator as QcAllocator>::Params,
    ops: Vec<AllocatorOp<<P::Allocation as PropAllocation>::Params>>,
) -> bool {
    let mut checker: AllocatorChecker<P> = AllocatorChecker::new(params, ops.capacity()).unwrap();
    checker.run(ops)
}

#[test]
fn slab_allocations_are_mutually_exclusive() {
    let mut qc = QuickCheck::new().max_tests(MAX_TESTS);
    qc.quickcheck(check::<MutuallyExclusive<Slab<Global>>> as fn(_, _) -> bool);
}

#[test]
fn buddy_allocations_are_mutually_exclusive() {
    let mut qc = QuickCheck::new().max_tests(MAX_TESTS);
    qc.quickcheck(check::<MutuallyExclusive<Buddy<16, 1, Global>>> as fn(_, _) -> bool);
    qc.quickcheck(check::<MutuallyExclusive<Buddy<128, 2, Global>>> as fn(_, _) -> bool);
    qc.quickcheck(check::<MutuallyExclusive<Buddy<1024, 4, Global>>> as fn(_, _) -> bool);
    qc.quickcheck(check::<MutuallyExclusive<Buddy<4096, 8, Global>>> as fn(_, _) -> bool);
}

#[test]
fn bump_allocations_are_mutually_exclusive() {
    let mut qc = QuickCheck::new().max_tests(MAX_TESTS);
    qc.quickcheck(check::<MutuallyExclusive<Bump<Global>>> as fn(_, _) -> bool);
}

// Version sync ================================================================
#[test]
fn html_root_url() {
    version_sync::assert_html_root_url_updated!("src/lib.rs");
}
