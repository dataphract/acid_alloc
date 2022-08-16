#![deny(unsafe_op_in_unsafe_fn)]

use std::{alloc::Layout, ops::Range, ptr::NonNull};

use acid_alloc::{AllocInitError, Buddy, Global};
use alloc_hater::Subject;

pub struct BuddySubject<const BLK_SIZE: usize, const LEVELS: usize>(
    Buddy<BLK_SIZE, LEVELS, Global>,
);

impl<const BLK_SIZE: usize, const LEVELS: usize> BuddySubject<BLK_SIZE, LEVELS> {
    pub fn new(num_blocks: usize) -> Result<Self, AllocInitError> {
        let b = Buddy::try_new(num_blocks)?;
        Ok(BuddySubject(b))
    }

    pub fn new_with_offset_gaps(
        num_blocks: usize,
        gaps: impl IntoIterator<Item = Range<usize>>,
    ) -> Result<Self, AllocInitError> {
        let b = Buddy::try_new_with_offset_gaps(num_blocks, gaps)?;
        Ok(BuddySubject(b))
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize> Subject for BuddySubject<BLK_SIZE, LEVELS> {
    type AllocError = acid_alloc::AllocError;

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, Self::AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _layout: std::alloc::Layout) {
        unsafe { self.0.deallocate(ptr) };
    }
}