#![deny(unsafe_op_in_unsafe_fn)]

use std::{alloc::Layout, mem::ManuallyDrop, ops::Range, ptr::NonNull};

use acid_alloc::{AllocInitError, Buddy, Global, Raw};
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
    type Op = ();
    type AllocError = acid_alloc::AllocError;

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, Self::AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, _layout: std::alloc::Layout) {
        unsafe { self.0.deallocate(ptr) };
    }

    fn handle_custom_op(&mut self, (): ()) {}
}

pub struct RawBuddySubject<const BLK_SIZE: usize, const LEVELS: usize>(
    ManuallyDrop<Buddy<BLK_SIZE, LEVELS, Raw>>,
);

impl<const BLK_SIZE: usize, const LEVELS: usize> RawBuddySubject<BLK_SIZE, LEVELS> {
    pub fn new_unpopulated(num_blocks: usize) -> Result<Self, AllocInitError> {
        let metadata_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::metadata_layout(num_blocks)?;
        let region_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::region_layout(num_blocks)?;

        let metadata = NonNull::new(unsafe { std::alloc::alloc(metadata_layout) })
            .ok_or(AllocInitError::AllocFailed(metadata_layout))?;
        let region =
            NonNull::new(unsafe { std::alloc::alloc(region_layout) }).ok_or_else(|| {
                unsafe { std::alloc::dealloc(metadata.as_ptr(), metadata_layout) };
                AllocInitError::AllocFailed(region_layout)
            })?;

        let buddy = unsafe { Buddy::new_raw_unpopulated(metadata, region, num_blocks) }?;

        Ok(RawBuddySubject(ManuallyDrop::new(buddy)))
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize> Drop for RawBuddySubject<BLK_SIZE, LEVELS> {
    fn drop(&mut self) {
        unsafe {
            let raw = ManuallyDrop::take(&mut self.0);
            let parts = raw.into_raw_parts();
            std::alloc::dealloc(parts.metadata.as_ptr(), parts.metadata_layout);
            std::alloc::dealloc(parts.region.as_ptr(), parts.region_layout);
        }
    }
}
