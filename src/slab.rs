//! A slab allocator.

use core::{
    alloc::{Layout, LayoutError},
    mem,
    num::NonZeroUsize,
    ptr::NonNull,
};

use crate::{layout_error, AllocInitError, BackingAllocator, BasePtr, BlockLink, Raw};

#[cfg(not(feature = "unstable"))]
use crate::polyfill::*;

pub struct Slab<const BLK_SIZE: usize, A: BackingAllocator> {
    base: BasePtr,
    free_list: Option<NonZeroUsize>,
    backing_allocator: A,
}

impl<const BLK_SIZE: usize, A: BackingAllocator> Slab<BLK_SIZE, A> {
    /// Construct a new `Slab` from a raw pointer.
    ///
    /// # Safety
    ///
    /// `region` must be a pointer to a region that satisfies the [`Layout`]
    /// returned by [`Self::region_layout(num_blocks)`], and it must be valid
    /// for reads and writes for the entire size indicated by that `Layout`.
    pub unsafe fn new_raw(
        region: NonNull<u8>,
        num_blocks: usize,
    ) -> Result<Slab<BLK_SIZE, Raw>, AllocInitError> {
        if BLK_SIZE < mem::size_of::<BlockLink>() {
            return Err(AllocInitError::InvalidConfig);
        }

        // Ensure the region size fits in a usize.
        let layout = Self::region_layout(num_blocks).map_err(|_| AllocInitError::InvalidConfig)?;

        // Ensure pointer calculations will not overflow.
        // TODO: use checked_add directly on region.addr() when nonzero_ops is stable.
        let region_end = region
            .addr()
            .get()
            .checked_add(layout.size())
            .ok_or(AllocInitError::InvalidLocation)?;

        // SAFETY: region_end was calculated via checked addition on the address
        // of a NonNull.
        let region_end = unsafe { NonZeroUsize::new_unchecked(region_end) };

        let base = BasePtr { ptr: region };

        for block_addr in (region.addr().get()..region_end.get()).step_by(BLK_SIZE) {
            // SAFETY: block_addr is a step between region.addr() and
            // region_end, both of which are nonzero.
            let block_addr = unsafe { NonZeroUsize::new_unchecked(block_addr) };

            // Safe unchecked sub: region_end is nonzero.
            let is_not_last = block_addr.get() < region_end.get() - 1;

            let next = is_not_last.then(|| {
                // Safe unchecked add: block_addr is at least BLK_SIZE less than
                // region_end, as region_end is the upper bound of the iterator
                // and BLK_SIZE is the step.
                let next_addr = block_addr.get() + BLK_SIZE;

                // SAFETY: next_addr is known to be greater than block_addr and
                // not to have overflown.
                unsafe { NonZeroUsize::new_unchecked(next_addr) }
            });

            unsafe { base.init_link_at(block_addr, BlockLink { next }) };
        }

        Ok(Slab {
            base,
            free_list: (num_blocks > 0).then(|| base.ptr.addr()),
            backing_allocator: Raw,
        })
    }
}

impl<const BLK_SIZE: usize, A: BackingAllocator> Slab<BLK_SIZE, A> {
    /// Returns the layout requirements of the region managed by a `Slab` of
    /// this type.
    pub fn region_layout(num_blocks: usize) -> Result<Layout, LayoutError> {
        let total_size = BLK_SIZE
            .checked_mul(num_blocks)
            .ok_or_else(|| layout_error())?;

        Layout::from_size_align(total_size, BLK_SIZE)
    }
}
