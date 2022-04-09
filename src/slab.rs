//! Slab allocation.
//!
//! A slab allocator divides the managed region into a fixed number of
//! equally-sized blocks. This design has limited flexibility, as the blocks
//! cannot be split or coalesced; all allocations have the same size.
//!
//! ## Characteristics
//!
//! #### Time complexity
//!
//! | Operation                | Best-case | Worst-case |
//! |--------------------------|-----------|------------|
//! | Allocate                 | O(1)      | O(1)       |
//! | Deallocate               | O(1)      | O(1)       |
//!
//! #### Fragmentation
//!
//! Due to slab allocators' fixed-size allocations, they exhibit no external
//! fragmentation. The degree of internal fragmentation is dependent on the
//! difference between the average allocation size and the allocator's block
//! size.

use crate::core::{
    alloc::{AllocError, Layout, LayoutError},
    fmt, mem,
    num::NonZeroUsize,
    ptr::NonNull,
};

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
use crate::Global;

#[cfg(feature = "unstable")]
use alloc::alloc::{Allocator, Global};

#[cfg(not(feature = "unstable"))]
use crate::core::ptr::NonNullStrict;

use crate::{layout_error, AllocInitError, BackingAllocator, BasePtr, BlockLink, Raw};

/// A slab allocator.
pub struct Slab<const BLK_SIZE: usize, A: BackingAllocator> {
    base: BasePtr,
    num_blocks: usize,
    free_list: Option<NonZeroUsize>,
    backing_allocator: A,
}

impl<const BLK_SIZE: usize> Slab<BLK_SIZE, Raw> {
    /// Constructs a new `Slab` from a raw pointer.
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
        unsafe { RawSlab::try_new(region, num_blocks).map(|s| s.with_backing_allocator(Raw)) }
    }
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
impl<const BLK_SIZE: usize> Slab<BLK_SIZE, Global> {
    /// Attempts to construct a new `Slab` backed by the global allocator.
    pub fn try_new(num_blocks: usize) -> Result<Slab<BLK_SIZE, Global>, AllocInitError> {
        let region_layout =
            Self::region_layout(num_blocks).map_err(|_| AllocInitError::InvalidConfig)?;

        unsafe {
            let region_raw = alloc::alloc::alloc(region_layout);
            let region_ptr =
                NonNull::new(region_raw).ok_or(AllocInitError::AllocFailed(region_layout))?;

            RawSlab::<BLK_SIZE>::try_new(region_ptr, num_blocks)
                .map(|s| s.with_backing_allocator(Global))
        }
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl<const BLK_SIZE: usize> Slab<BLK_SIZE, Global> {
    /// Attempts to construct a new `Slab` backed by the global allocator.
    pub fn try_new(num_blocks: usize) -> Result<Slab<BLK_SIZE, Global>, AllocInitError> {
        Self::try_new_in(num_blocks, Global)
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl<const BLK_SIZE: usize, A> Slab<BLK_SIZE, A>
where
    A: Allocator,
{
    /// Attempts to construct a new `Slab` backed by `backing_allocator`.
    pub fn try_new_in(
        num_blocks: usize,
        backing_allocator: A,
    ) -> Result<Slab<BLK_SIZE, A>, AllocInitError> {
        let region_layout =
            Self::region_layout(num_blocks).map_err(|_| AllocInitError::InvalidConfig)?;

        unsafe {
            let region_ptr = backing_allocator
                .allocate(region_layout)
                .map_err(|_| AllocInitError::AllocFailed(region_layout))?
                .cast();

            RawSlab::<BLK_SIZE>::try_new(region_ptr, num_blocks)
                .map(|s| s.with_backing_allocator(backing_allocator))
        }
    }
}

impl<const BLK_SIZE: usize, A: BackingAllocator> Slab<BLK_SIZE, A> {
    /// Attempts to allocate a fixed-size block from the slab.
    ///
    /// The returned block is guaranteed to be aligned to `BLK_SIZE`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no blocks are available.
    pub fn allocate_block(&mut self) -> Result<NonNull<[u8; BLK_SIZE]>, AllocError> {
        let old_head = self.free_list.ok_or(AllocError)?;

        unsafe {
            let link_mut = self.base.link_mut(old_head);
            self.free_list = link_mut.next.take();
        }

        let block_ptr = self.base.with_addr(old_head).cast::<[u8; BLK_SIZE]>();

        Ok(block_ptr)
    }

    /// Attempts to allocate a block of memory from the slab.
    ///
    /// The returned block is guaranteed to be aligned to `BLK_SIZE`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any of the following are true:
    /// - Either of `layout.size()` or `layout.align()` are greater than
    ///   `BLK_SIZE`.
    /// - No blocks are available.
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() > BLK_SIZE || layout.align() > BLK_SIZE {
            return Err(AllocError);
        }

        let block = self.allocate_block()?;

        Ok(self.base.with_addr_and_size(block.addr(), layout.size()))
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        let addr = ptr.addr();

        unsafe {
            self.base.link_mut(addr).next = self.free_list;
            self.free_list = Some(addr);
        }
    }
}

impl<const BLK_SIZE: usize, A: BackingAllocator> Slab<BLK_SIZE, A> {
    /// Returns the layout requirements of the region managed by a `Slab` of
    /// this type.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the total size and alignment of the region cannot be
    /// represented as a [`Layout`].
    pub fn region_layout(num_blocks: usize) -> Result<Layout, LayoutError> {
        let total_size = BLK_SIZE
            .checked_mul(num_blocks)
            .ok_or_else(|| layout_error())?;

        Layout::from_size_align(total_size, BLK_SIZE)
    }
}

impl<const BLK_SIZE: usize, A: BackingAllocator> fmt::Debug for Slab<BLK_SIZE, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slab")
            .field("base", &self.base.ptr)
            .field("BLK_SIZE", &BLK_SIZE)
            .finish()
    }
}

impl<const BLK_SIZE: usize, A: BackingAllocator> Drop for Slab<BLK_SIZE, A> {
    fn drop(&mut self) {
        // Safe unwrap: this layout was checked when the allocator was constructed.
        let region_layout = Self::region_layout(self.num_blocks).unwrap();
        unsafe {
            self.backing_allocator
                .deallocate(self.base.ptr, region_layout)
        }
    }
}

struct RawSlab<const BLK_SIZE: usize> {
    base: BasePtr,
    num_blocks: usize,
    free_list: Option<NonZeroUsize>,
}

impl<const BLK_SIZE: usize> RawSlab<BLK_SIZE> {
    /// Attempts to construct a new `Slab` from a raw pointer.
    ///
    /// # Safety
    ///
    /// `region` must be a pointer to a region that satisfies the [`Layout`]
    /// returned by [`Self::region_layout(num_blocks)`], and it must be valid
    /// for reads and writes for the entire size indicated by that `Layout`.
    unsafe fn try_new(
        region: NonNull<u8>,
        num_blocks: usize,
    ) -> Result<RawSlab<BLK_SIZE>, AllocInitError> {
        if BLK_SIZE < mem::size_of::<BlockLink>() {
            return Err(AllocInitError::InvalidConfig);
        }

        // Ensure the region size fits in a usize.
        let layout = Slab::<BLK_SIZE, Raw>::region_layout(num_blocks)
            .map_err(|_| AllocInitError::InvalidConfig)?;

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

        Ok(RawSlab {
            base,
            num_blocks,
            free_list: (num_blocks > 0).then(|| base.ptr.addr()),
        })
    }

    fn with_backing_allocator<A: BackingAllocator>(
        self,
        backing_allocator: A,
    ) -> Slab<BLK_SIZE, A> {
        Slab {
            base: self.base,
            num_blocks: self.num_blocks,
            free_list: self.free_list,
            backing_allocator,
        }
    }
}
