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

use core::ptr;

use crate::core::{
    alloc::{AllocError, Layout, LayoutError},
    fmt, mem,
    num::NonZeroUsize,
    ptr::NonNull,
};

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
use alloc::alloc::Global;

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
use crate::{core::alloc::LayoutExt, Global};

#[cfg(feature = "unstable")]
use crate::core::alloc::Allocator;

#[cfg(not(feature = "unstable"))]
use crate::core::ptr::NonNullStrict;

use crate::{layout_error, AllocInitError, BackingAllocator, BasePtr, BlockLink, Raw};

/// A slab allocator.
pub struct Slab<A: BackingAllocator> {
    base: BasePtr,
    block_size: usize,
    num_blocks: usize,
    free_list: Option<NonZeroUsize>,
    backing_allocator: A,
}

impl Slab<Raw> {
    /// Constructs a new `Slab` from a raw pointer.
    ///
    /// For a discussion of slab allocation, see the [module-level
    /// documentation].
    ///
    /// # Errors
    ///
    /// Returns an error if the `Layout` indicated by
    /// [`Self::region_layout(num_blocks)`] would not fit between `region` and
    /// the end of the address space.
    ///
    /// # Safety
    ///
    /// `region` must be a pointer to a region that satisfies the [`Layout`]
    /// returned by [`Self::region_layout(num_blocks)`], and it must be valid
    /// for reads and writes for the entire size indicated by that `Layout`.
    ///
    /// [module-level documentation]: crate::slab
    /// [`Self::region_layout(num_blocks)`]: Slab::region_layout
    pub unsafe fn new_raw(
        region: NonNull<u8>,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Slab<Raw>, AllocInitError> {
        unsafe {
            RawSlab::try_new(region, block_size, num_blocks).map(|s| s.with_backing_allocator(Raw))
        }
    }
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
impl Slab<Global> {
    /// Attempts to construct a new `Slab` backed by the global allocator.
    ///
    /// In particular, the memory managed by this `Slab` is allocated from the
    /// global allocator according to the layout indicated by
    /// [`Self::region_layout(num_blocks)`].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    ///
    /// [`Self::region_layout(num_blocks)`]: Slab::region_layout
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn try_new(block_size: usize, num_blocks: usize) -> Result<Slab<Global>, AllocInitError> {
        let region_layout = Self::region_layout(block_size, num_blocks)
            .map_err(|_| AllocInitError::InvalidConfig)?;

        unsafe {
            let region_ptr = if region_layout.size() == 0 {
                region_layout.dangling()
            } else {
                // SAFETY: region size is not zero
                let region_raw = alloc::alloc::alloc(region_layout);
                NonNull::new(region_raw).ok_or_else(|| {
                    alloc::alloc::dealloc(region_raw, region_layout);
                    AllocInitError::AllocFailed(region_layout)
                })?
            };

            match RawSlab::try_new(region_ptr, block_size, num_blocks) {
                Ok(s) => Ok(s.with_backing_allocator(Global)),
                Err(e) => {
                    if region_layout.size() != 0 {
                        alloc::alloc::dealloc(region_ptr.as_ptr(), region_layout);
                    }

                    Err(e)
                }
            }
        }
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl Slab<Global> {
    /// Attempts to construct a new `Slab` backed by the global allocator.
    ///
    /// In particular, the memory managed by this `Slab` is allocated from the
    /// global allocator according to the layout indicated by
    /// [`Self::region_layout(num_blocks)`].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    ///
    /// [`Self::region_layout(num_blocks)`]: Slab::region_layout
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn try_new(block_size: usize, num_blocks: usize) -> Result<Slab<Global>, AllocInitError> {
        Self::try_new_in(block_size, num_blocks, Global)
    }
}

#[cfg(feature = "unstable")]
impl Slab<A>
where
    A: Allocator,
{
    /// Attempts to construct a new `Slab` backed by `backing_allocator`.
    ///
    /// In particular, the memory managed by this `Slab` is allocated from
    /// `backing_allocator` according to the layout indicated by
    /// [`Self::region_layout(num_blocks)`].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from
    /// `backing_allocator`.
    ///
    /// [`Self::region_layout(num_blocks)`]: Slab::region_layout
    #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
    pub fn try_new_in(
        block_size: usize,
        num_blocks: usize,
        backing_allocator: A,
    ) -> Result<Slab<A>, AllocInitError> {
        let region_layout =
            Self::region_layout(num_blocks).map_err(|_| AllocInitError::InvalidConfig)?;

        unsafe {
            let region_ptr = if region_layout.size() == 0 {
                region_layout.dangling()
            } else {
                backing_allocator
                    .allocate(region_layout)
                    .map_err(|_| AllocInitError::AllocFailed(region_layout))?
                    .cast()
            };

            match RawSlab::try_new(region_ptr, num_blocks) {
                Ok(s) => Ok(s.with_backing_allocator(backing_allocator)),
                Err(e) => {
                    if region_layout.size() != 0 {
                        backing_allocator.deallocate(region_ptr, region_layout);
                    }

                    return Err(e);
                }
            }
        }
    }
}

impl<A> Slab<A>
where
    A: BackingAllocator,
{
    /// Returns the layout requirements of the region managed by a `Slab` of
    /// this type.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the total size and alignment of the region cannot be
    /// represented as a [`Layout`].
    pub fn region_layout(block_size: usize, num_blocks: usize) -> Result<Layout, LayoutError> {
        let total_size = block_size
            .checked_mul(num_blocks)
            .ok_or_else(layout_error)?;

        let align = 1_usize
            .checked_shl(block_size.trailing_zeros())
            .ok_or(layout_error())?;

        Layout::from_size_align(total_size, align)
    }

    /// Attempts to allocate a block of memory from the slab.
    ///
    /// The returned block is guaranteed to be aligned to `1 <<
    /// block_size.trailing_zeros()` bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any of the following are true:
    /// - Either of `layout.size()` or `layout.align()` are greater than
    ///   `block_size`.
    /// - No blocks are available.
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() > self.block_size || layout.align() > self.block_size {
            return Err(AllocError);
        }

        let old_head = self.free_list.take().ok_or(AllocError)?;

        unsafe {
            let link_mut = self.base.link_mut(old_head);
            self.free_list = link_mut.next.take();
        }

        Ok(self.base.with_addr_and_size(old_head, layout.size()))
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        let addr = ptr.addr();

        unsafe {
            self.base.link_mut(addr).next = self.free_list;
            self.free_list = Some(addr);
        }
    }

    /// Returns a pointer to the managed region.
    ///
    /// It is undefined behavior to dereference the returned pointer or upgrade
    /// it to a reference if there are any outstanding allocations.
    pub fn region(&mut self) -> NonNull<[u8]> {
        NonNull::new(ptr::slice_from_raw_parts_mut(
            self.base.ptr.as_ptr(),
            self.num_blocks * self.block_size,
        ))
        .unwrap()
    }
}

impl<A> fmt::Debug for Slab<A>
where
    A: BackingAllocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slab")
            .field("base", &self.base.ptr)
            .field("block_size", &self.block_size)
            .field("num_blocks", &self.num_blocks)
            .finish()
    }
}

impl<A> Drop for Slab<A>
where
    A: BackingAllocator,
{
    fn drop(&mut self) {
        // Safe unwrap: this layout was checked when the allocator was constructed.
        let region_layout = Self::region_layout(self.block_size, self.num_blocks).unwrap();

        if region_layout.size() != 0 {
            unsafe {
                self.backing_allocator
                    .deallocate(self.base.ptr, region_layout)
            }
        }
    }
}

struct RawSlab {
    base: BasePtr,
    block_size: usize,
    num_blocks: usize,
    free_list: Option<NonZeroUsize>,
}

impl RawSlab {
    /// Attempts to construct a new `Slab` from a raw pointer.
    ///
    /// # Safety
    ///
    /// `region` must be a pointer to a region that satisfies the [`Layout`]
    /// returned by [`Self::region_layout(num_blocks)`], and it must be valid
    /// for reads and writes for the entire size indicated by that `Layout`.
    unsafe fn try_new(
        region: NonNull<u8>,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<RawSlab, AllocInitError> {
        if block_size < mem::size_of::<BlockLink>() {
            return Err(AllocInitError::InvalidConfig);
        }

        // Ensure the region size fits in a usize.
        let layout = Slab::<Raw>::region_layout(block_size, num_blocks)
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

        // Initialize the free list by emplacing links in each block.
        for block_addr in (region.addr().get()..region_end.get()).step_by(block_size) {
            // SAFETY: block_addr is a step between region.addr() and
            // region_end, both of which are nonzero.
            let block_addr = unsafe { NonZeroUsize::new_unchecked(block_addr) };

            // Safe unchecked sub: region_end is nonzero.
            let is_not_last = block_addr.get() < region_end.get() - block_size;

            let next = is_not_last.then(|| {
                // Safe unchecked add: block_addr is at least block_size less than
                // region_end, as region_end is the upper bound of the iterator
                // and block_size is the step.
                let next_addr = block_addr.get() + block_size;

                // SAFETY: next_addr is known to be greater than block_addr and
                // not to have overflown.
                unsafe { NonZeroUsize::new_unchecked(next_addr) }
            });

            unsafe { base.init_link_at(block_addr, BlockLink { next }) };
        }

        Ok(RawSlab {
            base,
            block_size,
            num_blocks,
            free_list: (num_blocks > 0).then(|| base.ptr.addr()),
        })
    }

    fn with_backing_allocator<A: BackingAllocator>(self, backing_allocator: A) -> Slab<A> {
        Slab {
            base: self.base,
            block_size: self.block_size,
            num_blocks: self.num_blocks,
            free_list: self.free_list,
            backing_allocator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_block_size_errors() {
        Slab::<Global>::try_new(0, 0).unwrap_err();
        Slab::<Global>::try_new(0, 1).unwrap_err();
        Slab::<Global>::try_new(1, 0).unwrap_err();
        Slab::<Global>::try_new(1, 1).unwrap_err();
        Slab::<Global>::try_new(mem::size_of::<BlockLink>() - 1, 0).unwrap_err();
        Slab::<Global>::try_new(mem::size_of::<BlockLink>() - 1, 1).unwrap_err();
    }

    #[test]
    fn overflow_address_space_errors() {
        Slab::<Global>::try_new(usize::MAX, 2).unwrap_err();
    }

    #[test]
    fn zero_blocks() {
        let mut slab = Slab::<Global>::try_new(128, 0).unwrap();
        slab.allocate(Layout::from_size_align(0, 1).unwrap())
            .unwrap_err();
    }

    #[test]
    fn one_block() {
        const BLK_SIZE: usize = 128;
        let mut slab = Slab::<Global>::try_new(BLK_SIZE, 1).unwrap();

        let layout = Layout::from_size_align(0, 1).unwrap();
        let b = slab.allocate(layout).unwrap();
        slab.allocate(layout).unwrap_err();
        unsafe { slab.deallocate(b.cast()) };

        slab.allocate(Layout::from_size_align(BLK_SIZE + 1, 1).unwrap())
            .unwrap_err();
    }

    #[test]
    fn two_blocks() {
        const BLK_SIZE: usize = 128;

        let mut slab = Slab::<Global>::try_new(BLK_SIZE, 2).unwrap();
        let layout = Layout::from_size_align(1, 1).unwrap();

        // Allocate a, b
        // Free a, b
        let a = slab.allocate(layout).unwrap();
        let b = slab.allocate(layout).unwrap();
        slab.allocate(layout).unwrap_err();
        unsafe {
            slab.deallocate(a.cast());
            slab.deallocate(b.cast());
        }

        // Allocate a, b
        // Free b, a
        let a = slab.allocate(layout).unwrap();
        let b = slab.allocate(layout).unwrap();
        slab.allocate(layout).unwrap_err();
        unsafe {
            slab.deallocate(b.cast());
            slab.deallocate(a.cast());
        }
    }
}
