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
    alloc::{AllocError, Layout},
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

use crate::{AllocInitError, BackingAllocator, BasePtr, BlockLink, Raw};

/// A slab allocator.
///
/// For a general discussion of slab allocation, see the [module-level
/// documentation].
///
/// # Configuration
///
/// Each constructor takes two parameters:
/// - `block_size` is the size in bytes of each allocation.
/// - `num_blocks` is the maximum number of allocations that a `Slab` may make
///   at once.
///
/// The minimum alignment of each allocated block is determined by
/// `block_size.trailing_zeros()`.
///
/// [module-level documentation]: crate::slab
pub struct Slab<A: BackingAllocator> {
    base: BasePtr,
    free_list: Option<NonZeroUsize>,
    block_size: u32,
    block_align: u32,
    num_blocks: u32,
    outstanding: u32,
    backing_allocator: A,
}

unsafe impl<A> Send for Slab<A> where A: BackingAllocator + Send {}
unsafe impl<A> Sync for Slab<A> where A: BackingAllocator + Sync {}

impl Slab<Raw> {
    /// Constructs a new `Slab` from a raw pointer to a region of memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the `Layout` indicated by
    /// [`Self::region_layout()`][0] would not fit between `region` and the end
    /// of the address space.
    ///
    /// # Safety
    ///
    /// `region` must be a pointer to a region that fits the [`Layout`] returned
    /// by [`Self::region_layout()`][0], and it must be valid for reads and
    /// writes for the entire size indicated by that `Layout`.
    ///
    /// [module-level documentation]: crate::slab
    /// [0]: Slab::region_layout
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
    /// [`Self::region_layout(block_size, num_blocks)`][0].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    ///
    /// [0]: Slab::region_layout
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
    /// [`Self::region_layout(block_size, num_blocks)`][0].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    ///
    /// [0]: Slab::region_layout
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn try_new(block_size: usize, num_blocks: usize) -> Result<Slab<Global>, AllocInitError> {
        Self::try_new_in(block_size, num_blocks, Global)
    }
}

#[cfg(feature = "unstable")]
impl<A> Slab<A>
where
    A: Allocator,
{
    /// Attempts to construct a new `Slab` backed by `backing_allocator`.
    ///
    /// In particular, the memory managed by this `Slab` is allocated from
    /// `backing_allocator` according to the layout indicated by
    /// [`Self::region_layout(num_blocks)`][0].
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from
    /// `backing_allocator`.
    ///
    /// [0]: Slab::region_layout
    #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
    pub fn try_new_in(
        block_size: usize,
        num_blocks: usize,
        backing_allocator: A,
    ) -> Result<Slab<A>, AllocInitError> {
        let region_layout = Self::region_layout(block_size, num_blocks)
            .map_err(|_| AllocInitError::InvalidConfig)?;

        unsafe {
            let region_ptr = if region_layout.size() == 0 {
                region_layout.dangling()
            } else {
                backing_allocator
                    .allocate(region_layout)
                    .map_err(|_| AllocInitError::AllocFailed(region_layout))?
                    .cast()
            };

            match RawSlab::try_new(region_ptr, block_size, num_blocks) {
                Ok(s) => Ok(s.with_backing_allocator(backing_allocator)),
                Err(e) => {
                    if region_layout.size() != 0 {
                        backing_allocator.deallocate(region_ptr, region_layout);
                    }

                    Err(e)
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
    pub fn region_layout(block_size: usize, num_blocks: usize) -> Result<Layout, AllocInitError> {
        let total_size = block_size
            .checked_mul(num_blocks)
            .ok_or(AllocInitError::InvalidConfig)?;
        u32::try_from(total_size).map_err(|_| AllocInitError::InvalidConfig)?;

        let align = 1_usize
            .checked_shl(block_size.trailing_zeros())
            .ok_or(AllocInitError::InvalidConfig)?;
        u32::try_from(align).map_err(|_| AllocInitError::InvalidConfig)?;

        Layout::from_size_align(total_size, align).map_err(|_| AllocInitError::InvalidConfig)
    }

    /// Attempts to allocate a block of memory with the specified layout.
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
        if layout.size() > self.block_size as usize || layout.align() > self.block_align as usize {
            return Err(AllocError);
        }

        let old_head = self.free_list.take().ok_or(AllocError)?;

        unsafe {
            let link_mut = self.base.link_mut(old_head);
            self.free_list = link_mut.next.take();
        }

        self.outstanding += 1;

        Ok(self.base.with_addr_and_size(old_head, layout.size()))
    }

    /// Attempts to allocate a block of memory from the slab.
    ///
    /// The layout of the returned block is determined by the allocator
    /// configuration. The returned block is guaranteed to be aligned to `1 <<
    /// block_size.trailing_zeros()` bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no blocks are available.
    pub fn allocate_block(&mut self) -> Result<NonNull<[u8]>, AllocError> {
        let old_head = self.free_list.take().ok_or(AllocError)?;

        unsafe {
            let link_mut = self.base.link_mut(old_head);
            self.free_list = link_mut.next.take();
        }

        self.outstanding += 1;

        Ok(self.base.with_addr_and_size(old_head, self.block_size()))
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this
    /// allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        let addr = ptr.addr();

        unsafe {
            self.base.link_mut(addr).next = self.free_list;
            self.free_list = Some(addr);
        }

        self.outstanding -= 1;
    }

    /// Returns a pointer to the managed region.
    ///
    /// It is undefined behavior to dereference the returned pointer or upgrade
    /// it to a reference if there are any outstanding allocations.
    #[inline]
    pub fn region(&mut self) -> NonNull<[u8]> {
        NonNull::new(ptr::slice_from_raw_parts_mut(
            self.base.ptr().as_ptr(),
            self.size(),
        ))
        .unwrap()
    }

    /// Returns the size in bytes of an allocated block.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # fn main() {
    /// use acid_alloc::Slab;
    ///
    /// let slab = Slab::try_new(96, 8).unwrap();
    /// assert_eq!(slab.block_size(), 96);
    /// # }
    ///
    /// # #[cfg(not(feature = "alloc"))]
    /// # fn main() {}
    /// ```
    #[inline]
    pub fn block_size(&self) -> usize {
        // Safe cast: block size is provided to constructors as a usize.
        self.block_size as usize
    }

    /// Returns the minimum alignment of an allocated block.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # fn main() {
    /// use acid_alloc::Slab;
    ///
    /// let slab = Slab::try_new(48, 12).unwrap();
    /// assert_eq!(slab.block_align(), 16);
    /// # }
    ///
    /// # #[cfg(not(feature = "alloc"))]
    /// # fn main() {}
    /// ```
    #[inline]
    pub fn block_align(&self) -> usize {
        // Safe cast: block align cannot exceed 1 << (usize::BITS - 1).
        self.block_align as usize
    }

    /// Returns the number of blocks managed by this allocator.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # fn main() {
    /// use acid_alloc::Slab;
    ///
    /// let slab = Slab::try_new(64, 7).unwrap();
    /// assert_eq!(slab.num_blocks(), 7);
    /// # }
    ///
    /// # #[cfg(not(feature = "alloc"))]
    /// # fn main() {}
    /// ```
    #[inline]
    pub fn num_blocks(&self) -> usize {
        // Safe cast: num_blocks is provided to constructors as a usize.
        self.num_blocks as usize
    }

    /// Returns the size in bytes of the managed region.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # fn main() {
    /// use acid_alloc::Slab;
    ///
    /// let slab = Slab::try_new(128, 4).unwrap();
    /// assert_eq!(slab.size(), 512);
    /// # }
    ///
    /// # #[cfg(not(feature = "alloc"))]
    /// # fn main() {}
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        // Safe unchecked mul: checked by constructor.
        self.block_size() * self.num_blocks()
    }

    /// Returns the first address above the managed region.
    ///
    /// If the managed region ends at the end of the address space, returns
    /// `None`.
    #[inline]
    pub fn limit(&self) -> Option<NonZeroUsize> {
        self.base
            .addr()
            .get()
            .checked_add(self.size())
            .and_then(NonZeroUsize::new)
    }

    /// Returns `true` _iff_ `ptr` is within this allocator's managed region.
    ///
    /// Note that a return value of `true` does not indicate that `ptr` points
    /// to an outstanding allocation.
    #[inline]
    pub fn contains_ptr(&self, ptr: NonNull<u8>) -> bool {
        self.base.addr() <= ptr.addr() && self.limit().map(|lim| ptr.addr() < lim).unwrap_or(true)
    }

    /// Returns `true` _iff_ this allocator has allocated all its memory.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.free_list.is_none()
    }

    /// Returns the number of outstanding allocations.
    #[inline]
    pub fn outstanding(&self) -> usize {
        self.outstanding.try_into().unwrap()
    }
}

impl<A> fmt::Debug for Slab<A>
where
    A: BackingAllocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slab")
            .field("base", &self.base.ptr())
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
        let region_layout = Self::region_layout(self.block_size(), self.num_blocks()).unwrap();

        if region_layout.size() != 0 {
            unsafe {
                self.backing_allocator
                    .deallocate(self.base.ptr(), region_layout)
            }
        }
    }
}

struct RawSlab {
    base: BasePtr,
    free_list: Option<NonZeroUsize>,
    block_size: u32,
    block_align: u32,
    num_blocks: u32,
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

        let base = BasePtr::new(region, layout.size());

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

        let block_align = 1_u32
            .checked_shl(block_size.trailing_zeros())
            .ok_or(AllocInitError::InvalidConfig)?;

        Ok(RawSlab {
            base,
            free_list: (num_blocks > 0).then(|| base.addr()),
            block_size: block_size
                .try_into()
                .map_err(|_| AllocInitError::InvalidConfig)?,
            block_align,
            num_blocks: num_blocks
                .try_into()
                .map_err(|_| AllocInitError::InvalidConfig)?,
        })
    }

    fn with_backing_allocator<A: BackingAllocator>(self, backing_allocator: A) -> Slab<A> {
        Slab {
            base: self.base,
            free_list: self.free_list,
            block_size: self.block_size,
            block_align: self.block_align,
            num_blocks: self.num_blocks,
            outstanding: 0,
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
        assert_eq!(slab.outstanding(), 1);

        slab.allocate(layout).unwrap_err();
        assert_eq!(slab.outstanding(), 1);
        unsafe { slab.deallocate(b.cast()) };
        assert_eq!(slab.outstanding(), 0);

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
        assert_eq!(slab.outstanding(), 1);

        let b = slab.allocate(layout).unwrap();
        assert_eq!(slab.outstanding(), 2);

        slab.allocate(layout).unwrap_err();
        assert_eq!(slab.outstanding(), 2);

        unsafe {
            slab.deallocate(a.cast());
            assert_eq!(slab.outstanding(), 1);

            slab.deallocate(b.cast());
            assert_eq!(slab.outstanding(), 0);
        }

        // Allocate a, b
        // Free b, a
        let a = slab.allocate(layout).unwrap();
        assert_eq!(slab.outstanding(), 1);

        let b = slab.allocate(layout).unwrap();
        assert_eq!(slab.outstanding(), 2);

        slab.allocate(layout).unwrap_err();
        assert_eq!(slab.outstanding(), 2);

        unsafe {
            slab.deallocate(b.cast());
            assert_eq!(slab.outstanding(), 1);

            slab.deallocate(a.cast());
            assert_eq!(slab.outstanding(), 0);
        }
    }
}
