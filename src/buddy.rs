//! Binary-buddy allocation.
//!
//! The buddy algorithm divides the managed region into a fixed number of
//! equal, power-of-two-sized blocks. Each block can be recursively split in
//! half a fixed number of times in order to provide finer-grained allocations.
//! Buddy allocators excel in cases where most allocations have a power-of-two
//! size.
//!
//! ## Characteristics
//!
//! #### Time complexity
//!
//! | Operation                | Best-case | Worst-case                 |
//! |--------------------------|-----------|----------------------------|
//! | Allocate (size <= align) | O(1)      | O(log<sub>2</sub>_levels_) |
//! | Deallocate               | O(1)      | O(log<sub>2</sub>_levels_) |
//!
//! #### Fragmentation
//!
//! Buddy allocators exhibit limited external fragmentation, but suffer up to
//! 50% internal fragmentation because all allocatable blocks have a
//! power-of-two size.

#![cfg(any(feature = "sptr", feature = "unstable"))]

use core::{
    iter::{self, Peekable},
    mem::ManuallyDrop,
    ops::Range,
};

use crate::core::{
    alloc::{AllocError, Layout},
    cmp, fmt,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ptr::NonNull,
};

#[cfg(feature = "unstable")]
use crate::core::alloc::Allocator;

#[cfg(not(feature = "unstable"))]
use crate::core::{
    alloc::LayoutExt,
    num::UsizeExt,
    ptr::{NonNullStrict, Strict},
};

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
use alloc::alloc::Global;

use crate::{bitmap::Bitmap, AllocInitError, BackingAllocator, BasePtr, DoubleBlockLink, Raw};

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
use crate::Global;

struct BuddyLevel {
    block_size: usize,
    block_pow: u32,
    free_list: Option<NonZeroUsize>,
    buddies: Bitmap,
    splits: Option<Bitmap>,
}

impl BuddyLevel {
    /// Retrieves the index of the block which starts `block_ofs` bytes from the
    /// base.
    ///
    /// `block_ofs` must be a multiple of `self.block_size`.
    #[inline]
    fn index_of(&self, block_ofs: usize) -> usize {
        // Safe unchecked shr: public Buddy ctors guarantee that
        // self.block_pow < usize::BITS
        block_ofs >> self.block_pow as usize
    }

    /// Retrieves the index of the buddy bit for the block which starts
    /// `block_ofs` bytes from the base.
    #[inline]
    fn buddy_bit(&self, block_ofs: usize) -> usize {
        // Safe unchecked shr: RHS is < usize::BITS
        self.index_of(block_ofs) >> 1
    }

    /// Retrieves the offset of the buddy of the block which starts
    /// `block_ofs` bytes from the base.
    #[inline]
    fn buddy_ofs(&self, block_ofs: usize) -> usize {
        block_ofs ^ self.block_size
    }

    #[cfg(test)]
    fn enumerate_free_list(&self, base: BasePtr) -> usize {
        let mut item = self.free_list;
        let mut num = 0;
        let mut prev = None;

        while let Some(it) = item {
            num += 1;

            let link = unsafe { base.double_link_mut(it) };

            assert_eq!(link.prev, prev);

            prev = item;
            item = link.next;
        }

        num
    }

    /// Pushes a block onto the free list.
    unsafe fn free_list_push(&mut self, base: BasePtr, block: NonZeroUsize) {
        assert_eq!(block.get() & (mem::align_of::<DoubleBlockLink>() - 1), 0);

        let new_head = block;

        if let Some(old_head) = self.free_list {
            let old_head_mut = unsafe { base.double_link_mut(old_head) };
            old_head_mut.prev = Some(new_head);
        }

        let old_head = self.free_list;

        // If `old_head` exists, it now points back to `new_head`.

        unsafe {
            base.init_double_link_at(
                new_head,
                DoubleBlockLink {
                    next: old_head,
                    prev: None,
                },
            )
        };

        // `new_head` now points forward to `old_head`.
        // `old_head` now points back to `new_head`.

        self.free_list = Some(new_head);
    }

    /// Removes the specified block from the free list.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - The memory at `block` must be within the provenance of `base` and valid for reads and
    ///   writes for `size_of::<BlockLink>()` bytes.
    /// - `block` must be the address of an element of `self.free_list`.
    unsafe fn free_list_remove(&mut self, base: BasePtr, block: NonZeroUsize) {
        unsafe {
            let removed = base.double_link_mut(block);
            debug_assert!(removed.next.map_or(true, |next| base.contains_addr(next)));

            match removed.prev {
                // Link `prev` forward to `next`.
                Some(p) => {
                    base.double_link_mut(p).next = removed.next;
                }

                // If there's no previous block, then `removed` is the head of
                // the free list.
                None => self.free_list = removed.next,
            }

            if let Some(n) = removed.next {
                // Link `next` back to `prev`.
                base.double_link_mut(n).prev = removed.prev;
            }
        }
    }

    /// Allocates a block from the free list.
    unsafe fn allocate(&mut self, base: BasePtr) -> Option<NonZeroUsize> {
        let block = self.free_list?;

        unsafe { self.free_list_remove(base, block) };
        let ofs = base.offset_to(block);
        self.buddies.toggle(self.buddy_bit(ofs));

        Some(block)
    }

    /// Assigns a block to this level.
    unsafe fn assign(&mut self, base: BasePtr, block: NonZeroUsize) {
        let ofs = base.offset_to(block);

        let buddy_bit = self.buddy_bit(ofs);
        assert!(!self.buddies.get(buddy_bit));
        self.buddies.set(buddy_bit, true);

        unsafe { self.free_list_push(base, block) };
    }

    unsafe fn deallocate(
        &mut self,
        base: BasePtr,
        block: NonNull<u8>,
        coalesce: bool,
    ) -> Option<NonNull<u8>> {
        let block = block.addr();
        let block_ofs = base.offset_to(block);
        let buddy_bit = self.buddy_bit(block_ofs);
        self.buddies.toggle(buddy_bit);

        let split_bit = self.index_of(block_ofs);
        if let Some(splits) = self.splits.as_mut() {
            splits.set(split_bit, false);
        }

        if !coalesce || self.buddies.get(buddy_bit) {
            unsafe { self.free_list_push(base, block) };
            None
        } else {
            let buddy_ofs = self.buddy_ofs(block_ofs);
            let buddy =
                NonZeroUsize::new(base.addr().get().checked_add(buddy_ofs).unwrap()).unwrap();

            // Remove the buddy block from the free list.
            unsafe { self.free_list_remove(base, buddy) };

            // Return the coalesced block.
            let coalesced_ofs = buddy_ofs & !self.block_size;
            Some(base.with_offset(coalesced_ofs).unwrap())
        }
    }
}

/// A binary-buddy allocator.
///
/// For a discussion of the buddy algorithm, see the [module-level documentation].
///
/// # Configuration
///
/// This type has two const parameters:
/// - `BLK_SIZE` is the size of the largest allocations the allocator can make.
/// - `LEVELS` is the number of levels in the allocator.
///
/// Each constructor also takes one runtime parameter:
/// - `num_blocks` is the number of top-level blocks of `BLK_SIZE` bytes managed by the allocator.
///
/// These parameters are subject to the following invariants:
/// - `BLK_SIZE` must be a power of two.
/// - `LEVELS` must be nonzero and less than `usize::BITS`.
/// - The minumum block size must be at least `2 * mem::size_of<usize>()`; it can be calculated with
///   the formula `BLK_SIZE >> (LEVELS - 1)`.
/// - The total size in bytes of the managed region must be less than `usize::MAX`.
///
/// Attempting to construct a `Buddy` whose const parameters violate
/// these invariants will result in an error.
///
/// # Unpopulated initialization
///
/// A `Buddy` can be initialized for a region of memory without immediately allowing it to access
/// any of that memory using the [`Buddy::new_raw_unpopulated`] constructor. Memory in that region
/// can then be added to the allocator over time.
///
/// This is particularly useful when using a `Buddy` to allocate discontiguous regions of physical
/// memory. For example, consider a 16 KiB region of physical memory in which the region between 4-8
/// KiB is reserved by the platform for memory-mapped I/O.
///
/// ```text
///       | RAM              | MMIO             | RAM               | RAM             |
///   0x##0000           0x##1000           0x##2000            0x##3000          0x##4000
/// ```
///
/// Initializing an allocator normally for the entire region would result in writes to the MMIO
/// region, with unpredictable consequences.
///
/// ```no_run
/// // These values result in a minimum block size of 4096.
/// type Buddy = acid_alloc::Buddy<16384, 3, acid_alloc::Raw>;
///
/// # fn main() {
/// # /*
/// let region = /* ... */;
/// let metadata = /* ... */;
/// # */
/// # let region: core::ptr::NonNull<u8> = unimplemented!();
/// # let metadata: core::ptr::NonNull<u8> = unimplemented!();
///
/// // ⚠ Bad Things occur!
/// let buddy = unsafe { Buddy::new_raw(region, metadata, 1).unwrap() };
/// # }
/// ```
///
/// However, creating the allocator using `new_raw_unpopulated()` does not issue any writes.
/// Instead, valid subregions can be added to the allocator explicitly.
///
/// ```no_run
/// # use core::{alloc::Layout, num::NonZeroUsize};
/// # const BLK_SIZE: usize = 16384;
/// # const LEVELS: usize = 3;
/// # type Buddy = acid_alloc::Buddy<BLK_SIZE, LEVELS, acid_alloc::Raw>;
/// # fn main() {
/// # let region: core::ptr::NonNull<u8> = unimplemented!();
/// # let metadata: core::ptr::NonNull<u8> = unimplemented!();
/// // Define the usable regions of memory.
/// let low_start = NonZeroUsize::new(region.as_ptr() as usize).unwrap();
/// let low_end = NonZeroUsize::new(low_start.get() + 4096).unwrap();
/// let high_start = NonZeroUsize::new(low_start.get() + 8192).unwrap();
/// let high_end = NonZeroUsize::new(low_start.get() + 16384).unwrap();
///
/// unsafe {
///     // No accesses to `region` are made during this call.
///     let mut buddy = Buddy::new_raw_unpopulated(region, metadata, 1).unwrap();
///
///     // The allocator has no memory yet, so it can't make allocations.
///     assert!(buddy.allocate(Layout::new::<[u8; 4096]>()).is_err());
///
///     // Add the valid regions to the allocator.
///     buddy.add_region(low_start..low_end);
///     buddy.add_region(high_start..high_end);
///
///     // Now allocations succeed.
///     let block = buddy.allocate(Layout::new::<[u8; 4096]>()).unwrap();
/// }
///
/// # }
/// ```
///
/// # Example
///
/// ```
/// # #![cfg_attr(feature = "unstable", feature(allocator_api))]
/// # #[cfg(feature = "alloc")]
/// use core::{alloc::Layout, mem::MaybeUninit, ptr::NonNull};
///
/// # #[cfg(feature = "alloc")]
/// use acid_alloc::{Buddy, Global};
///
/// // Minimum block size == BLK_SIZE >> (LEVELS - 1)
/// //                 16 ==     4096 >> (     9 - 1)
/// # #[cfg(feature = "alloc")]
/// type CustomBuddy = Buddy<4096, 9, Global>;
///
/// # #[cfg(feature = "alloc")]
/// # fn main() {
/// // Create a new `Buddy` with four 4KiB blocks, backed by the global allocator.
/// let mut buddy = CustomBuddy::try_new(4).expect("buddy initialization failed.");
///
/// // Allocate space for an array of `u64`s.
/// let len = 8;
/// let layout = Layout::array::<u64>(len).expect("layout overflowed");
/// let mut buf: NonNull<[u8]> = buddy.allocate(layout).expect("allocation failed");
/// let mut uninit: NonNull<[MaybeUninit<u64>; 8]> = buf.cast();
///
/// // Initialize the array.
/// unsafe {
///     let mut arr: &mut [MaybeUninit<u64>; 8] = uninit.as_mut();
///     for (i, elem) in arr.iter_mut().enumerate() {
///         elem.as_mut_ptr().write(i as u64);
///     }
/// }
///
/// // Deallocate the array.
/// unsafe { buddy.deallocate(buf.cast()) };
/// # }
///
/// # #[cfg(not(feature = "alloc"))]
/// # fn main() {}
/// ```
///
/// [module-level documentation]: crate::buddy
pub struct Buddy<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> {
    raw: RawBuddy<BLK_SIZE, LEVELS>,
    backing_allocator: A,
}

unsafe impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator + Send> Send
    for Buddy<BLK_SIZE, LEVELS, A>
{
}
unsafe impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator + Sync> Sync
    for Buddy<BLK_SIZE, LEVELS, A>
{
}

impl<const BLK_SIZE: usize, const LEVELS: usize> Buddy<BLK_SIZE, LEVELS, Raw> {
    /// Constructs a new `Buddy` from raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `region` must be a pointer to a region that satisfies the [`Layout`] returned by
    ///   [`Self::region_layout(num_blocks)`], and it must be valid for reads and writes for the
    ///   entire size indicated by that `Layout`.
    /// - `metadata` must be a pointer to a region that satisfies the [`Layout`] returned by
    ///   [`Self::metadata_layout(num_blocks)`], and it must be valid for reads and writes for the
    ///   entire size indicated by that `Layout`.
    /// - The regions pointed to by `region` and `metadata` must not overlap.
    /// - No references to the memory at `region` or `metadata` may exist when this function is
    ///   called.
    /// - As long as the returned `Buddy` exists:
    ///     - No accesses may be made to the memory at `region` except by way of methods on the
    ///       returned `Buddy`.
    ///     - No accesses may be made to the memory at `metadata`.
    ///
    ///  # Errors
    ///
    ///  This constructor returns an error if the allocator configuration is
    ///  invalid.
    ///
    /// [`Self::region_layout(num_blocks)`]: Self::region_layout
    /// [`Self::metadata_layout(num_blocks)`]: Self::metadata_layout
    /// [`Layout`]: core::alloc::Layout
    pub unsafe fn new_raw(
        metadata: NonNull<u8>,
        region: NonNull<u8>,
        num_blocks: usize,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, Raw>, AllocInitError> {
        unsafe {
            RawBuddy::try_new_with_address_gaps(metadata, region, num_blocks, iter::empty())
                .map(|p| p.with_backing_allocator(Raw))
        }
    }

    /// Constructs a new `Buddy` from raw pointers without populating it.
    ///
    /// The returned `Buddy` will be unable to allocate until address ranges are made available to
    /// it using [`Buddy::add_region`]. See [unpopulated initialization] for details.
    ///
    /// [unpopulated initialization]: crate::buddy::Buddy#unpopulated-initialization
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `region` must be a pointer to a region that satisfies the [`Layout`] returned by
    ///   [`Self::region_layout(num_blocks)`].
    /// - `metadata` must be a pointer to a region that satisfies the [`Layout`] returned by
    ///   [`Self::metadata_layout(num_blocks)`], and it must be valid for reads and writes for the
    ///   entire size indicated by that `Layout`.
    /// - The regions pointed to by `region` and `metadata` must not overlap.
    /// - No references to the memory at `metadata` may exist when this function is called.
    /// - As long as the returned `Buddy` exists, no accesses may be made to the memory at
    ///   `metadata`.
    ///
    /// [`Self::region_layout(num_blocks)`]: Self::region_layout
    /// [`Self::metadata_layout(num_blocks)`]: Self::metadata_layout
    /// [`Layout`]: core::alloc::Layout
    pub unsafe fn new_raw_unpopulated(
        metadata: NonNull<u8>,
        region: NonNull<u8>,
        num_blocks: usize,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, Raw>, AllocInitError> {
        unsafe {
            RawBuddy::try_new_unpopulated(metadata, region, num_blocks)
                .map(|p| p.with_backing_allocator(Raw))
        }
    }

    /// Populates a region not already managed by this allocator.
    ///
    /// # Panics
    ///
    /// This method panics if any of the following are true:
    /// - Either bound of `addr_range` falls outside the allocator region.
    /// - Either bound of `addr_range` is not aligned to the value returned by
    ///   [`Self::min_block_size()`][0].
    ///
    /// [0]: Buddy::min_block_size
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `range` must not overlap any range of addresses already managed by this allocator.
    /// - No references to the memory indicated by `addr_range` may exist when this function is
    ///   called.
    /// - As long as `self` exists, no accesses may be made to the memory indicated by `addr_range`
    ///   except by way of methods on `self`.
    pub unsafe fn add_region(&mut self, addr_range: Range<NonZeroUsize>) {
        unsafe { self.raw.add_region(addr_range) };
    }

    /// Decomposes the allocator into its raw parts.
    ///
    /// # Safety
    ///
    /// This function must only be called if no references to outstanding
    /// allocations exist .
    pub unsafe fn into_raw_parts(self) -> RawBuddyParts {
        let metadata = self.raw.metadata;
        let metadata_layout = Self::metadata_layout(self.raw.num_blocks.get()).unwrap();
        let region = self.raw.base.ptr();
        let region_layout = Self::region_layout(self.raw.num_blocks.get()).unwrap();

        let _ = ManuallyDrop::new(self);

        RawBuddyParts {
            metadata,
            metadata_layout,
            region,
            region_layout,
        }
    }
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
impl<const BLK_SIZE: usize, const LEVELS: usize> Buddy<BLK_SIZE, LEVELS, Global> {
    /// Attempts to construct a new `Buddy` backed by the global allocator.
    ///
    /// # Errors
    ///
    /// If allocation fails, returns `Err(AllocError)`.
    pub fn try_new(num_blocks: usize) -> Result<Buddy<BLK_SIZE, LEVELS, Global>, AllocInitError> {
        Self::try_new_with_offset_gaps(num_blocks, iter::empty())
    }

    /// Attempts to construct a new `Buddy` backed by the global allocator, with
    /// gaps specified by offset ranges.
    ///
    /// # Errors
    ///
    /// If allocation fails, returns `Err(AllocError)`.
    #[doc(hidden)]
    pub fn try_new_with_offset_gaps<I>(
        num_blocks: usize,
        gaps: I,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, Global>, AllocInitError>
    where
        I: IntoIterator<Item = Range<usize>>,
    {
        let region_layout = Self::region_layout(num_blocks)?;
        let metadata_layout = Self::metadata_layout(num_blocks)?;

        let num_blocks = NonZeroUsize::new(num_blocks).ok_or(AllocInitError::InvalidConfig)?;

        unsafe {
            let region_ptr = {
                let raw = alloc::alloc::alloc(region_layout);
                NonNull::new(raw).ok_or(AllocInitError::AllocFailed(region_layout))?
            };

            let metadata_ptr = {
                let raw = alloc::alloc::alloc(metadata_layout);
                NonNull::new(raw).ok_or_else(|| {
                    alloc::alloc::dealloc(region_ptr.as_ptr(), region_layout);
                    AllocInitError::AllocFailed(metadata_layout)
                })?
            };

            match RawBuddy::<BLK_SIZE, LEVELS>::try_new_with_offset_gaps(
                metadata_ptr,
                region_ptr,
                num_blocks.get(),
                gaps,
            ) {
                Ok(b) => Ok(b.with_backing_allocator(Global)),
                Err(e) => {
                    alloc::alloc::dealloc(region_ptr.as_ptr(), region_layout);
                    alloc::alloc::dealloc(metadata_ptr.as_ptr(), metadata_layout);

                    Err(e)
                }
            }
        }
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl<const BLK_SIZE: usize, const LEVELS: usize> Buddy<BLK_SIZE, LEVELS, Global> {
    /// Attempts to construct a new `Buddy` backed by the global allocator.
    ///
    /// # Errors
    ///
    /// If allocation fails, or if the allocator configuration is invalid,
    /// returns `Err`.
    #[cfg_attr(docs_rs, doc(cfg(feature = "alloc")))]
    pub fn try_new(num_blocks: usize) -> Result<Buddy<BLK_SIZE, LEVELS, Global>, AllocInitError> {
        Buddy::<BLK_SIZE, LEVELS, Global>::try_new_in(num_blocks, Global)
    }

    /// Attempts to construct a new `Buddy` backed by the global allocator, with
    /// gaps specified by offset ranges.
    ///
    /// # Errors
    ///
    /// If allocation fails, or if the allocator configuration is invalid,
    /// returns `Err`.
    #[doc(hidden)]
    pub fn try_new_with_offset_gaps<I>(
        num_blocks: usize,
        gaps: I,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, Global>, AllocInitError>
    where
        I: IntoIterator<Item = Range<usize>>,
    {
        Buddy::<BLK_SIZE, LEVELS, Global>::try_new_with_offset_gaps_in(num_blocks, gaps, Global)
    }
}

#[cfg(feature = "unstable")]
impl<const BLK_SIZE: usize, const LEVELS: usize, A: Allocator> Buddy<BLK_SIZE, LEVELS, A> {
    /// Attempts to construct a new `Buddy` backed by `allocator`.
    ///
    /// # Errors
    ///
    /// If allocation fails, returns `Err(AllocError)`.
    #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
    pub fn try_new_in(
        num_blocks: usize,
        allocator: A,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, A>, AllocInitError> {
        Self::try_new_with_offset_gaps_in(num_blocks, iter::empty(), allocator)
    }

    /// Attempts to construct a new `Buddy` backed by `allocator`, with gaps
    /// specified by offset ranges.
    ///
    /// # Errors
    ///
    /// If allocation fails, returns `Err(AllocError)`.
    #[doc(hidden)]
    #[cfg_attr(docs_rs, doc(cfg(feature = "unstable")))]
    pub fn try_new_with_offset_gaps_in<I>(
        num_blocks: usize,
        gaps: I,
        allocator: A,
    ) -> Result<Buddy<BLK_SIZE, LEVELS, A>, AllocInitError>
    where
        I: IntoIterator<Item = Range<usize>>,
    {
        let region_layout = Self::region_layout(num_blocks)?;
        let metadata_layout = Self::metadata_layout(num_blocks)?;

        let num_blocks = NonZeroUsize::new(num_blocks).ok_or(AllocInitError::InvalidConfig)?;

        let region = allocator
            .allocate(region_layout)
            .map_err(|_| AllocInitError::AllocFailed(region_layout))?;

        let metadata = match allocator.allocate(metadata_layout) {
            Ok(m) => m,
            Err(_) => unsafe {
                // SAFETY: region was received as NonNull via Allocator::allocate
                let region_ptr = NonNull::new_unchecked(region.as_ptr() as *mut u8);
                allocator.deallocate(region_ptr, region_layout);
                return Err(AllocInitError::AllocFailed(metadata_layout));
            },
        };

        unsafe {
            // SAFETY: both pointers were received as NonNull via Allocator::allocate
            let region_ptr = NonNull::new_unchecked(region.as_ptr() as *mut u8);
            let metadata_ptr = NonNull::new_unchecked(metadata.as_ptr() as *mut u8);

            RawBuddy::<BLK_SIZE, LEVELS>::try_new_with_offset_gaps(
                metadata_ptr,
                region_ptr,
                num_blocks.get(),
                gaps,
            )
            .map(|p| p.with_backing_allocator(allocator))
        }
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> Buddy<BLK_SIZE, LEVELS, A> {
    /// Returns the smallest block size that can be allocated by an allocator of this type.
    pub const fn min_block_size() -> Result<usize, AllocInitError> {
        if LEVELS == 0 || !BLK_SIZE.is_power_of_two() || LEVELS >= usize::BITS as usize {
            return Err(AllocInitError::InvalidConfig);
        }

        let min_block_size = BLK_SIZE >> (LEVELS - 1);

        if min_block_size < mem::size_of::<DoubleBlockLink>() {
            return Err(AllocInitError::InvalidConfig);
        }

        Ok(min_block_size)
    }

    /// Returns the layout requirements of the region managed by an allocator of
    /// this type.
    pub fn region_layout(num_blocks: usize) -> Result<Layout, AllocInitError> {
        let num_blocks = NonZeroUsize::new(num_blocks).ok_or(AllocInitError::InvalidConfig)?;
        Self::region_layout_impl(num_blocks)
    }

    fn region_layout_impl(num_blocks: NonZeroUsize) -> Result<Layout, AllocInitError> {
        let min_block_size = Self::min_block_size()?;
        let levels: u32 = LEVELS
            .try_into()
            .map_err(|_| AllocInitError::InvalidConfig)?;

        let size = 2usize
            .pow(levels - 1)
            .checked_mul(min_block_size)
            .ok_or(AllocInitError::InvalidConfig)?
            .checked_mul(num_blocks.get())
            .ok_or(AllocInitError::InvalidConfig)?;
        let align = BLK_SIZE;

        Layout::from_size_align(size, align).map_err(|_| AllocInitError::InvalidConfig)
    }

    /// Returns the layout requirements of the metadata region for an allocator
    /// of this type.
    pub fn metadata_layout(num_blocks: usize) -> Result<Layout, AllocInitError> {
        let num_blocks = NonZeroUsize::new(num_blocks).ok_or(AllocInitError::InvalidConfig)?;
        Self::metadata_layout_impl(num_blocks)
    }

    /// Returns the layout requirements of the metadata region for an allocator
    /// of this type.
    fn metadata_layout_impl(num_blocks: NonZeroUsize) -> Result<Layout, AllocInitError> {
        const fn sum_of_powers_of_2(max: u32) -> usize {
            2_usize.pow(max + 1) - 1
        }

        let levels: u32 = LEVELS.try_into().unwrap();

        // Each level needs one buddy bit per pair of blocks.
        let num_pairs = (num_blocks.get() + 1) / 2;

        // This is the layout required for the buddy bitmap of level 0.
        let buddy_l0_layout = Bitmap::map_layout(num_pairs);

        // Each subsequent level requires at most twice as much space as the
        // level 0 bitmap. It may require less if the number of level 0 blocks
        // is not a multiple of the bitmap block size, but for simplicity each
        // level is given exactly twice the space of the previous level.
        let (buddy_layout, _) = buddy_l0_layout
            .repeat(sum_of_powers_of_2(levels - 1))
            .unwrap();

        if LEVELS == 1 {
            // There's only one level, so no split bitmap is required.
            return Ok(buddy_layout);
        }

        // Each level except level (LEVELS - 1) needs one split bit per block.
        let split_l0_layout = Bitmap::map_layout(num_blocks.get());

        // Let K equal the size of a split bitmap for level 0. If LEVELS is:
        // - 2, then 1 split bitmap is needed of size (2 - 1)K = K.
        // - 3, then 2 split bitmaps are needed of total size (3 - 1)K + (2 - 1)K = 3K.
        // - ...
        // - N, then 2 ^ (N - 2) split bitmaps are needed of total size:
        //
        //     (N - 1)K + ((N - 1) - 1)K + ... + (2 - 1)K
        //   = 2 * (2 ^ (N - 1) - 1) * K
        //   = (sum from x = 1 to (LEVELS - 1) of 2^x) * K
        let (split_layout, _) = split_l0_layout
            .repeat(sum_of_powers_of_2(levels - 1))
            .map_err(|_| AllocInitError::InvalidConfig)?;
        let (full_layout, _) = buddy_layout
            .extend(split_layout)
            .map_err(|_| AllocInitError::InvalidConfig)?;

        Ok(full_layout)
    }

    #[cfg(test)]
    #[inline]
    fn enumerate_free_list(&self, level: usize) -> usize {
        self.raw.enumerate_free_list(level)
    }

    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`][0] which satisfies `layout`.
    ///
    /// The contents of the block are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a suitable block could not be allocated.
    ///
    /// [0]: core::ptr::NonNull
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.raw.allocate(layout)
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        unsafe { self.raw.deallocate(ptr) }
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> fmt::Debug
    for Buddy<BLK_SIZE, LEVELS, A>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buddy")
            .field("metadata", &self.raw.metadata)
            .field("base", &self.raw.base.ptr())
            .field("num_blocks", &self.raw.num_blocks)
            .finish()
    }
}

impl<const BLK_SIZE: usize, const LEVELS: usize, A: BackingAllocator> Drop
    for Buddy<BLK_SIZE, LEVELS, A>
{
    fn drop(&mut self) {
        let region = self.raw.base.ptr();
        let metadata = self.raw.metadata;
        let num_blocks = self.raw.num_blocks;

        let region_layout = Self::region_layout_impl(num_blocks).unwrap();
        let metadata_layout = Self::metadata_layout_impl(num_blocks).unwrap();

        unsafe {
            self.backing_allocator.deallocate(region, region_layout);
            self.backing_allocator.deallocate(metadata, metadata_layout);
        }
    }
}

/// The raw parts of a `Buddy`.
#[derive(Debug)]
pub struct RawBuddyParts {
    /// A pointer to the metadata region.
    pub metadata: NonNull<u8>,
    /// The layout of the metadata region.
    pub metadata_layout: Layout,
    /// A pointer to the managed region.
    pub region: NonNull<u8>,
    /// The layout of the managed region.
    pub region_layout: Layout,
}

/// Like a `Buddy`, but without a `Drop` impl or an associated
/// allocator.
///
/// This assists in tacking on the allocator type parameter because this struct can be
/// moved out of, while `Buddy` itself cannot.
struct RawBuddy<const BLK_SIZE: usize, const LEVELS: usize> {
    base: BasePtr,
    metadata: NonNull<u8>,
    num_blocks: NonZeroUsize,
    levels: [BuddyLevel; LEVELS],
}

impl<const BLK_SIZE: usize, const LEVELS: usize> RawBuddy<BLK_SIZE, LEVELS> {
    fn with_backing_allocator<A: BackingAllocator>(
        self,
        backing_allocator: A,
    ) -> Buddy<BLK_SIZE, LEVELS, A> {
        Buddy {
            raw: self,
            backing_allocator,
        }
    }

    #[cfg(any(feature = "unstable", feature = "alloc", test))]
    unsafe fn try_new_with_offset_gaps<I>(
        metadata: NonNull<u8>,
        base: NonNull<u8>,
        num_blocks: usize,
        gaps: I,
    ) -> Result<RawBuddy<BLK_SIZE, LEVELS>, AllocInitError>
    where
        I: IntoIterator<Item = Range<usize>>,
    {
        let region_addr = base.addr().get();

        let gaps = gaps.into_iter().map(|ofs| {
            let start = NonZeroUsize::new(region_addr.checked_add(ofs.start).unwrap()).unwrap();
            let end = NonZeroUsize::new(region_addr.checked_add(ofs.end).unwrap()).unwrap();
            start..end
        });

        unsafe { Self::try_new_with_address_gaps(metadata, base, num_blocks, gaps) }
    }

    /// Construct a new `RawBuddy` from raw pointers, with internal gaps.
    ///
    /// The address ranges in `gaps` are guaranteed not to be read from or
    /// written to.
    unsafe fn try_new_with_address_gaps<I>(
        metadata: NonNull<u8>,
        base: NonNull<u8>,
        num_blocks: usize,
        gaps: I,
    ) -> Result<RawBuddy<BLK_SIZE, LEVELS>, AllocInitError>
    where
        I: IntoIterator<Item = Range<NonZeroUsize>>,
    {
        let mut buddy = unsafe { Self::try_new_unpopulated(metadata, base, num_blocks) }?;

        let min_block_size = Buddy::<BLK_SIZE, LEVELS, Raw>::min_block_size().unwrap();
        let gaps = AlignedRanges::new(gaps.into_iter(), min_block_size);

        let mut start = buddy.base.addr();
        for gap in gaps {
            let end = gap.start;
            unsafe { buddy.add_region(start..end) };
            start = gap.end;
        }
        let end = buddy.base.limit();

        unsafe { buddy.add_region(start..end) };

        Ok(buddy)
    }

    unsafe fn add_region(&mut self, addr_range: Range<NonZeroUsize>) {
        // Make sure the region can be managed by this allocator.
        assert!(self.base.addr() <= addr_range.start);
        assert!(addr_range.end <= self.base.limit());

        let min_block_size = Self::min_block_size().unwrap();

        // Require the range bounds to be aligned on block boundaries.
        assert_eq!(addr_range.start.get() % min_block_size, 0);
        assert_eq!(addr_range.end.get() % min_block_size, 0);

        let min_pow = min_block_size.trailing_zeros();
        let max_pow = BLK_SIZE.trailing_zeros();

        let mut curs = addr_range.start;
        while curs < addr_range.end {
            let curs_pow = curs.trailing_zeros().min(max_pow);
            // Cursor should never go out of alignment with min block size.
            assert!(curs_pow >= min_pow);

            let curs_align = 1 << curs_pow;
            let curs_ofs = curs.get() - self.base.addr().get();

            // Safe unchecked sub: `curs < range.end`
            let remaining = addr_range.end.get() - curs.get();
            // Safe unchecked sub and shift: `remaining` is nonzero, so
            // `remaining.leading_zeros() + 1 <= usize::BITS`
            let remaining_po2 = 1 << (usize::BITS - (remaining.leading_zeros() + 1));

            // Necessarily <= BLK_SIZE, as curs_pow is in the range
            // [min_pow, max_pow].
            let block_size: usize = cmp::min(remaining_po2, curs_align);

            // Split all blocks that begin at this cursor position but are larger
            // than `block_size`.
            //
            // Note that the blocks may already be split, as sub-blocks may
            // already have been populated.
            let init_level = (max_pow - curs_pow) as usize;
            let target_level = (max_pow - block_size.trailing_zeros()) as usize;
            for lv in self.levels.iter_mut().take(target_level).skip(init_level) {
                // Mark the block as split.
                let split_bit = lv.index_of(curs_ofs);
                if let Some(s) = lv.splits.as_mut() {
                    s.set(split_bit, true);
                }
            }

            unsafe { self.deallocate(self.base.with_addr(curs)) };

            // TODO: call curs.checked_add() directly when nonzero_ops is stable
            curs = curs
                .get()
                .checked_add(block_size)
                .and_then(NonZeroUsize::new)
                .unwrap();
        }
    }

    pub unsafe fn try_new_unpopulated(
        metadata: NonNull<u8>,
        base: NonNull<u8>,
        num_blocks: usize,
    ) -> Result<RawBuddy<BLK_SIZE, LEVELS>, AllocInitError> {
        let num_blocks = NonZeroUsize::new(num_blocks).ok_or(AllocInitError::InvalidConfig)?;
        let min_block_size = Buddy::<BLK_SIZE, LEVELS, Raw>::min_block_size()?;
        let meta_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::metadata_layout_impl(num_blocks)?;
        let region_layout = Buddy::<BLK_SIZE, LEVELS, Raw>::region_layout_impl(num_blocks)?;

        // Ensure pointer calculations will not overflow.
        // TODO: use checked_add directly on NonNull when nonzero_ops is stable.
        let meta_end = metadata
            .addr()
            .get()
            .checked_add(meta_layout.size())
            .ok_or(AllocInitError::InvalidLocation)?;
        base.addr()
            .get()
            .checked_add(region_layout.size())
            .ok_or(AllocInitError::InvalidLocation)?;

        // TODO: use MaybeUninit::uninit_array when not feature gated
        let mut levels: [MaybeUninit<BuddyLevel>; LEVELS] = unsafe {
            // SAFETY: An uninitialized `[MaybeUninit<_>; _]` is valid.
            MaybeUninit::<[MaybeUninit<BuddyLevel>; LEVELS]>::uninit().assume_init()
        };

        let mut meta_curs = metadata.as_ptr();

        // Initialize the per-level metadata.
        for (li, level) in levels.iter_mut().enumerate() {
            let block_size = 2_usize.pow((LEVELS - li) as u32 - 1) * min_block_size;
            let block_factor = 2_usize.pow(li as u32);
            let num_blocks = block_factor * num_blocks.get();
            let num_pairs = (num_blocks + 1) / 2;

            let buddy_size = Bitmap::map_layout(num_pairs).size();
            let buddy_bitmap = unsafe { Bitmap::new(num_pairs, meta_curs as *mut u64) };

            meta_curs = unsafe {
                meta_curs.offset(
                    buddy_size
                        .try_into()
                        .expect("buddy bitmap layout size overflows isize"),
                )
            };

            let split_bitmap = if li < LEVELS - 1 {
                let split_size = Bitmap::map_layout(num_blocks).size();
                let split_bitmap = unsafe { Bitmap::new(num_blocks, meta_curs as *mut u64) };

                meta_curs = unsafe {
                    meta_curs.offset(
                        split_size
                            .try_into()
                            .expect("split bitmap layout size overflows isize"),
                    )
                };

                Some(split_bitmap)
            } else {
                None
            };

            unsafe {
                level.as_mut_ptr().write(BuddyLevel {
                    block_size,
                    block_pow: block_size.trailing_zeros(),
                    free_list: None,
                    buddies: buddy_bitmap,
                    splits: split_bitmap,
                });
            }
        }

        if meta_curs.addr() > meta_end {
            panic!(
                "metadata cursor overran layout size: curs = {meta_end}, layout = {meta_layout:?}"
            );
        }

        // Convert to an initialized array.
        let levels = unsafe {
            // TODO: When `MaybeUninit::array_assume_init()` is stable, use that
            // instead.
            //
            // SAFETY:
            // - `levels` is fully initialized.
            // - `MaybeUninit<T>` and `T` have the same layout.
            // - `MaybeUninit<T>` won't drop `T`, so no double-frees.
            (&levels as *const _ as *const [BuddyLevel; LEVELS]).read()
        };

        let base = BasePtr::new(base, region_layout.size());

        Ok(RawBuddy {
            base,
            metadata,
            num_blocks,
            levels,
        })
    }

    const fn min_block_size() -> Result<usize, AllocInitError> {
        if LEVELS == 0 || !BLK_SIZE.is_power_of_two() || LEVELS >= usize::BITS as usize {
            return Err(AllocInitError::InvalidConfig);
        }

        let min_block_size = BLK_SIZE >> (LEVELS - 1);

        if min_block_size < mem::size_of::<DoubleBlockLink>() {
            return Err(AllocInitError::InvalidConfig);
        }

        Ok(min_block_size)
    }

    #[cfg(test)]
    #[inline]
    fn enumerate_free_list(&self, level: usize) -> usize {
        self.levels[level].enumerate_free_list(self.base)
    }

    fn level_for(&self, size: usize) -> Option<usize> {
        fn round_up_pow2(x: usize) -> Option<usize> {
            match x {
                0 => None,
                1 => Some(1),
                x if x >= (1 << 63) => None,
                _ => Some(2usize.pow((x - 1).ilog2() as u32 + 1)),
            }
        }

        let min_block_size = self.levels[LEVELS - 1].block_size;
        let max_block_size = self.levels[0].block_size;
        if size > max_block_size {
            return None;
        }

        let alloc_size = cmp::max(round_up_pow2(size).unwrap(), min_block_size);
        let level: usize = (max_block_size.ilog2() - alloc_size.ilog2())
            .try_into()
            .unwrap();

        Some(level)
    }

    fn min_free_level(&self, block_ofs: usize) -> usize {
        let min_block_size = self.levels[LEVELS - 1].block_size;
        let max_block_size = self.levels[0].block_size;

        if block_ofs == 0 {
            return 0;
        }

        let max_size = 1 << block_ofs.trailing_zeros();
        if max_size > max_block_size {
            return 0;
        }

        assert!(max_size >= min_block_size);

        (max_block_size.ilog2() - max_size.ilog2())
            .try_into()
            .unwrap()
    }

    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`][0] which satisfies `layout`.
    ///
    /// The contents of the block are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a suitable block could not be allocated.
    ///
    /// [0]: core::ptr::NonNull
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 || layout.align() > layout.size() {
            return Err(AllocError);
        }

        let target_level = self.level_for(layout.size()).ok_or(AllocError)?;

        // If there is a free block of the correct size, return it immediately.
        if let Some(block) = unsafe { self.levels[target_level].allocate(self.base) } {
            return Ok(self.base.with_addr_and_size(block, layout.size()));
        }

        // Otherwise, scan increasing block sizes until a free block is found.
        let (block, init_level) = (0..target_level)
            .rev()
            .find_map(|level| unsafe {
                self.levels[level]
                    .allocate(self.base)
                    .map(|block| (block, level))
            })
            .ok_or(AllocError)?;

        let block_ofs = self.base.offset_to(block);

        // Split the block repeatedly to obtain a suitably sized block.
        for level in init_level..target_level {
            // Split the block. The address of the front half does not change.
            let half_block_size = self.levels[level].block_size / 2;
            let back_half = NonZeroUsize::new(block.get() + half_block_size).unwrap();

            // Mark the block as split.
            let split_bit = self.levels[level].index_of(block_ofs);
            if let Some(s) = self.levels[level].splits.as_mut() {
                s.set(split_bit, true);
            }

            // Add one half of the split block to the next level's free list.
            unsafe { self.levels[level + 1].assign(self.base, back_half) };
        }

        // The returned block inherits the provenance of the base pointer.
        Ok(self.base.with_addr_and_size(block, layout.size()))
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        // Some addresses can't come from earlier levels because their addresses
        // imply a smaller block size.
        let block_ofs = self.base.offset_to(ptr.addr());
        let min_level = self.min_free_level(block_ofs);

        let mut at_level = None;
        for level in min_level..self.levels.len() {
            if self.levels[level]
                .splits
                .as_ref()
                .map(|s| !s.get(self.levels[level].index_of(block_ofs)))
                .unwrap_or(true)
            {
                at_level = Some(level);
                break;
            }
        }

        let at_level = at_level.expect("no level found to free block");

        let mut block = Some(ptr);
        for level in (0..=at_level).rev() {
            match block.take() {
                Some(b) => unsafe {
                    block = self.levels[level].deallocate(self.base, b, level != 0);
                },
                None => break,
            }
        }

        assert!(block.is_none(), "top level coalesced a block");
    }
}

struct AlignedRanges<I>
where
    I: Iterator<Item = Range<NonZeroUsize>>,
{
    align: usize,
    inner: Peekable<I>,
}

impl<I> AlignedRanges<I>
where
    I: Iterator<Item = Range<NonZeroUsize>>,
{
    fn new(iter: I, align: usize) -> AlignedRanges<I> {
        assert!(align.is_power_of_two());

        AlignedRanges {
            align,
            inner: iter.peekable(),
        }
    }
}

impl<I> Iterator for AlignedRanges<I>
where
    I: Iterator<Item = Range<NonZeroUsize>>,
{
    type Item = Range<NonZeroUsize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cur;
        loop {
            cur = self.inner.next()?;
            if !cur.is_empty() {
                break;
            }
        }

        let align = self.align;

        // Align start down.
        let start = cur.start.get() & !(align - 1);

        let mut end;
        loop {
            // Align end up.
            end = {
                let less_one = cur.end.get() - 1;
                let above = less_one
                    .checked_add(align)
                    .expect("end overflowed when aligned up");
                above & !(align - 1)
            };

            // Peek the next range.
            let next = match self.inner.peek() {
                Some(next) => next.clone(),
                None => break,
            };

            assert!(next.start >= cur.end);
            let next_start = next.start.get() & !(align - 1);

            if next_start <= end {
                assert!(next.end >= cur.end);

                // Merge contiguous ranges. `cur.end` will be aligned up on the
                // next loop iteration.
                cur.end = next.end;

                // Consume the peeked item.
                let _ = self.inner.next();
            } else {
                // The ranges are discontiguous.
                break;
            }
        }

        cur.start = NonZeroUsize::new(start).expect("start aligned down to null");
        cur.end = NonZeroUsize::new(end).unwrap();

        Some(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    use crate::core::{alloc::Layout, ptr::NonNull, slice};
    use std::prelude::rust_2021::*;

    #[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
    use crate::Global;

    #[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
    use alloc::alloc::Global;

    #[test]
    fn zero_levels_errors() {
        let _ = Buddy::<256, 0, Global>::try_new(8).unwrap_err();
    }

    #[test]
    fn overflow_address_space_errors() {
        let _ = Buddy::<256, 1, Global>::try_new(usize::MAX).unwrap_err();
    }

    #[test]
    fn too_many_levels_errors() {
        const LEVELS: usize = usize::BITS as usize;
        let _ = Buddy::<256, LEVELS, Global>::try_new(8).unwrap_err();
    }

    #[test]
    fn non_power_of_two_block_size_errors() {
        let _ = Buddy::<0, 1, Global>::try_new(8).unwrap_err();
        let _ = Buddy::<255, 4, Global>::try_new(8).unwrap_err();
    }

    #[test]
    fn too_small_min_block_size_errors() {
        const LEVELS: usize = 8;
        const MIN_SIZE: usize = core::mem::size_of::<usize>() / 2;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let _ = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap_err();
    }

    #[test]
    fn zero_blocks_errors() {
        Buddy::<128, 4, Global>::try_new(0).unwrap_err();
    }

    #[test]
    fn one_level() {
        Buddy::<16, 1, Global>::try_new(1).unwrap();
        Buddy::<128, 1, Global>::try_new(1).unwrap();
        Buddy::<4096, 1, Global>::try_new(1).unwrap();
    }

    #[test]
    fn create_and_destroy() {
        // These parameters give a maximum block size of 1KiB and a total size of 8KiB.
        const LEVELS: usize = 8;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();
        drop(allocator);
    }

    #[test]
    fn alloc_empty() {
        const LEVELS: usize = 4;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let mut allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();

        let layout = Layout::from_size_align(0, 1).unwrap();
        allocator.allocate(layout).unwrap_err();
    }

    #[test]
    fn alloc_min_size() {
        const LEVELS: usize = 4;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let mut allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();

        let layout = Layout::from_size_align(1, 1).unwrap();
        let a = allocator.allocate(layout).unwrap();
        let _b = allocator.allocate(layout).unwrap();
        let c = allocator.allocate(layout).unwrap();
        unsafe {
            allocator.deallocate(a.cast());
            allocator.deallocate(c.cast());
        }
    }

    #[test]
    fn alloc_write_and_free() {
        const LEVELS: usize = 8;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let mut allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();

        unsafe {
            let layout = Layout::from_size_align(64, MIN_SIZE).unwrap();
            let ptr: NonNull<u8> = allocator.allocate(layout).unwrap().cast();

            {
                // Do this in a separate scope so that the slice no longer
                // exists when ptr is freed
                let buf: &mut [u8] = slice::from_raw_parts_mut(ptr.as_ptr(), layout.size());
                for (i, byte) in buf.iter_mut().enumerate() {
                    *byte = i as u8;
                }
            }

            allocator.deallocate(ptr);
        }
    }

    #[test]
    fn coalesce_one() {
        // This configuration gives a 2-level buddy allocator with one
        // splittable top-level block.
        const LEVELS: usize = 2;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 1;

        let mut allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();

        let full_layout = Layout::from_size_align(2 * MIN_SIZE, MIN_SIZE).unwrap();
        let half_layout = Layout::from_size_align(MIN_SIZE, MIN_SIZE).unwrap();

        unsafe {
            // Allocate two minimum-size blocks to split the top block.
            let a = allocator.allocate(half_layout).unwrap();
            let b = allocator.allocate(half_layout).unwrap();

            // Free both blocks, coalescing them.
            allocator.deallocate(a.cast());
            allocator.deallocate(b.cast());

            // Allocate the entire region to ensure coalescing worked.
            let c = allocator.allocate(full_layout).unwrap();
            allocator.deallocate(c.cast());

            // Same as above.
            let a = allocator.allocate(half_layout).unwrap();
            let b = allocator.allocate(half_layout).unwrap();

            // Free both blocks, this time in reverse order.
            allocator.deallocate(a.cast());
            allocator.deallocate(b.cast());

            let c = allocator.allocate(full_layout).unwrap();
            allocator.deallocate(c.cast());
        }
    }

    #[test]
    fn coalesce_many() {
        const LEVELS: usize = 4;
        const MIN_SIZE: usize = 16;
        const BLK_SIZE: usize = MIN_SIZE << (LEVELS - 1);
        const NUM_BLOCKS: usize = 8;

        let mut allocator = Buddy::<BLK_SIZE, LEVELS, Global>::try_new(NUM_BLOCKS).unwrap();

        for lvl in (0..LEVELS).rev() {
            let alloc_size = 2usize.pow((LEVELS - lvl - 1) as u32) * MIN_SIZE;
            let layout = Layout::from_size_align(alloc_size, MIN_SIZE).unwrap();
            let num_allocs = 2usize.pow(lvl as u32) * NUM_BLOCKS;

            let mut allocs = Vec::with_capacity(num_allocs);
            for _ in 0..num_allocs {
                let ptr = allocator.allocate(layout).unwrap();

                {
                    // Do this in a separate scope so that the slice no longer
                    // exists when ptr is freed
                    let buf: &mut [u8] =
                        unsafe { slice::from_raw_parts_mut(ptr.as_ptr().cast(), layout.size()) };
                    for (i, byte) in buf.iter_mut().enumerate() {
                        *byte = (i % 256) as u8;
                    }
                }

                allocs.push(ptr);
            }

            for alloc in allocs {
                unsafe {
                    allocator.deallocate(alloc.cast());
                }
            }
        }
    }

    #[test]
    fn one_level_gaps() {
        type Alloc = Buddy<16, 1, Global>;
        let layout = Layout::from_size_align(1, 1).unwrap();

        for gaps in std::vec![[0..1], [15..16], [0..16], [15..32]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(1, gaps).unwrap();
            allocator.allocate(layout).unwrap_err();
        }

        for gaps in std::vec![[0..32], [0..17], [15..32], [15..17]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(2, gaps).unwrap();
            allocator.allocate(layout).unwrap_err();
        }

        for gaps in std::vec![[0..48], [0..33], [15..48], [15..33]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(3, gaps).unwrap();
            allocator.allocate(layout).unwrap_err();
        }

        for gaps in std::vec![[0..32], [0..17], [15..32], [15..17], [16..48], [16..33]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(3, gaps).unwrap();
            let a = allocator.allocate(layout).unwrap();
            unsafe { allocator.deallocate(a.cast()) };
        }

        let mut allocator = Alloc::try_new_with_offset_gaps(1, [16..32]).unwrap();
        let a = allocator.allocate(layout).unwrap();
        unsafe { allocator.deallocate(a.cast()) };
    }

    #[test]
    fn two_level_gaps() {
        type Alloc = Buddy<32, 2, Global>;
        let one_byte = Layout::from_size_align(1, 1).unwrap();
        let half = Layout::from_size_align(16, 16).unwrap();
        let full = Layout::from_size_align(32, 32).unwrap();

        for gaps in std::vec![[0..32], [0..17], [15..32], [8..24], [0..48], [15..48]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(1, gaps).unwrap();
            allocator.allocate(one_byte).unwrap_err();
        }

        for gaps in std::vec![[0..16], [16..32], [16..48]] {
            let mut allocator = Alloc::try_new_with_offset_gaps(1, gaps).unwrap();

            // Can't allocate the entire region.
            allocator.allocate(full).unwrap_err();

            // Can allocate the half-region not covered by the gap.
            let a = allocator.allocate(half).unwrap();
            // No memory left after that allocation.
            allocator.allocate(one_byte).unwrap_err();
            unsafe { allocator.deallocate(a.cast()) };

            // Ensure freeing doesn't cause coalescing into the gap.
            allocator.allocate(full).unwrap_err();
        }
    }

    #[test]
    fn three_level_gaps() {
        type Alloc = Buddy<128, 4, Global>;
        let layout = Layout::from_size_align(16, 4).unwrap();

        // X = gap, s = split, f = free
        // [       s       |       F       ]
        // [   s   |   s   |       |       ]
        // [ X | X | s | F |   |   |   |   ]
        // [ | | | |X|F| | | | | | | | | | ]
        let mut allocator = Alloc::try_new_with_offset_gaps(2, [0..72]).unwrap();
        std::println!("base = {:08x}", allocator.raw.base.addr().get());

        // [       s       |       F       ] 128
        // [   s   |   s   |       |       ]  64
        // [ X | X | s | F |   |   |   |   ]  32
        // [ | | | |X|a| | | | | | | | | | ]  16
        let a = allocator.allocate(layout).unwrap();
        std::println!("a = {:08x}", a.as_ptr() as *mut () as usize);
        assert_eq!(allocator.enumerate_free_list(0), 1);
        assert_eq!(allocator.enumerate_free_list(1), 0);
        assert_eq!(allocator.enumerate_free_list(2), 1);
        assert_eq!(allocator.enumerate_free_list(3), 0);

        // [       s       |       F       ] 128
        // [   s   |   s   |       |       ]  64
        // [ X | X | s | s |   |   |   |   ]  32
        // [ | | | |X|a|b|F| | | | | | | | ]  16
        let b = allocator.allocate(layout).unwrap();
        std::println!("b = {:08x}", b.as_ptr() as *mut () as usize);
        assert_eq!(allocator.enumerate_free_list(0), 1);
        assert_eq!(allocator.enumerate_free_list(1), 0);
        assert_eq!(allocator.enumerate_free_list(2), 0);
        assert_eq!(allocator.enumerate_free_list(3), 1);

        // [       s       |       F       ] 128
        // [   s   |   s   |       |       ]  64
        // [ X | X | s | s |   |   |   |   ]  32
        // [ | | | |X|F|b|F| | | | | | | | ]  16
        unsafe { allocator.deallocate(a.cast()) };
        assert_eq!(allocator.enumerate_free_list(0), 1);
        assert_eq!(allocator.enumerate_free_list(1), 0);
        assert_eq!(allocator.enumerate_free_list(2), 0);
        assert_eq!(allocator.enumerate_free_list(3), 2);
        for lev in allocator.raw.levels.iter() {
            for bit in lev.buddies.iter() {
                let s = if bit { "1" } else { "0" };

                std::print!("{s}");
            }

            std::println!();
        }

        // [       s       |       F       ] 128
        // [   s   |   s   |       |       ]  64
        // [ X | X | s | F |   |   |   |   ]  32
        // [ | | | |X|F| | | | | | | | | | ]  16
        unsafe { allocator.deallocate(b.cast()) };
    }

    #[test]
    fn add_coalesce() {
        const BLK_SIZE: usize = 128;
        const LEVELS: usize = 2;
        type Alloc = Buddy<BLK_SIZE, LEVELS, Raw>;
        const NUM_BLOCKS: usize = 1;

        let region_layout = Alloc::region_layout(NUM_BLOCKS).unwrap();
        let metadata_layout = Alloc::metadata_layout(NUM_BLOCKS).unwrap();
        let region = NonNull::new(unsafe { std::alloc::alloc(region_layout) }).unwrap();
        let metadata = NonNull::new(unsafe { std::alloc::alloc(metadata_layout) }).unwrap();

        let mut buddy =
            unsafe { Alloc::new_raw_unpopulated(metadata, region, NUM_BLOCKS).unwrap() };

        let base_addr = buddy.raw.base.addr();
        let middle = base_addr
            .get()
            .checked_add(BLK_SIZE / 2)
            .and_then(NonZeroUsize::new)
            .unwrap();
        let limit = buddy.raw.base.limit();

        let left = base_addr..middle;
        let right = middle..limit;

        let half_layout = Layout::from_size_align(BLK_SIZE / 2, BLK_SIZE / 2).unwrap();
        let full_layout = Layout::from_size_align(BLK_SIZE, BLK_SIZE).unwrap();

        // The allocator is unpopulated, so this should fail.
        buddy.allocate(half_layout).unwrap_err();

        // Populate the left block.
        unsafe { buddy.add_region(left) };

        // Now that the left half is populated, allocation should succeed.
        let left_blk = buddy.allocate(half_layout).unwrap();
        unsafe { buddy.deallocate(left_blk.cast()) };

        // Populate the right block. This should cause the blocks to coalesce.
        unsafe { buddy.add_region(right) };

        // Since both halves have been populated and coalesced, this should succeed.
        let full_blk = buddy.allocate(full_layout).unwrap();
        unsafe { buddy.deallocate(full_blk.cast()) };

        drop(buddy);

        unsafe {
            std::alloc::dealloc(region.as_ptr(), region_layout);
            std::alloc::dealloc(metadata.as_ptr(), metadata_layout);
        }
    }
}
