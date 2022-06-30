//! Bump allocation.
//!
//! A bump allocator is a simple and fast allocator
//!
//! ## Characteristics
//!
//! #### Time complexity
//!
//! | Operation                | Best-case | Worst-case |
//! |--------------------------|-----------|------------|
//! | Allocate                 | O(1)      | O(1)       |
//! | Deallocate all           | O(1)      | O(1)       |
//!
//! #### Fragmentation
//!
//! Because bump allocators can allocate blocks of any size, they suffer minimal
//! internal fragmentation. External fragmentation correlates linearly with the
//! amount of deallocated data until all blocks are deallocated or the allocator
//! is manually reset.

use core::{fmt, ptr};

use crate::{
    core::{
        alloc::{AllocError, Layout},
        num::NonZeroUsize,
        ptr::NonNull,
    },
    AllocInitError, BackingAllocator, BasePtr, Raw,
};

#[cfg(feature = "unstable")]
use crate::core::alloc::Allocator;

#[cfg(not(feature = "unstable"))]
use crate::core::ptr::NonNullStrict;

#[cfg(any(feature = "alloc", test))]
use crate::Global;

/// A bump allocator.
pub struct Bump<A: BackingAllocator> {
    base: BasePtr,
    limit: NonZeroUsize,
    low_mark: NonZeroUsize,
    outstanding: usize,
    layout: Layout,
    backing_allocator: A,
}

impl Bump<Raw> {
    /// Constructs a new `Bump` from a raw pointer.
    ///
    /// # Safety
    ///
    /// `region` must be valid for reads and writes for `size` bytes.
    pub unsafe fn new_raw(
        region: NonNull<u8>,
        layout: Layout,
    ) -> Result<Bump<Raw>, AllocInitError> {
        unsafe { RawBump::try_new(region, layout).map(|b| b.with_backing_allocator(Raw)) }
    }

    /// Splits the allocator at the low mark.
    ///
    /// Returns a tuple of two `Bump`s. The first manages all of `self`'s
    /// unallocated memory, while the second manages all of `self`'s allocated
    /// memory; all prior allocations from `self` are backed by the second
    /// element.
    pub fn split(self) -> (Bump<Raw>, Bump<Raw>) {
        let base_addr = self.base.addr();
        let lower_size = self.low_mark.get().checked_sub(base_addr.get()).unwrap();
        let lower_limit = self.low_mark;
        let lower = Bump {
            base: BasePtr::new(self.base.ptr(), lower_size),
            limit: lower_limit,
            outstanding: 0,
            low_mark: lower_limit,
            layout: Layout::from_size_align(lower_size, 1).unwrap(),
            backing_allocator: Raw,
        };

        let upper_size = self.layout.size().checked_sub(lower_size).unwrap();
        let new_base = BasePtr::new(self.base.with_addr(self.low_mark), upper_size);
        let upper = Bump {
            base: new_base,
            limit: self.limit,
            low_mark: self.low_mark,
            outstanding: self.outstanding,
            // TODO: Alignment may be higher in some cases. Is that useful with Raw?
            layout: Layout::from_size_align(upper_size, 1).unwrap(),
            backing_allocator: Raw,
        };

        (lower, upper)
    }
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
impl Bump<Global> {
    /// Attempts to construct a new `Bump` backed by the global allocator.
    ///
    /// In particular, the memory managed by this `Bump` is allocated from the
    /// global allocator according to `layout`.
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    pub fn try_new(layout: Layout) -> Result<Bump<Global>, AllocInitError> {
        if layout.size() == 0 {
            return Err(AllocInitError::InvalidConfig);
        }

        unsafe {
            let region_raw = alloc::alloc::alloc(layout);
            let region_ptr = NonNull::new(region_raw).ok_or(AllocInitError::AllocFailed(layout))?;

            match RawBump::try_new(region_ptr, layout) {
                Ok(b) => Ok(b.with_backing_allocator(Global)),
                Err(e) => {
                    alloc::alloc::dealloc(region_ptr.as_ptr(), layout);
                    Err(e)
                }
            }
        }
    }
}

#[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
impl Bump<Global> {
    /// Attempts to construct a new `Bump` backed by the global allocator.
    ///
    /// In particular, the memory managed by this `Bump` is allocated from the
    /// global allocator according to `layout`.
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from the
    /// global allocator.
    pub fn try_new(layout: Layout) -> Result<Bump<Global>, AllocInitError> {
        Self::try_new_in(layout, Global)
    }
}

#[cfg(feature = "unstable")]
impl<A> Bump<A>
where
    A: Allocator,
{
    /// Attempts to construct a new `Bump` backed by `backing_allocator`.
    ///
    /// In particular, the memory managed by this `Bump` is allocated from
    /// `backing_allocator` according to `layout`.
    ///
    /// # Errors
    ///
    /// Returns an error if sufficient memory could not be allocated from
    /// `backing_allocator`.
    pub fn try_new_in(layout: Layout, backing_allocator: A) -> Result<Bump<A>, AllocInitError> {
        unsafe {
            let region_ptr = backing_allocator
                .allocate(layout)
                .map_err(|_| AllocInitError::AllocFailed(layout))?;

            match RawBump::try_new(region_ptr.cast(), layout) {
                Ok(b) => Ok(b.with_backing_allocator(backing_allocator)),
                Err(e) => {
                    backing_allocator.deallocate(region_ptr.cast(), layout);
                    Err(e)
                }
            }
        }
    }
}

impl<A> Bump<A>
where
    A: BackingAllocator,
{
    /// Attempts to allocate a block of memory according to `layout`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if there is insufficient memory remaining to accommodate
    /// `layout`.
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let new_low_unaligned = self
            .low_mark
            .get()
            .checked_sub(layout.size())
            .ok_or(AllocError)?;

        let new_low_mark = new_low_unaligned & !(layout.align() - 1);

        if new_low_mark < self.base.addr().get() {
            return Err(AllocError);
        }

        self.outstanding += 1;

        // SAFETY: new_low_mark >= base, which is non-null
        self.low_mark = unsafe { NonZeroUsize::new_unchecked(new_low_mark) };

        Ok(self.base.with_addr_and_size(self.low_mark, layout.size()))
    }

    /// Deallocates a block of memory.
    ///
    /// This operation does not increase the amount of available memory unless
    /// `ptr` is the last outstanding allocation from this allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    ///
    /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        drop(ptr);
        self.outstanding = self.outstanding.checked_sub(1).unwrap();

        if self.outstanding == 0 {
            // Reset the allocator.
            self.low_mark = self.limit;
        }
    }

    /// Resets the bump allocator.
    ///
    /// This method invalidates all outstanding allocations from this allocator.
    /// The destructors of allocated objects will not be run.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - No references to data allocated by this `Bump` may exist when the method
    ///   is called.
    /// - Any pointers to data previously allocated by this allocator may no
    ///   longer be dereferenced or passed to [`Bump::deallocate()`].
    ///
    /// [`Bump::deallocate()`]: Bump::deallocate
    pub unsafe fn reset(&mut self) {
        self.low_mark = self.limit;
    }

    /// Returns a pointer to the managed region.
    ///
    /// It is undefined behavior to dereference the returned pointer or upgrade
    /// it to a reference if there are any outstanding allocations.
    pub fn region(&mut self) -> NonNull<[u8]> {
        NonNull::new(ptr::slice_from_raw_parts_mut(
            self.base.ptr().as_ptr(),
            self.layout.size(),
        ))
        .unwrap()
    }
}

impl<A> Drop for Bump<A>
where
    A: BackingAllocator,
{
    fn drop(&mut self) {
        unsafe {
            self.backing_allocator
                .deallocate(self.base.ptr(), self.layout)
        };
    }
}

impl<A> fmt::Debug for Bump<A>
where
    A: BackingAllocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bump")
            .field("base", &self.base)
            .field("limit", &self.limit)
            .field("low_mark", &self.low_mark)
            .finish()
    }
}

struct RawBump {
    base: BasePtr,
    limit: NonZeroUsize,
    layout: Layout,
}

impl RawBump {
    fn with_backing_allocator<A: BackingAllocator>(self, backing_allocator: A) -> Bump<A> {
        Bump {
            base: self.base,
            limit: self.limit,
            low_mark: self.limit,
            outstanding: 0,
            layout: self.layout,
            backing_allocator,
        }
    }

    unsafe fn try_new(region: NonNull<u8>, layout: Layout) -> Result<RawBump, AllocInitError> {
        // Verify that the base pointer matches the layout.
        let addr = region.addr().get();
        if addr & !(layout.align() - 1) != addr {
            return Err(AllocInitError::InvalidConfig);
        }

        let base = BasePtr::new(region, layout.size());
        let limit = NonZeroUsize::new(
            region
                .addr()
                .get()
                .checked_add(layout.size())
                .ok_or(AllocInitError::InvalidLocation)?,
        )
        .unwrap();

        Ok(RawBump {
            base,
            limit,
            layout,
        })
    }
}
