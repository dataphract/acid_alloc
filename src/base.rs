use crate::core::{
    num::NonZeroUsize,
    ptr::{self, NonNull},
};

#[cfg(not(feature = "unstable"))]
use crate::core::ptr::{NonNullStrict, Strict};

/// A pointer to the base of the region of memory managed by an allocator.
#[derive(Copy, Clone, Debug)]
pub struct BasePtr {
    ptr: NonNull<u8>,
    extent: usize,
}

impl BasePtr {
    /// Creates a `BasePtr` from `ptr`.
    ///
    /// The returned value assumes the provenance of `ptr`.
    #[inline]
    pub fn new(ptr: NonNull<u8>, extent: usize) -> BasePtr {
        ptr.addr()
            .get()
            .checked_add(extent)
            .expect("region limit overflows usize");

        BasePtr { ptr, extent }
    }

    /// Returns the base pointer as a `NonNull<u8>`.
    #[inline]
    pub fn ptr(self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    pub fn limit(self) -> NonZeroUsize {
        NonZeroUsize::new(self.ptr.addr().get() + self.extent).unwrap()
    }

    #[inline]
    pub fn contains_addr(self, addr: NonZeroUsize) -> bool {
        self.ptr.addr() <= addr && addr < self.limit()
    }

    /// Returns the address of the base pointer.
    #[inline]
    pub fn addr(self) -> NonZeroUsize {
        self.ptr.addr()
    }

    /// Calculates the offset from `self` to `block`.
    pub fn offset_to(self, block: NonZeroUsize) -> usize {
        block.get().checked_sub(self.ptr.addr().get()).unwrap()
    }

    /// Initializes a `BlockLink` at the given address.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `addr` must be a properly aligned address for `BlockLink` values.
    /// - The memory at `addr` must be within the provenance of `self` and valid
    ///   for reads and writes for `size_of::<BlockLink>()` bytes.
    /// - The memory at `addr` must be unallocated by the associated allocator.
    #[inline]
    pub unsafe fn init_link_at(self, addr: NonZeroUsize, link: BlockLink) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.contains_addr(addr));
            if let Some(next) = link.next {
                debug_assert!(self.contains_addr(next), "next link out of region");
            }
        }

        unsafe {
            self.with_addr(addr)
                .cast::<BlockLink>()
                .as_ptr()
                .write(link)
        };
    }

    /// Initializes a `DoubleBlockLink` at the given address.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `addr` must be a properly aligned address for `DoubleBlockLink` values.
    /// - The memory at `addr` must be within the provenance of `self` and valid
    ///   for reads and writes for `size_of::<DoubleBlockLink>()` bytes.
    /// - The memory at `addr` must be unallocated by the associated allocator.
    #[inline]
    pub unsafe fn init_double_link_at(self, addr: NonZeroUsize, link: DoubleBlockLink) {
        debug_assert!(self.contains_addr(addr));

        debug_assert!(
            link.next.map_or(true, |next| self.contains_addr(next)),
            "next link out of region"
        );
        debug_assert!(
            link.prev.map_or(true, |prev| self.contains_addr(prev),),
            "prev link out of region"
        );

        unsafe {
            self.with_addr(addr)
                .cast::<DoubleBlockLink>()
                .as_ptr()
                .write(link)
        };
    }

    /// Returns a mutable reference to the `BlockLink` at `link`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `link` must be a properly aligned address for `BlockLink` values.
    /// - The memory at `link` must contain a properly initialized `BlockLink` value.
    /// - The memory at `link` must be within the provenance of `self` and
    ///   unallocated by the associated allocator.
    #[inline]
    pub unsafe fn link_mut<'a>(self, link: NonZeroUsize) -> &'a mut BlockLink {
        debug_assert!(self.contains_addr(link));

        unsafe { self.ptr.with_addr(link).cast::<BlockLink>().as_mut() }
    }

    /// Returns a mutable reference to the `DoubleBlockLink` at `link`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `link` must be a properly aligned address for `DoubleBlockLink` values.
    /// - The memory at `link` must contain a properly initialized `DoubleBlockLink` value.
    /// - The memory at `link` must be within the provenance of `self` and
    ///   unallocated by the associated allocator.
    #[inline]
    pub unsafe fn double_link_mut<'a>(self, link: NonZeroUsize) -> &'a mut DoubleBlockLink {
        debug_assert!(self.contains_addr(link));

        let link = unsafe { self.ptr.with_addr(link).cast::<DoubleBlockLink>().as_mut() };

        debug_assert!(
            link.next.map_or(true, |next| self.contains_addr(next)),
            "next link out of region"
        );
        debug_assert!(
            link.prev.map_or(true, |prev| self.contains_addr(prev),),
            "prev link out of region"
        );

        link
    }

    /// Creates a new pointer with the given address.
    ///
    /// The returned pointer has the provenance of this pointer.
    #[inline]
    pub fn with_addr(self, addr: NonZeroUsize) -> NonNull<u8> {
        debug_assert!(self.contains_addr(addr));

        self.ptr.with_addr(addr)
    }

    #[inline]
    pub fn with_addr_and_size(self, addr: NonZeroUsize, len: usize) -> NonNull<[u8]> {
        debug_assert!(self.contains_addr(addr));

        let ptr = self.ptr.as_ptr().with_addr(addr.get());
        let raw_slice = ptr::slice_from_raw_parts_mut(ptr, len);

        unsafe { NonNull::new_unchecked(raw_slice) }
    }

    /// Creates a new pointer with the given offset.
    ///
    /// The returned pointer has the provenance of this pointer.
    #[inline]
    pub fn with_offset(self, offset: usize) -> Option<NonNull<u8>> {
        let raw = self.ptr.addr().get().checked_add(offset)?;
        let addr = NonZeroUsize::new(raw)?;

        debug_assert!(self.contains_addr(addr));

        Some(self.ptr.with_addr(addr))
    }
}

// Rather than using pointers, store only the addresses of the previous and
// next links.  This avoids accidentally violating stacked borrows; the
// links "point to" other blocks, but by forgoing actual pointers, no borrow
// is implied.
//
// NOTE: Using this method, any actual pointer to a block must be acquired
// via the allocator base pointer, and NOT by casting these addresses
// directly!

/// A link in a linked list of blocks of memory.
///
/// This type is meant to be embedded in the block itself, forming an intrusive
/// linked list.
#[repr(C)]
pub struct BlockLink {
    pub next: Option<NonZeroUsize>,
}

/// A double link in a linked list of blocks of memory.
///
/// This type is meant to be embedded in the block itself, forming an intrusive
/// doubly linked list.
#[repr(C)]
#[derive(Debug)]
pub struct DoubleBlockLink {
    pub prev: Option<NonZeroUsize>,
    pub next: Option<NonZeroUsize>,
}
