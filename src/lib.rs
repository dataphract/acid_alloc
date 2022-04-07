//! Bare metal-friendly allocators.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "unstable", feature(alloc_layout_extra))]
#![cfg_attr(feature = "unstable", feature(allocator_api))]
#![cfg_attr(feature = "unstable", feature(int_log))]
#![cfg_attr(
    all(feature = "unstable", not(feature = "sptr")),
    feature(strict_provenance)
)]
//#![cfg_attr(feature = "docs_rs", feature(doc_cfg))]
#![feature(doc_cfg)]
// This is necessary to allow `sptr` and `polyfill` to shadow methods provided
// by unstable features.
#![allow(unstable_name_collisions)]

macro_rules! requires_sptr_or_unstable {
    ($($it:item)*) => {
        $(
            #[cfg(any(feature = "sptr", feature = "unstable"))]
            $it
        )*
    };
}

#[cfg(not(any(feature = "sptr", feature = "unstable")))]
compile_error!("At least one of these crate features must be enabled: [\"sptr\", \"unstable\"].");

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

requires_sptr_or_unstable! {
    mod bitmap;
    pub mod buddy;

    #[cfg(not(feature = "unstable"))]
    mod polyfill;

    #[cfg(test)]
    mod tests;

    use core::{alloc::Layout, num::NonZeroUsize, ptr::{self, NonNull}};

    #[cfg(feature = "unstable")]
    use core::alloc::Allocator;

    #[cfg(not(feature = "unstable"))]
    use sptr::Strict;

    pub use crate::buddy::BuddyAllocator;

    #[cfg(not(feature = "unstable"))]
    use crate::polyfill::*;

    /// The error type for allocator constructors.
    #[derive(Clone, Debug)]
    pub enum AllocInitError {
        /// A necessary allocation failed.
        ///
        /// This variant is returned when a constructor attempts to allocate
        /// memory, either for metadata or the managed region, but the
        /// underlying allocator fails.
        ///
        /// The variant contains the [`Layout`] that could not be allocated.
        AllocFailed(Layout),

        /// The configuration of the allocator is invalid.
        ///
        /// This variant is returned when an allocator's configuration
        /// parameters are impossible to satisfy.
        InvalidConfig,
    }

    /// A pointer to the base of the region of memory managed by an allocator.
    #[derive(Copy, Clone, Debug)]
    struct BasePtr {
        ptr: NonNull<u8>,
    }

    impl BasePtr {
        /// Calculates the offset from `self` to `block`.
        fn offset_to(self, block: NonZeroUsize) -> usize {
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
        unsafe fn init_link_at(self, addr: NonZeroUsize, link: BlockLink) {
            unsafe {
                self.with_addr(addr)
                    .cast::<BlockLink>()
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
        unsafe fn link_mut<'a>(self, link: NonZeroUsize) -> &'a mut BlockLink {
            unsafe { self.ptr.with_addr(link).cast::<BlockLink>().as_mut() }
        }

        /// Creates a new pointer with the given address.
        ///
        /// The returned pointer has the provenance of this pointer.
        fn with_addr(self, addr: NonZeroUsize) -> NonNull<u8> {
            self.ptr.with_addr(addr)
        }

        fn with_addr_and_size(self, addr: NonZeroUsize, len: usize) -> NonNull<[u8]> {
            let ptr = self.ptr.as_ptr().with_addr(addr.get());
            let raw_slice = ptr::slice_from_raw_parts_mut(ptr, len);

            unsafe { NonNull::new_unchecked(raw_slice) }
        }

        /// Creates a new pointer with the given offset.
        ///
        /// The returned pointer has the provenance of this pointer.
        fn with_offset(self, offset: usize) -> Option<NonNull<u8>> {
            let raw = self.ptr.addr().get().checked_add(offset)?;
            let addr = NonZeroUsize::new(raw)?;
            Some(self.ptr.with_addr(addr))
        }
    }

    /// A link in a linked list of blocks of memory.
    ///
    /// This type is meant to be embedded in the block itself, forming an intrusive
    /// linked list.
    #[repr(C)]
    struct BlockLink {
        // Rather than using pointers, store only the addresses of the previous and
        // next links.  This avoids accidentally violating stacked borrows; the
        // links "point to" other blocks, but by forgoing actual pointers, no borrow
        // is implied.
        //
        // NOTE: Using this method, any actual pointer to a block must be acquired
        // via the allocator base pointer, and NOT by casting these addresses
        // directly!
        prev: Option<NonZeroUsize>,
        next: Option<NonZeroUsize>,
    }

    /// Indicates an allocation failure due to resource exhaustion or an unsupported
    /// set of arguments.
    #[cfg(not(feature = "unstable"))]
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct AllocError;

    /// Types which provide memory which backs an allocator.
    ///
    /// This is a supertrait of [`Allocator`], and is implemented by the following types:
    /// - The `Raw` marker type indicates that an allocator is not backed by another
    ///   allocator. This is the case when constructing the allocator from raw
    ///   pointers. Memory used by this allocator can be reclaimed using
    ///   `.into_raw_parts()`.
    /// - The `Global` marker type indicates that an allocator is backed by the
    ///   global allocator. The allocator will free its memory on drop.
    /// - Any type `A` which implements [`Allocator`] indicates that an allocator is
    ///   backed by an instance of `A`. The allocator will free its memory on drop.
    ///
    /// [`Allocator`]: https://doc.rust-lang.org/stable/core/alloc/trait.Allocator.html
    pub trait BackingAllocator: Sealed {
        /// Deallocates the memory referenced by `ptr`.
        ///
        /// # Safety
        ///
        /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator, and
        /// * `layout` must [*fit*] that block of memory.
        ///
        /// [*currently allocated*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#currently-allocated-memory
        /// [*fit*]: https://doc.rust-lang.org/nightly/alloc/alloc/trait.Allocator.html#memory-fitting
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
    }

    /// A marker type indicating that an allocator is backed by raw pointers.
    #[derive(Clone, Debug)]
    pub struct Raw;
    impl Sealed for Raw {}
    impl BackingAllocator for Raw {
        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    #[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
    /// The global memory allocator.
    #[derive(Clone, Debug)]
    pub struct Global;

    #[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
    impl Sealed for Global {}

    #[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
    impl BackingAllocator for Global {
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }

    #[cfg(feature = "unstable")]
    impl<A: Allocator> Sealed for A {}
    #[cfg(feature = "unstable")]
    impl<A: Allocator> BackingAllocator for A {
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { Allocator::deallocate(self, ptr, layout) };
        }
    }

    #[doc(hidden)]
    mod private {
        pub trait Sealed {}
    }
    use private::Sealed;
}
