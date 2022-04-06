//! Bare metal-friendly allocators.

#![no_std]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "unstable", feature(alloc_layout_extra))]
#![cfg_attr(feature = "unstable", feature(allocator_api))]
#![cfg_attr(feature = "unstable", feature(int_log))]
#![cfg_attr(
    all(feature = "unstable", not(feature = "sptr")),
    feature(strict_provenance)
)]
// This is necessary to allow `sptr` and `polyfill` to shadow methods provided
// by unstable features.
#![allow(unstable_name_collisions)]

#[cfg(not(any(feature = "sptr", feature = "unstable")))]
compile_error!("Either the \"sptr\" or \"unstable\" feature must be enabled.");

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

mod bitmap;
#[cfg(any(feature = "sptr", feature = "unstable"))]
mod buddy;
mod polyfill;

#[cfg(any(feature = "sptr", feature = "unstable"))]
pub use crate::buddy::BuddyAllocator;

#[cfg(all(any(feature = "sptr", feature = "unstable"), test))]
mod tests;

use core::{alloc::Layout, num::NonZeroUsize, ptr::NonNull};

#[cfg(any(feature = "sptr", feature = "unstable"))]
use core::ptr;

#[cfg(feature = "unstable")]
use core::alloc::Allocator;

#[cfg(feature = "sptr")]
use sptr::Strict;

#[cfg(all(feature = "sptr", not(feature = "unstable")))]
use crate::polyfill::*;

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

#[derive(Copy, Clone, Debug)]
struct BasePtr {
    ptr: NonNull<u8>,
}

/// A pointer to the base of the region of memory managed by an allocator.
#[cfg(any(feature = "sptr", feature = "unstable"))]
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

    fn with_addr_and_len(self, addr: NonZeroUsize, len: usize) -> NonNull<[u8]> {
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

/// Types which can be used to provide memory which backs an allocator.
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
pub struct Raw;
impl Sealed for Raw {}
impl BackingAllocator for Raw {
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
}

#[cfg(all(any(feature = "alloc", test), not(feature = "unstable")))]
/// The global memory allocator.
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
