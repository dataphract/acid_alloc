//! A binary-buddy allocator.

#![no_std]
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
use core::mem::{self, MaybeUninit};

#[cfg(feature = "unstable")]
use core::alloc::Allocator;

#[cfg(all(feature = "unstable", any(feature = "alloc", test)))]
use alloc::alloc::Global;

#[cfg(feature = "sptr")]
use sptr::Strict;

#[cfg(all(feature = "sptr", not(feature = "unstable")))]
use crate::polyfill::*;

/// A link in a linked list of buddy blocks.
///
/// This type is meant to be embedded in the block itself.
#[repr(C)]
struct BuddyLink {
    // Rather than using a pointer, store only the address of the next link.
    // This avoids accidentally violating stacked borrows; the link "points to"
    // the next block, but by forgoing an actual pointer, it does not imply a
    // borrow.
    //
    // NOTE: Using this method, any actual pointer to a block must be acquired
    // via the allocator base pointer, and NOT by casting this address directly!
    next: Option<NonZeroUsize>,
}

#[cfg(any(feature = "sptr", feature = "unstable"))]
impl BuddyLink {
    /// Initializes a `BuddyLink` at the pointed-to location.
    ///
    /// The address of the initialized `BuddyLink` is returned rather than the
    /// pointer. This indicates to the compiler that any effective mutable
    /// borrow of `ptr` has ended.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `ptr` must be valid for reads and writes for `size_of::<BuddyLink>()`
    ///   bytes.
    /// - If `next` is `Some(n)`, then `n` must be the address of an
    ///   initialized `BuddyLink` value.
    unsafe fn init(
        mut ptr: NonNull<MaybeUninit<BuddyLink>>,
        next: Option<NonZeroUsize>,
    ) -> NonZeroUsize {
        assert_eq!(ptr.as_ptr().align_offset(mem::align_of::<BuddyLink>()), 0);

        let addr = unsafe {
            let uninit_mut: &mut MaybeUninit<_> = ptr.as_mut();
            let ptr = uninit_mut.as_mut_ptr();
            ptr.write(BuddyLink { next });
            ptr.addr()
        };

        NonZeroUsize::new(addr).unwrap()
    }
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

    /// Returns a mutable reference to the `BuddyLink` at `link`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `link` must be a properly aligned address for `BuddyLink` values.
    /// - The memory at `link` must contain a properly initialized `BuddyLink` value.
    /// - The memory at `link` must be unallocated by the associated allocator.
    unsafe fn link_mut<'a>(self, link: NonZeroUsize) -> &'a mut BuddyLink {
        unsafe { self.ptr.with_addr(link).cast::<BuddyLink>().as_mut() }
    }

    /// Creates a new pointer with the given address.
    ///
    /// The returned pointer has the provenance of this pointer.
    fn with_addr(self, addr: NonZeroUsize) -> NonNull<u8> {
        self.ptr.with_addr(addr)
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
