//! Bare-metal allocators.
//!
//! ---
//! This crate provides allocators that are suitable for use on bare metal or with low-level
//! allocation facilities like `mmap(2)`/`brk(2)`.
//!
//! ## Allocators
//!
//! The following allocators are available:
//!
//! - **[`Buddy`], a binary-buddy allocator**. O(log<sub>2</sub>_levels_) worst-case allocation and
//!   deallocation. Supports splitting and coalescing blocks by powers of 2. Good choice for
//!   periodic medium-to-large allocations.
//! - **[`Bump`], a bump allocator**. O(1) allocation. Extremely fast to allocate and flexible in
//!   terms of allocation layout, but unable to deallocate individual items. Good choice for
//!   allocations that will never be deallocated or that will be deallocated en masse.
//! - **[`Slab`], a slab allocator**. O(1) allocation and deallocation. All allocated blocks are the
//!   same size, making this allocator a good choice when allocating many similarly-sized objects.
//!
//! ## Features
//!
//! All allocators provided by this crate are available in a `#![no_std]`,
//! `#![cfg(no_global_oom_handling)]` environment. Additional functionality is available when
//! enabling feature flags:
//!
//! <table>
//!  <tr>
//!   <th>Flag</th>
//!   <th>Default?</th>
//!   <th>Requires nightly?</th>
//!   <th>Description</th>
//!  </tr>
//!  <tr><!-- sptr -->
//!   <td><code>sptr</code></td>
//!   <td>Yes</td>
//!   <td>No</td>
//!   <td>
//!    Uses the <a href="https://crates.io/crates/sptr"><code>sptr</code></a> polyfill for Strict Provenance.
//!   </td>
//!  </tr>
//!  <tr>
//!   <td><code>unstable</code></td>
//!   <td>No</td>
//!   <td>Yes</td>
//!   <td>
//!    Exposes constructors for allocators backed by implementors of the
//!    unstable<code>Allocator</code> trait, and enables the internal use of
//!    nightly-only Rust features. Obviates <code>sptr</code>.
//!   </td>
//!  </tr>
//!  <tr>
//!   <td><code>alloc</code></td>
//!   <td>No</td>
//!   <td>No</td>
//!   <td>
//!    Exposes constructors for allocators backed by the global allocator.
//!   </td>
//!  </tr>
//! </table>
//!
//! [`sptr`]: https://crates.io/crates/sptr

#![no_std]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "unstable", feature(alloc_layout_extra))]
#![cfg_attr(feature = "unstable", feature(allocator_api))]
#![cfg_attr(feature = "unstable", feature(int_log))]
#![cfg_attr(feature = "unstable", feature(strict_provenance))]
#![cfg_attr(docs_rs, feature(doc_cfg))]
// This is necessary to allow `sptr` and `crate::core` to shadow methods provided by unstable
// features.
#![allow(unstable_name_collisions)]

#[cfg(test)]
extern crate std;

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
    mod base;
    mod bitmap;
    pub mod buddy;
    pub mod bump;
    pub mod slab;

    #[cfg(test)]
    mod tests;

    #[cfg(not(feature = "unstable"))]
    pub(crate) mod core;

    #[cfg(feature = "unstable")]
    pub(crate) mod core {
        pub use core::{alloc, cmp, fmt, mem, num, ptr, slice, sync};
    }

    use crate::{
        base::{BasePtr, BlockLink, DoubleBlockLink},
        core::{
            alloc::{Layout},
            ptr::NonNull,
        },
    };

    #[cfg(not(feature = "unstable"))]
    use crate::core::alloc::LayoutError;

    #[cfg(feature = "unstable")]
    use crate::core::alloc::Allocator;

    #[doc(inline)]
    pub use crate::{buddy::Buddy, bump::Bump, core::alloc::AllocError, slab::Slab};

    #[cfg(not(feature = "unstable"))]
    pub(crate) fn layout_error() -> LayoutError {
        // HACK: LayoutError is #[non_exhaustive], so it can't be constructed outside the standard
        // library. As a workaround, deliberately pass bad values to the constructor to get one.
        Layout::from_size_align(0, 0).unwrap_err()
    }

    /// The error type for allocator constructors.
    #[derive(Clone, Debug)]
    pub enum AllocInitError {
        /// A necessary allocation failed.
        ///
        /// This variant is returned when a constructor attempts to allocate memory, either for
        /// metadata or the managed region, but the underlying allocator fails.
        ///
        /// The variant contains the [`Layout`] that could not be allocated.
        AllocFailed(Layout),

        /// The configuration of the allocator is invalid.
        ///
        /// This variant is returned when an allocator's configuration parameters are impossible to
        /// satisfy.
        InvalidConfig,

        /// The location of the allocator is invalid.
        ///
        /// This variant is returned when the full size of the managed region would not fit at the
        /// provided address, i.e., pointer calculations would overflow.
        InvalidLocation,
    }

    /// Types which provide memory which backs an allocator.
    ///
    /// This is a supertrait of [`Allocator`], and is implemented by the following types:
    /// - The `Raw` marker type indicates that an allocator is not backed by another allocator. This
    ///   is the case when constructing the allocator from raw pointers. Memory used by this
    ///   allocator can be reclaimed using `.into_raw_parts()`.
    /// - The `Global` marker type indicates that an allocator is backed by the global allocator.
    ///   The allocator will free its memory on drop.
    /// - Any type `A` which implements [`Allocator`] indicates that an allocator is backed by an
    ///   instance of `A`. The allocator will free its memory on drop.
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

    #[cfg(all(any(feature = "alloc", test), feature = "unstable"))]
    pub use alloc::alloc::Global;

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
