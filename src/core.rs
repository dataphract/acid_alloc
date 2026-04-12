//! Re-exports of `core`, plus stable polyfills.
//!
//! The stable-compatible implementations in this module are copied more-or-less
//! verbatim from the standard library source.

#![cfg(not(feature = "unstable"))]

pub use core::{cmp, fmt, mem};

pub(crate) mod alloc {
    pub use core::alloc::*;

    // #![feature(alloc_layout_extra)]

    /// Indicates that an allocation failed.
    ///
    /// This may occur due to resource exhaustion or an unsupported set of arguments.
    ///
    /// When the `unstable` feature is enabled, this type is a re-export of
    /// [`core::alloc::AllocError`].
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct AllocError;
}

pub mod num {
    pub use core::num::*;
}

pub(crate) use core::ptr;
