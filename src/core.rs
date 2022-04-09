//! Re-exports of `core`, plus stable polyfills.
//!
//! The stable-compatible implementations in this module are copied more-or-less
//! verbatim from the standard library source.

#![cfg(not(feature = "unstable"))]

pub use core::{cmp, fmt, mem, slice, sync};

pub(crate) mod alloc {
    pub use core::alloc::*;

    // #![feature(alloc_layout_extra)]

    use core::ptr::NonNull;

    use sptr::invalid_mut;

    use crate::layout_error;

    /// Indicates an allocation failure due to resource exhaustion or an unsupported
    /// set of arguments.
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct AllocError;

    pub trait LayoutExt {
        fn padding_needed_for(&self, align: usize) -> usize;
        fn repeat(&self, n: usize) -> Result<(Self, usize), LayoutError>
        where
            Self: Sized;
        fn dangling(&self) -> NonNull<u8>;
    }

    impl LayoutExt for Layout {
        #[inline]
        fn padding_needed_for(&self, align: usize) -> usize {
            let len = self.size();

            // Rounded up value is:
            //   len_rounded_up = (len + align - 1) & !(align - 1);
            // and then we return the padding difference: `len_rounded_up - len`.
            //
            // We use modular arithmetic throughout:
            //
            // 1. align is guaranteed to be > 0, so align - 1 is always
            //    valid.
            //
            // 2. `len + align - 1` can overflow by at most `align - 1`,
            //    so the &-mask with `!(align - 1)` will ensure that in the
            //    case of overflow, `len_rounded_up` will itself be 0.
            //    Thus the returned padding, when added to `len`, yields 0,
            //    which trivially satisfies the alignment `align`.
            //
            // (Of course, attempts to allocate blocks of memory whose
            // size and padding overflow in the above manner should cause
            // the allocator to yield an error anyway.)

            let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
            len_rounded_up.wrapping_sub(len)
        }

        #[inline]
        fn repeat(&self, n: usize) -> Result<(Self, usize), LayoutError> {
            // This cannot overflow. Quoting from the invariant of Layout:
            // > `size`, when rounded up to the nearest multiple of `align`,
            // > must not overflow (i.e., the rounded value must be less than
            // > `usize::MAX`)
            let padded_size = self.size() + self.padding_needed_for(self.align());
            let alloc_size = padded_size.checked_mul(n).ok_or_else(layout_error)?;

            // SAFETY: self.align is already known to be valid and alloc_size has been
            // padded already.
            unsafe {
                Ok((
                    Layout::from_size_align_unchecked(alloc_size, self.align()),
                    padded_size,
                ))
            }
        }

        #[inline]
        #[cfg(feature = "sptr")]
        fn dangling(&self) -> NonNull<u8> {
            // SAFETY: align is guaranteed to be non-zero
            unsafe { NonNull::new_unchecked(invalid_mut::<u8>(self.align())) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn layout_error_returns_error() {
            let _: LayoutError = layout_error();
        }
    }
}

pub mod num {
    pub use core::num::*;

    // #![feature(int_log)]

    pub trait UsizeExt {
        fn log2(self) -> u32;
    }

    impl UsizeExt for usize {
        #[inline]
        fn log2(self) -> u32 {
            Self::BITS - 1 - self.leading_zeros()
        }
    }
}

#[cfg(not(feature = "sptr"))]
pub(crate) use core::ptr;

#[cfg(feature = "sptr")]
pub(crate) mod ptr {
    pub use core::ptr::*;

    // #![feature(strict_provenance)]

    use core::num::NonZeroUsize;

    pub use sptr::Strict;

    pub trait NonNullStrict<T> {
        fn addr(self) -> NonZeroUsize
        where
            T: Sized;

        fn with_addr(self, addr: NonZeroUsize) -> Self
        where
            T: Sized;

        fn map_addr(self, f: impl FnOnce(NonZeroUsize) -> NonZeroUsize) -> Self
        where
            T: Sized;
    }

    impl<T> NonNullStrict<T> for NonNull<T> {
        fn addr(self) -> NonZeroUsize
        where
            T: Sized,
        {
            // SAFETY: The pointer is guaranteed by the type to be non-null,
            // meaning that the address will be non-zero.
            unsafe { NonZeroUsize::new_unchecked(self.as_ptr().addr()) }
        }

        fn with_addr(self, addr: NonZeroUsize) -> Self
        where
            T: Sized,
        {
            // SAFETY: The result of `ptr::from::with_addr` is non-null because `addr` is guaranteed to be non-zero.
            unsafe { NonNull::new_unchecked(self.as_ptr().with_addr(addr.get()) as *mut _) }
        }

        fn map_addr(self, f: impl FnOnce(NonZeroUsize) -> NonZeroUsize) -> Self
        where
            T: Sized,
        {
            self.with_addr(f(self.addr()))
        }
    }
}
