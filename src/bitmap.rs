use crate::core::{alloc::Layout, mem};

#[cfg(not(feature = "unstable"))]
use crate::core::alloc::LayoutExt;

pub struct Bitmap {
    num_bits: usize,
    map: *mut u64,
}

impl Bitmap {
    pub fn map_layout(num_bits: usize) -> Layout {
        let num_blocks = Self::num_blocks(num_bits);

        Layout::new::<u64>()
            .repeat(num_blocks)
            .expect("bitmap metadata layout error")
            .0
    }

    /// Constructs a new bitmap of `len` bits, backed by `map`.
    ///
    /// A `Layout` describing a suitable region for `map` can be obtained with
    /// `PageFrameBitmap::map_layout(num_bits)`.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    /// - `map` must be valid for reads and writes for `len *
    ///   mem::size_of::<u64>()` many bytes, and it must be properly aligned.
    /// - `map` must point to `(num_bits + 63) / 64` consecutive properly
    ///   initialized `u64` values.
    pub unsafe fn new(num_bits: usize, map: *mut u64) -> Bitmap {
        assert!(num_bits > 0);
        assert!(!map.is_null());
        assert!(map.align_offset(mem::align_of::<u64>()) == 0);

        let num_blocks = Self::num_blocks(num_bits);

        for i in 0..(num_blocks as isize) {
            unsafe { map.offset(i).write(0) };
        }

        Bitmap { num_bits, map }
    }

    #[inline]
    // TODO: make const when Option::unwrap becomes const
    pub fn num_blocks(num_bits: usize) -> usize {
        (num_bits.checked_add(u64::BITS as usize - 1).unwrap())
            .checked_div(64)
            .unwrap()
    }

    /// Returns a tuple of the index of the `u64` containing `bit` and a mask
    /// which extracts it.
    #[inline]
    const fn index_and_mask(bit: usize) -> (usize, u64) {
        (
            bit / u64::BITS as usize,
            1 << (bit as u64 % u64::BITS as u64),
        )
    }

    /// Gets the value of the indexed bit.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.num_bits);

        let (block_idx, mask) = Self::index_and_mask(index);

        let block_idx: isize = block_idx
            .try_into()
            .expect("get: index overflowed an isize");

        unsafe { self.map.offset(block_idx).read() & mask != 0 }
    }

    /// Sets the value of the indexed bit.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.num_bits);

        let (block_idx, mask) = Self::index_and_mask(index);

        let block_idx: isize = block_idx
            .try_into()
            .expect("set: index overflowed an isize");

        unsafe {
            let block_ptr = self.map.offset(block_idx);
            let block = block_ptr.read();
            block_ptr.write(match value {
                true => block | mask,
                false => block & !mask,
            });
        }
    }

    /// Toggles the value of the indexed bit.
    #[inline]
    pub fn toggle(&mut self, index: usize) {
        assert!(index < self.num_bits);

        let (block_idx, mask) = Self::index_and_mask(index);

        let block_idx: isize = block_idx
            .try_into()
            .expect("toggle: index overflowed an isize");

        unsafe {
            let block_ptr = self.map.offset(block_idx);
            let block = block_ptr.read();
            block_ptr.write(block ^ mask);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::mem::ManuallyDrop;
    use std::prelude::rust_2021::*;

    use super::*;

    struct VecBitmap {
        bitmap: ManuallyDrop<Bitmap>,
        len: usize,
        cap: usize,
    }

    impl VecBitmap {
        fn new(num_bits: usize) -> VecBitmap {
            let num_blocks = Bitmap::num_blocks(num_bits);

            let mut v = Vec::with_capacity(num_blocks);
            v.resize(num_blocks, 0);

            // TODO: use Vec::into_raw_parts when stable
            let mut v = ManuallyDrop::new(v);
            let map = v.as_mut_ptr();
            let len = v.len();
            let cap = v.capacity();

            VecBitmap {
                bitmap: ManuallyDrop::new(unsafe { Bitmap::new(num_bits, map) }),
                len,
                cap,
            }
        }
    }

    impl Drop for VecBitmap {
        fn drop(&mut self) {
            unsafe {
                let Bitmap { map, .. } = ManuallyDrop::take(&mut self.bitmap);

                // Reconstitute the original Vec.
                let v = Vec::from_raw_parts(map, self.len, self.cap);

                // Explicit for clarity.
                drop(v);
            }
        }
    }

    #[test]
    fn init_many() {
        for num_bits in 1..=256 {
            let _ = VecBitmap::new(num_bits);
        }
    }
}
