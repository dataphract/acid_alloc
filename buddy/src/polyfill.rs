#[cfg(not(feature = "int_log"))]
pub trait UsizeExt {
    fn log2(self) -> u32;
}

#[cfg(not(feature = "int_log"))]
impl UsizeExt for usize {
    fn log2(self) -> u32 {
        Self::BITS - 1 - self.leading_zeros()
    }
}
