//! Arithmetic with no branch and no table: the operations of the portable backends. Also the
//! timing class as a type, with the stores of zero of the secret class.
//!
//! Each arithmetic function takes and gives values in -256..=255, so no operation overflows. The
//! wrapping operations add no overflow check, which is a branch on the data.

/// `plus` if `lo <= c <= hi`, and zero if not. `(lo - 1 - c) & (c - hi - 1)` is negative only in the
/// range, so its shift right by 8 is -1 in the range and 0 outside it.
#[inline(always)]
pub(crate) const fn in_range(c: i16, lo: u8, hi: u8, plus: i16) -> i16 {
    ((lo as i16 - 1).wrapping_sub(c) & c.wrapping_sub(hi as i16 + 1)) >> 8 & plus
}

/// `plus` if `value` is past `last`, and zero if not: `(last - value) >> 8` is -1 past it and 0 up
/// to it.
#[inline(always)]
pub(crate) const fn past(value: i16, last: i16, plus: i16) -> i16 {
    (last.wrapping_sub(value) >> 8) & plus
}

/// The low byte of a value.
#[inline(always)]
pub(crate) const fn low(value: i16) -> u8 {
    value.to_le_bytes()[0]
}

/// All ones if `value` is negative, and zero if not.
#[inline(always)]
pub(crate) const fn negative_mask(value: i16) -> usize {
    0_usize.wrapping_sub((value.cast_unsigned() >> 15) as usize)
}

/// The offset of the first value of `values` that is negative, or `values.len()` if none is. The
/// time depends only on the number of values.
///
/// It counts the values before the first negative one: a mask turns on at a negative value and
/// stays on, and the count stops when it turns on. Only `or`, `and`, and `add` run, with no select
/// and no minimum, which a compiler can lower to a branch or to a load at an offset from the data:
/// LLVM made a branch of `usize::min` on x86_64, and a load from a stack slot of the select of the
/// last negative offset on aarch64 (C103).
#[inline]
pub(crate) fn first_negative(values: impl Iterator<Item = i16>) -> usize {
    let mut seen = 0_usize;
    let mut first = 0_usize;
    for value in values {
        seen |= negative_mask(value);
        // The values of a slice are fewer than `usize::MAX`, so the count does not wrap. A checked
        // add would branch on it.
        first = first.wrapping_add(!seen & 1);
    }
    first
}

/// A timing class as a type: [`Public`] or [`Secret`]. A kernel that takes one compiles a copy for
/// each class. Thus the copy of the public class has no store of the secret class.
pub(crate) trait Class: Copy {
    /// In the secret class, writes zero to `array`, a temporary copy of input or result bytes,
    /// before its function returns. In the public class, it does nothing.
    #[cfg_attr(
        not(any(
            feature = "base64",
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(dead_code, reason = "hex with no SIMD kernel has no temporary array")
    )]
    fn clear<const N: usize>(self, array: &mut [u8; N]);

    /// In the secret class, writes zero to `output` after a decoder refuses its text. In the
    /// public class, it does nothing.
    fn wipe(self, output: &mut [u8]);
}

/// The class of the functions for public data.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Public;

/// The class of the functions for secret data.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Secret;

impl Class for Public {
    #[inline(always)]
    fn clear<const N: usize>(self, _: &mut [u8; N]) {}

    #[inline(always)]
    fn wipe(self, _: &mut [u8]) {}
}

impl Class for Secret {
    #[inline(always)]
    fn clear<const N: usize>(self, array: &mut [u8; N]) {
        zero::array(array);
    }

    #[inline]
    fn wipe(self, output: &mut [u8]) {
        zero::bytes(output);
    }
}

/// The stores of zero of the secret class. Each store is volatile, so the optimizer cannot remove
/// it, also when the caller drops the memory next. The compiler fence keeps each store in its place
/// among the other memory accesses.
#[expect(
    unsafe_code,
    reason = "each volatile store writes only memory that the function borrows mutably"
)]
mod zero {
    use core::{
        ptr::write_volatile,
        sync::atomic::{Ordering::SeqCst, compiler_fence},
    };

    /// Writes zero to `array` with one store of the whole array.
    #[cfg_attr(
        not(any(
            feature = "base64",
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(dead_code, reason = "hex with no SIMD kernel has no temporary array")
    )]
    #[inline(always)]
    pub(super) fn array<const N: usize>(array: &mut [u8; N]) {
        // SAFETY: `array` is a valid, aligned, and exclusive reference to `N` bytes.
        unsafe { write_volatile(array, [0; N]) };
        compiler_fence(SeqCst);
    }

    /// Writes zero to each byte of `bytes`, a slice of any length.
    #[inline]
    pub(super) fn bytes(bytes: &mut [u8]) {
        for byte in bytes {
            // SAFETY: `byte` is a valid, aligned, and exclusive reference to one `u8`.
            unsafe { write_volatile(byte, 0) };
        }
        compiler_fence(SeqCst);
    }
}
