//! Arithmetic with no branch and no table: the operations of the portable backends.
//!
//! Each function takes and gives values in -256..=255, so no operation overflows. The wrapping
//! operations add no overflow check, which is a branch on the data.

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
#[inline]
pub(crate) fn first_negative(values: impl DoubleEndedIterator<Item = i16> + ExactSizeIterator) -> usize {
    let mut first = values.len();
    for (offset, value) in values.enumerate().rev() {
        let bad = negative_mask(value);
        first = (offset & bad) | (first & !bad);
    }
    first
}
