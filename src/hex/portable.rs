//! The portable backend and the reference of each SIMD backend: arithmetic with no branch and no
//! table on the data.

use crate::ct::{in_range, low, past};

/// The character of a nibble: `0`–`9`, then `a`–`f`.
#[inline(always)]
const fn symbol(nibble: i16) -> u8 {
    low(nibble.wrapping_add(b'0' as i16).wrapping_add(past(nibble, 9, 39)))
}

/// The nibble of a character, or -1 for a byte that is not `0`–`9` or `a`–`f`. Each term is the
/// nibble plus one in the range of its characters, and zero outside it.
#[inline(always)]
pub(super) const fn nibble(byte: u8) -> i16 {
    let c = byte as i16;
    (-1_i16)
        .wrapping_add(in_range(c, b'0', b'9', c.wrapping_sub(47)))
        .wrapping_add(in_range(c, b'a', b'f', c.wrapping_sub(86)))
}

/// Encodes `bytes` into `out`, which has twice the length.
pub(super) fn encode(bytes: &[u8], out: &mut [u8]) {
    for (&byte, pair) in bytes.iter().zip(out.as_chunks_mut::<2>().0) {
        let byte = i16::from(byte);
        *pair = [symbol(byte >> 4), symbol(byte & 15)];
    }
}

/// Decodes `text` into `out`, which has half the length. Gives whether each character is a hex
/// digit.
pub(super) fn decode(text: &[u8], out: &mut [u8]) -> bool {
    let mut values = 0;
    for (&[hi, lo], byte) in text.as_chunks::<2>().0.iter().zip(out) {
        let (hi, lo) = (nibble(hi), nibble(lo));
        *byte = low(hi << 4 | lo);
        values |= hi | lo;
    }
    values >= 0
}
