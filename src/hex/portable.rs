use crate::ct::{in_range, low, past};

#[inline(always)]
const fn symbol(nibble: i16) -> u8 {
    low(nibble.wrapping_add(b'0' as i16).wrapping_add(past(nibble, 9, 39)))
}

#[inline(always)]
pub(super) const fn nibble(byte: u8) -> i16 {
    let c = byte as i16;
    (-1_i16)
        .wrapping_add(in_range(c, b'0', b'9', c.wrapping_sub(47)))
        .wrapping_add(in_range(c, b'a', b'f', c.wrapping_sub(86)))
}

pub(super) fn encode(bytes: &[u8], out: &mut [u8]) {
    for (&byte, pair) in bytes.iter().zip(out.as_chunks_mut::<2>().0) {
        let byte = i16::from(byte);
        *pair = [symbol(byte >> 4), symbol(byte & 15)];
    }
}

pub(super) fn decode(text: &[u8], out: &mut [u8]) -> bool {
    let mut values = 0;
    for (&[hi, lo], byte) in text.as_chunks::<2>().0.iter().zip(out) {
        let (hi, lo) = (nibble(hi), nibble(lo));
        *byte = low(hi << 4 | lo);
        values |= hi | lo;
    }
    values >= 0
}
