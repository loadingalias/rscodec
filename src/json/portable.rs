//! The 64-bit SWAR kernel: the reference of the SIMD kernels, and the kernel of a CPU with no SIMD
//! kernel. It finds the escapes by a walk over the `\` bytes, not by the subtraction of the SIMD
//! kernels, so the two methods check each other.

use super::block::{self, Block, Carry, Classes, Escapes};

/// 0x01 in each byte.
const ONES: u64 = u64::from_ne_bytes([0x01; 8]);
/// 0x80 in each byte.
const HIGH: u64 = u64::from_ne_bytes([0x80; 8]);

/// The block of the first 64 bytes of `bytes`, with spaces after them when `bytes` has fewer, after
/// the 4 bytes `before` in little-endian order and after `carry`.
#[inline(always)]
pub(super) fn block(bytes: &[u8], before: u32, carry: Carry) -> Block {
    let words = bytes.first_chunk::<64>().map_or_else(
        || block::tail(bytes),
        |bytes| core::array::from_fn(|index| u64::from_le_bytes(bytes.as_chunks::<8>().0[index])),
    );
    let classes = classify(words, before);
    classes.block(escapes(classes.backslash, carry.escaped()), carry, block::prefix_xor)
}

/// The classes of the little-endian `words`.
#[inline]
fn classify(words: [u64; 8], before: u32) -> Classes {
    let mut classes = Classes::default();
    for (index, word) in words.into_iter().enumerate() {
        let equal = |byte: u8| zeros(word ^ (ONES * u64::from(byte)));
        // `|` 0x20 makes `[` and `]` into `{` and `}`, and no other byte.
        let curly = |byte: u8| zeros((word | (ONES * 0x20)) ^ (ONES * u64::from(byte)));
        let set = |mask: &mut u64, bytes: u64| *mask |= pack(bytes) << (8 * index);
        set(
            &mut classes.structural,
            curly(b'{') | curly(b'}') | equal(b':') | equal(b','),
        );
        set(
            &mut classes.whitespace,
            equal(b' ') | equal(b'\t') | equal(b'\n') | equal(b'\r'),
        );
        set(&mut classes.quote, equal(b'"'));
        set(&mut classes.backslash, equal(b'\\'));
        set(&mut classes.control, zeros(word & (ONES * 0xe0)));
    }
    classes.utf8_error = utf8_error(words, before);
    classes
}

/// 0x80 in each byte of `word` that is zero, and zero in each other byte. No carry passes from one
/// byte to the next.
#[inline]
const fn zeros(word: u64) -> u64 {
    !(((word & !HIGH) + !HIGH) | word) & HIGH
}

/// The bit 7 of each byte of `bytes` in bits 0 to 7 of the result, from byte 0. The product puts
/// each bit in a different place, so no carry changes a bit of the result.
#[inline]
const fn pack(bytes: u64) -> u64 {
    (bytes >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

/// The UTF-8 check of each byte of `words`. In a block of ASCII, only the first three bytes can
/// end an error, and they end one exactly when `before` ends incomplete.
#[inline]
fn utf8_error(words: [u64; 8], before: u32) -> bool {
    if words.iter().fold(0, |any, word| any | word) & HIGH == 0 {
        return block::ends_incomplete(before);
    }
    let [_, mut first, mut second, mut third] = before.to_le_bytes();
    let mut error = false;
    for byte in words.into_iter().flat_map(u64::to_le_bytes) {
        error |= block::utf8_error_at(first, second, third, byte);
        (first, second, third) = (second, third, byte);
    }
    error
}

/// The escapes of `backslash`, after a first byte that is escaped if `escaped` is 1, by a walk from
/// its lowest `\`. A `\` that is not escaped starts an escape, and it escapes the next byte.
#[inline]
fn escapes(backslash: u64, escaped: u64) -> Escapes {
    let mut escapes = Escapes { escaped, starts: 0 };
    let mut rest = backslash & !escaped;
    while rest != 0 {
        let start = rest.isolate_lowest_one();
        escapes.starts |= start;
        escapes.escaped |= start << 1;
        rest &= !(start | start << 1);
    }
    escapes
}
