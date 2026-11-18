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
#[inline]
pub(super) fn block(bytes: &[u8], before: u32, mut carry: Carry) -> Block {
    let words = bytes.first_chunk::<64>().map_or_else(
        || block::tail(bytes),
        |bytes| core::array::from_fn(|index| u64::from_le_bytes(bytes.as_chunks::<8>().0[index])),
    );
    let classes = classify(words, before);
    let escapes = escapes(classes.backslash, &mut carry);
    classes.block(escapes, carry, block::prefix_xor)
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

/// The escapes of `backslash`, by a walk from its lowest `\`. A `\` that is not escaped starts an
/// escape, and it escapes the next byte.
#[inline]
fn escapes(backslash: u64, carry: &mut Carry) -> Escapes {
    let mut escapes = Escapes {
        escaped: carry.escaped,
        starts: 0,
    };
    let mut rest = backslash & !carry.escaped;
    while rest != 0 {
        let start = rest.isolate_lowest_one();
        escapes.starts |= start;
        escapes.escaped |= start << 1;
        rest &= !(start | start << 1);
    }
    carry.escaped = escapes.starts >> 63;
    escapes
}

/// The proofs of the UTF-8 check of the reference, for `cargo kani`, on each input of up to
/// `BYTES_MAX` bytes: in one block, and across the end of a block.
#[cfg(kani)]
mod proofs {
    use super::{Carry, block};
    use crate::json::block::ends_incomplete;

    /// The largest input of the proofs.
    const BYTES_MAX: usize = 6;

    /// The UTF-8 error of the block of `bytes` after `before`.
    fn error(bytes: &[u8], before: u32) -> bool {
        block(bytes, before, Carry::default()).classes.utf8_error
    }

    /// An input of up to `BYTES_MAX` bytes, the last block, has an error exactly when it is not
    /// UTF-8.
    #[kani::proof]
    #[kani::unwind(65)]
    fn the_check_of_one_block_equals_std() {
        let input: [u8; BYTES_MAX] = kani::any();
        let len = kani::any_where(|&len: &usize| len <= BYTES_MAX);
        assert!(error(&input[..len], 0) == core::str::from_utf8(&input[..len]).is_err());
    }

    /// An input of `BYTES_MAX` bytes across the end of a block, with spaces before it, has an error
    /// in one of the two blocks exactly when it is not UTF-8. When the input ends at the end of the
    /// first block, `ends_incomplete` of its last bytes replaces the second block.
    #[kani::proof]
    #[kani::unwind(65)]
    fn the_check_of_two_blocks_equals_std() {
        let input: [u8; BYTES_MAX] = kani::any();
        let split = kani::any_where(|&split: &usize| split <= BYTES_MAX);
        let mut first = [b' '; 64];
        first[64 - split..].copy_from_slice(&input[..split]);
        let before = u32::from_le_bytes([first[60], first[61], first[62], first[63]]);
        let whole = error(&first, 0) || error(&input[split..], before);
        assert!(whole == core::str::from_utf8(&input).is_err());
        let ends = error(&first, 0) || ends_incomplete(before);
        assert!(ends == core::str::from_utf8(&input[..split]).is_err());
    }
}
