//! The parts that the SIMD kernels share: the tables of the classes, and the escapes by
//! subtraction.

use super::block::{Block, Carry, Classes, Escapes};

/// The block of `classes` after `carry`, with the escapes of [`escapes`]. `prefix_xor` gives the
/// XOR of bits `0..=i` of its argument in each bit `i`.
#[inline(always)]
pub(super) fn scan(classes: Classes, carry: Carry, prefix_xor: impl Fn(u64) -> u64) -> Block {
    classes.block(escapes(classes.backslash, carry.escaped()), carry, prefix_xor)
}

/// The escapes of the `\` bytes in `backslash`, after a first byte that is escaped if `escaped`
/// is 1, by the subtraction of simdjson (`json_escape_scanner`). The subtraction finds the parity
/// of each run of `\` bytes with no loop.
#[inline(always)]
pub(super) fn escapes(backslash: u64, escaped: u64) -> Escapes {
    const ODD: u64 = 0xaaaa_aaaa_aaaa_aaaa;
    let potential = backslash & !escaped;
    // A 1 at each escaped byte and at each `\` at the start of an escape, and a 0 at the other
    // bytes.
    let code = ((potential << 1 | ODD).wrapping_sub(potential)) ^ ODD;
    let starts = code & backslash;
    Escapes {
        escaped: code ^ (backslash | escaped),
        starts,
    }
}

/// The byte of each whitespace character at the index of its low nibble, and at each other index a
/// byte with another low nibble. A byte under 0x80 is whitespace exactly when it equals the entry
/// of its low nibble.
pub(super) const WHITESPACE: [u8; 16] = [
    b' ', 100, 100, 100, 17, 100, 113, 2, 100, b'\t', b'\n', 112, 100, b'\r', 100, 100,
];

/// The structural byte, with bit 5 set, of each low nibble that has one, and 0 at the other
/// indexes. `b | 0x20` equals the entry of the low nibble of `b` for each structural byte `b`, and
/// for 0x0c and 0x1a, which are under 0x20.
pub(super) const STRUCTURAL: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b':', b'{', b',', b'}', 0, 0];
