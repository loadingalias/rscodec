use super::block::{Block, Carry, Classes, Escapes};

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
    let code = ((potential << 1 | ODD).wrapping_sub(potential)) ^ ODD;
    let starts = code & backslash;
    Escapes {
        escaped: code ^ (backslash | escaped),
        starts,
    }
}

pub(super) const WHITESPACE: [u8; 16] = [
    b' ', 100, 100, 100, 17, 100, 113, 2, 100, b'\t', b'\n', 112, 100, b'\r', 100, 100,
];

pub(super) const STRUCTURAL: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b':', b'{', b',', b'}', 0, 0];
