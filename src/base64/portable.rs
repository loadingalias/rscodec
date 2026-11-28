//! The portable backend and the reference of each SIMD backend: arithmetic with no branch and no
//! table on the data.

use super::Profile;
use crate::ct::{Class, in_range, low, past};

/// The two characters of a profile after `9`, and the steps of the encoder to them.
#[derive(Clone, Copy)]
struct Alphabet {
    /// The characters of the sextets 62 and 63.
    c62: u8,
    c63: u8,
    /// The steps from the character of the sextet before.
    step62: i16,
    step63: i16,
}

impl Alphabet {
    #[inline(always)]
    const fn of(profile: Profile) -> Self {
        match profile {
            Profile::Url => Self {
                c62: b'-',
                c63: b'_',
                step62: -13,
                step63: 49,
            },
            Profile::Standard => Self {
                c62: b'+',
                c63: b'/',
                step62: -15,
                step63: 3,
            },
        }
    }

    /// The sextet (6-bit value) of a character, or -1 for a byte outside the alphabet. Each term is
    /// the sextet plus one in the range of its characters, and zero outside it.
    #[inline(always)]
    const fn sextet(self, byte: u8) -> i16 {
        let c = byte as i16;
        (-1_i16)
            .wrapping_add(in_range(c, b'A', b'Z', c.wrapping_sub(64)))
            .wrapping_add(in_range(c, b'a', b'z', c.wrapping_sub(70)))
            .wrapping_add(in_range(c, b'0', b'9', c.wrapping_add(5)))
            .wrapping_add(in_range(c, self.c62, self.c62, 63))
            .wrapping_add(in_range(c, self.c63, self.c63, 64))
    }

    /// The character of a sextet. Each term adds the step to the next range of the alphabet if
    /// `value` is past the last sextet of the range before it.
    #[inline(always)]
    const fn symbol(self, value: i16) -> u8 {
        low(value
            .wrapping_add(b'A' as i16)
            .wrapping_add(past(value, 25, 6))
            .wrapping_add(past(value, 51, -75))
            .wrapping_add(past(value, 61, self.step62))
            .wrapping_add(past(value, 62, self.step63)))
    }

    /// The three bytes of four characters, and the OR of their sextets, which is negative if a
    /// character is outside the alphabet.
    #[inline(always)]
    const fn decode_block(self, block: [u8; 4]) -> ([u8; 3], i16) {
        let [a, b, c, d] = [
            self.sextet(block[0]),
            self.sextet(block[1]),
            self.sextet(block[2]),
            self.sextet(block[3]),
        ];
        (
            [low(a << 2 | b >> 4), low(b << 4 | c >> 2), low(c << 6 | d)],
            a | b | c | d,
        )
    }

    /// The four characters of three bytes.
    #[inline(always)]
    const fn encode_block(self, block: [u8; 3]) -> [u8; 4] {
        let [a, b, c] = [block[0] as i16, block[1] as i16, block[2] as i16];
        [
            self.symbol(a >> 2),
            self.symbol((a << 4 | b >> 4) & 63),
            self.symbol((b << 2 | c >> 6) & 63),
            self.symbol(c & 63),
        ]
    }
}

/// The sextet of `byte` in the alphabet of `profile`, or -1 for a byte outside it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
pub(super) const fn sextet(profile: Profile, byte: u8) -> i16 {
    Alphabet::of(profile).sextet(byte)
}

/// The first byte outside the alphabet, or the length. Only membership enters the count.
pub(super) fn first_invalid(profile: Profile, symbols: &[u8]) -> usize {
    let alphabet = Alphabet::of(profile);
    crate::ct::first_negative(symbols.iter().map(|&byte| {
        let valid = byte.is_ascii_uppercase()
            | byte.is_ascii_lowercase()
            | byte.is_ascii_digit()
            | (byte == alphabet.c62)
            | (byte == alphabet.c63);
        -i16::from(!valid)
    }))
}

/// The character of `value`, a sextet, in the alphabet of `profile`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) const fn symbol(profile: Profile, value: i16) -> u8 {
    Alphabet::of(profile).symbol(value)
}

/// Encodes `bytes` into the characters `out`, with no padding. `out` has the length of the text.
pub(super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let alphabet = Alphabet::of(profile);
    let (blocks, rest) = bytes.as_chunks::<3>();
    let (chunks, tail) = out.as_chunks_mut::<4>();
    for (block, chunk) in blocks.iter().zip(chunks) {
        *chunk = alphabet.encode_block(*block);
    }
    if !rest.is_empty() {
        let mut block = [0; 3];
        put(&mut block, rest);
        let mut chars = alphabet.encode_block(block);
        put(tail, &chars);
        class.clear(&mut block);
        class.clear(&mut chars);
    }
}

/// Decodes the characters `text`, which have no padding, into `out`, which has the length of the
/// bytes. Gives whether each character is in the alphabet and the bits after the last byte are
/// zero.
pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let alphabet = Alphabet::of(profile);
    let (blocks, rest) = text.as_chunks::<4>();
    let (chunks, tail) = out.as_chunks_mut::<3>();
    let (mut values, mut trailing) = (0, 0);
    for (block, chunk) in blocks.iter().zip(chunks) {
        let block_values;
        (*chunk, block_values) = alphabet.decode_block(*block);
        values |= block_values;
    }
    if !rest.is_empty() {
        // 'A', which is zero, in place of the characters that are not there.
        let mut block = [b'A'; 4];
        put(&mut block, rest);
        let (mut bytes, block_values) = alphabet.decode_block(block);
        values |= block_values;
        put(tail, &bytes);
        // The byte after the result holds the trailing bits, and the `A` after the text adds none.
        trailing = bytes.get(tail.len()).copied().unwrap_or(0);
        class.clear(&mut block);
        class.clear(&mut bytes);
    }
    (values >= 0) & (trailing == 0)
}

/// Copies the first bytes of `from` into `into`, as many as the shorter of the two has, at most 3:
/// the tail of a block. Each byte takes its own store, and the branches test only the length. A
/// copy of a length that the compiler does not know is a call to `memcpy` on x86_64, whose code
/// the scan of the secret class cannot read.
#[inline(always)]
fn put(into: &mut [u8], from: &[u8]) {
    for at in 0..3 {
        if let (Some(into), Some(&from)) = (into.get_mut(at), from.get(at)) {
            *into = from;
        }
    }
}
