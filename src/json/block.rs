//! The classes of one block of 64 bytes, the tokens and the carry that each kernel gives, and the
//! UTF-8 check.

/// The classes of the bytes of one block, which each kernel finds on its way to the [`Block`]. Bit
/// `i` of each mask is byte `i` of the block.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Classes {
    /// `{`, `}`, `[`, `]`, `:`, and `,`.
    pub(super) structural: u64,
    /// Space, tab, line feed, and carriage return.
    pub(super) whitespace: u64,
    /// `"`, also an escaped one.
    pub(super) quote: u64,
    /// `\`.
    pub(super) backslash: u64,
    /// Each byte under 0x20.
    pub(super) control: u64,
    /// Whether [`utf8_error_at`] is true at a byte of the block.
    pub(super) utf8_error: bool,
}

/// What a kernel gives for one block: its tokens, which are the positions that the cursor gives,
/// and its carry. The two words fit in two registers, so a kernel that the CPU selects at run time
/// returns them with no store to memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    /// Each quote that opens or closes a string. Outside a string, each structural byte and the
    /// first byte of each scalar, which is a run of bytes that are not structural, whitespace, or a
    /// quote. In a string, each `\` that starts an escape and each byte under 0x20.
    pub(super) tokens: u64,
    pub(super) carry: Carry,
}

/// The state that one block gives to the next, and the UTF-8 check of the block, in one word. A
/// kernel reads only the state, so the cursor passes the word to the next block as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Carry(u64);

impl Carry {
    /// Set if the first byte of the next block follows a `\` that starts an escape.
    const ESCAPED: u64 = 1 << 0;
    /// Set if the last byte of the block is in a scalar.
    const SCALAR: u64 = 1 << 1;
    /// Set if a byte of the block ends an error of UTF-8 ([`utf8_error_at`]).
    const UTF8_ERROR: u64 = 1 << 2;
    /// Set if the next block starts in a string. It is the top bit, so one arithmetic shift gives
    /// the mask of [`Carry::in_string`].
    const IN_STRING: u64 = 1 << 63;

    /// 1 if the first byte of the next block follows a `\` that starts an escape, else 0.
    #[inline(always)]
    pub(super) const fn escaped(self) -> u64 {
        self.0 & Self::ESCAPED
    }

    /// 1 if the last byte of the block is in a scalar, else 0.
    #[inline(always)]
    const fn scalar(self) -> u64 {
        (self.0 & Self::SCALAR) >> 1
    }

    /// All ones if the next block starts in a string, else 0.
    #[inline(always)]
    const fn in_string(self) -> u64 {
        (self.0.cast_signed() >> 63).cast_unsigned()
    }

    /// Whether a byte of the block ends an error of UTF-8.
    #[inline(always)]
    pub(super) const fn utf8_error(self) -> bool {
        self.0 & Self::UTF8_ERROR != 0
    }

    /// Whether the next block starts in a string, after no `\` that starts an escape.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(super) const fn in_plain_string(self) -> bool {
        self.0 == Self::IN_STRING
    }
}

/// The escapes of one block.
#[derive(Clone, Copy, Debug)]
pub(super) struct Escapes {
    /// Each byte after a `\` that starts an escape.
    pub(super) escaped: u64,
    /// Each `\` that starts an escape: a `\` that is not escaped.
    pub(super) starts: u64,
}

impl Classes {
    /// The block of these classes and `escapes`, after `carry`. `escapes` starts from
    /// [`Carry::escaped`]. `prefix_xor` gives the XOR of bits `0..=i` of its argument in each bit
    /// `i`.
    #[inline(always)]
    pub(super) fn block(self, escapes: Escapes, carry: Carry, prefix_xor: impl Fn(u64) -> u64) -> Block {
        let quote = self.quote & !escapes.escaped;
        // Bit `i` is set if an odd number of quotes are at `0..=i`: an opening quote is in its
        // string, and a closing quote is not.
        let in_string = prefix_xor(quote) ^ carry.in_string();
        let inside = in_string & !quote;
        let outside = !(in_string | quote);
        let scalar = outside & !(self.structural | self.whitespace);
        let scalar_starts = scalar & !(scalar << 1 | carry.scalar());
        Block {
            tokens: quote | outside & self.structural | scalar_starts | inside & (escapes.starts | self.control),
            carry: Carry(
                in_string & Carry::IN_STRING
                    | ((scalar >> 63) * Carry::SCALAR)
                    | ((escapes.starts >> 63) * Carry::ESCAPED)
                    | (u64::from(self.utf8_error) * Carry::UTF8_ERROR),
            ),
        }
    }
}

/// The 4 bytes of `input` before `at` in little-endian order, or zero before the first block.
#[inline(always)]
pub(super) fn before(input: &[u8], at: usize) -> u32 {
    input[..at].last_chunk().map_or(0, |bytes| u32::from_le_bytes(*bytes))
}

/// The 8 little-endian words of the last block, whose bytes are `rest`, fewer than 64, with spaces
/// after them. Each word comes from loads of a fixed size, so the compiler keeps the block in
/// registers and calls no `memcpy`.
#[inline(always)]
pub(super) fn tail(rest: &[u8]) -> [u64; 8] {
    let mut words = [u64::from_ne_bytes([b' '; 8]); 8];
    for (word, bytes) in words.iter_mut().zip(rest.chunks(8)) {
        *word = self::word(bytes);
    }
    words
}

/// The bytes of `bytes`, at most 8, from the low byte of a little-endian word, and spaces after them.
/// Two overlapping loads of 4 bytes, or three loads of 1 byte, read each byte.
#[inline(always)]
pub(super) fn word(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let low = if let Some(all) = bytes.first_chunk::<8>() {
        u64::from_le_bytes(*all)
    } else if let (Some(first), Some(last)) = (bytes.first_chunk::<4>(), bytes.last_chunk::<4>()) {
        u64::from(u32::from_le_bytes(*first)) | u64::from(u32::from_le_bytes(*last)) << (8 * (len - 4))
    } else if let [first, ..] = bytes {
        let byte = |at: usize| u64::from(bytes[at]) << (8 * at);
        u64::from(*first) | byte(len / 2) | byte(len - 1)
    } else {
        0
    };
    // The bytes from `len`. `unbounded_shl` gives 0 for a shift of 64 bits, so 8 bytes get no space.
    let after = u32::try_from(8 * len).map_or(0, |bits| u64::MAX.unbounded_shl(bits));
    low | after & u64::from_ne_bytes([b' '; 8])
}

/// The XOR of bits `0..=i` of `bits` in each bit `i`, by six shifts.
#[inline(always)]
pub(super) fn prefix_xor(mut bits: u64) -> u64 {
    for shift in [1, 2, 4, 8, 16, 32] {
        bits ^= bits << shift;
    }
    bits
}

// The UTF-8 check of Keiser and Lemire, "Validating UTF-8 in less than one instruction per byte"
// (2021): three tables of the error classes of a pair of bytes, by the high nibble and the low
// nibble of the first byte and the high nibble of the second byte.

/// A lead byte or ASCII, then a lead byte or ASCII.
const TOO_SHORT: u8 = 1 << 0;
/// ASCII, then a continuation byte.
const TOO_LONG: u8 = 1 << 1;
/// 0xe0, then 0x80 to 0x9f.
const OVERLONG_3: u8 = 1 << 2;
/// 0xf4 then 0x90 to 0xbf, or 0xf5 to 0xff then 0x90 to 0xbf.
const TOO_LARGE: u8 = 1 << 3;
/// 0xed, then 0xa0 to 0xbf.
const SURROGATE: u8 = 1 << 4;
/// 0xc0 or 0xc1, then a continuation byte.
const OVERLONG_2: u8 = 1 << 5;
/// 0xf5 to 0xff, then 0x80 to 0x8f.
const TOO_LARGE_1000: u8 = 1 << 6;
/// 0xf0, then 0x80 to 0x8f.
const OVERLONG_4: u8 = 1 << 6;
/// A continuation byte, then a continuation byte. Not an error in the third or fourth byte of a
/// sequence.
const TWO_CONTINUATIONS: u8 = 1 << 7;
/// The classes that do not depend on the low nibble of the first byte.
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTINUATIONS;

/// The classes of the high nibble of the first byte.
pub(super) const FIRST_HIGH: [u8; 16] = [
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TWO_CONTINUATIONS,
    TWO_CONTINUATIONS,
    TWO_CONTINUATIONS,
    TWO_CONTINUATIONS,
    TOO_SHORT | OVERLONG_2,
    TOO_SHORT,
    TOO_SHORT | OVERLONG_3 | SURROGATE,
    TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
];

/// The classes of the low nibble of the first byte.
pub(super) const FIRST_LOW: [u8; 16] = [
    CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
    CARRY | OVERLONG_2,
    CARRY,
    CARRY,
    CARRY | TOO_LARGE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
];

/// The classes of the high nibble of the second byte.
pub(super) const SECOND_HIGH: [u8; 16] = [
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_LONG | OVERLONG_2 | TWO_CONTINUATIONS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    TOO_LONG | OVERLONG_2 | TWO_CONTINUATIONS | OVERLONG_3 | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTINUATIONS | SURROGATE | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTINUATIONS | SURROGATE | TOO_LARGE,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
];

/// Whether `byte` ends an error of UTF-8, after the bytes `first`, `second`, and `third` before it
/// in that order. An input, with ASCII before it, is UTF-8 exactly when no byte of it ends an error
/// and [`ends_incomplete`] is false after its last byte.
#[inline]
pub(super) fn utf8_error_at(first: u8, second: u8, third: u8, byte: u8) -> bool {
    let nibble = |byte: u8, table: &[u8; 16]| table[usize::from(byte & 0x0f)];
    let pair = nibble(third >> 4, &FIRST_HIGH) & nibble(third, &FIRST_LOW) & nibble(byte >> 4, &SECOND_HIGH);
    // A third or fourth byte of a sequence must be a continuation byte, which the pair classes as
    // `TWO_CONTINUATIONS`.
    let continues = if second >= 0xe0 || first >= 0xf0 {
        TWO_CONTINUATIONS
    } else {
        0
    };
    pair ^ continues != 0
}

/// Whether the 4 bytes `before`, in little-endian order, end in a sequence that needs more bytes.
#[inline]
pub(super) fn ends_incomplete(before: u32) -> bool {
    let [_, first, second, third] = before.to_le_bytes();
    third >= 0xc0 || second >= 0xe0 || first >= 0xf0
}
