#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Classes {
    pub(super) structural: u64,
    pub(super) whitespace: u64,
    pub(super) quote: u64,
    pub(super) backslash: u64,
    pub(super) control: u64,
    pub(super) utf8_error: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub(super) tokens: u64,
    pub(super) carry: Carry,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Carry(u64);

impl Carry {
    const ESCAPED: u64 = 1 << 0;
    const SCALAR: u64 = 1 << 1;
    const UTF8_ERROR: u64 = 1 << 2;
    const IN_STRING: u64 = 1 << 63;

    #[inline(always)]
    pub(super) const fn after_run(previous: u8) -> Self {
        Self(!matches!(previous, b',' | b'[' | b']') as u64 * Self::SCALAR)
    }

    #[inline(always)]
    pub(super) const fn escaped(self) -> u64 {
        self.0 & Self::ESCAPED
    }

    #[inline(always)]
    const fn scalar(self) -> u64 {
        (self.0 & Self::SCALAR) >> 1
    }

    #[inline(always)]
    const fn in_string(self) -> u64 {
        (self.0.cast_signed() >> 63).cast_unsigned()
    }

    #[inline(always)]
    pub(super) const fn utf8_error(self) -> bool {
        self.0 & Self::UTF8_ERROR != 0
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(super) const fn in_plain_string(self) -> bool {
        self.0 == Self::IN_STRING
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Escapes {
    pub(super) escaped: u64,
    pub(super) starts: u64,
}

impl Classes {
    #[inline(always)]
    pub(super) fn block(self, escapes: Escapes, carry: Carry, prefix_xor: impl Fn(u64) -> u64) -> Block {
        let quote = self.quote & !escapes.escaped;
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

#[inline(always)]
pub(super) fn before(input: &[u8], at: usize) -> u32 {
    input[..at].last_chunk().map_or(0, |bytes| u32::from_le_bytes(*bytes))
}

#[inline(always)]
pub(super) fn tail(rest: &[u8]) -> [u64; 8] {
    let mut words = [u64::from_ne_bytes([b' '; 8]); 8];
    for (word, bytes) in words.iter_mut().zip(rest.chunks(8)) {
        *word = self::word(bytes);
    }
    words
}

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
    let after = u32::try_from(8 * len).map_or(0, |bits| u64::MAX.unbounded_shl(bits));
    low | after & u64::from_ne_bytes([b' '; 8])
}

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
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTINUATIONS;

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

#[inline]
pub(super) fn ends_incomplete(before: u32) -> bool {
    let [_, first, second, third] = before.to_le_bytes();
    third >= 0xc0 || second >= 0xe0 || first >= 0xf0
}
