//! Strings: the borrowed string token, its unescape, and the checks of the escapes and of the
//! noncharacters.

use super::{Error, ErrorKind};

/// A string or a key of the input, as written between its quotes. The reader checked it: it is
/// UTF-8, and each escape is valid. It borrows the input.
///
/// `==` compares after unescape, with a `&str` or with another `Str`. Thus the key `"\u0061lg"`
/// equals `"alg"`.
#[derive(Clone, Copy, Debug)]
pub struct Str<'a> {
    raw: &'a str,
    /// The length after unescape.
    len: usize,
}

impl<'a> Str<'a> {
    #[inline(always)]
    pub(super) const fn new(raw: &'a str, len: usize) -> Self {
        Self { raw, len }
    }

    /// The string in the input, between its quotes.
    #[inline(always)]
    pub(super) const fn raw(&self) -> &'a str {
        self.raw
    }

    /// The length of the string after unescape, in bytes: the length of the buffer that
    /// [`Str::decode`] needs for a string with escapes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the string is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The string, if it has no escape. It borrows the input.
    #[inline]
    #[must_use]
    pub const fn borrowed(&self) -> Option<&'a str> {
        if self.raw.len() == self.len() {
            Some(self.raw)
        } else {
            None
        }
    }

    /// The string after unescape. A string with no escape borrows the input. A string with escapes
    /// decodes into the start of `buf`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Output`] if the string has escapes and `buf` is shorter than the decoded
    /// string. The offset of the error is the length of the decoded string.
    #[inline(always)]
    pub fn decode<'b>(&self, buf: &'b mut [u8]) -> Result<&'b str, Error>
    where
        'a: 'b,
    {
        match self.borrowed() {
            Some(text) => Ok(text),
            None => decode_escaped(self.raw, self.len(), buf),
        }
    }
}

/// [`Str::decode`] of the string `raw` with escapes, whose length after unescape is `len`. It does
/// not inline, so the common string, which has no escape, costs one compare in the code of the
/// caller. It takes plain values, not the `Str`: a reference across the call keeps each string token
/// of the caller in memory.
#[inline(never)]
#[expect(
    unsafe_code,
    reason = "the decoded bytes are runs of UTF-8 between ASCII bytes and the UTF-8 of characters"
)]
fn decode_escaped<'b>(raw: &str, len: usize, buf: &'b mut [u8]) -> Result<&'b str, Error> {
    let out = buf.get_mut(..len).ok_or_else(|| Error::new(ErrorKind::Output, len))?;
    let mut rest = raw.as_bytes();
    let mut written = 0;
    loop {
        let text = run(rest, &mut out[written..]);
        written += text;
        // After its run of text, `rest` ends, or it has a `\` that starts an escape.
        let Some(after) = rest.get(text + 1..) else {
            break;
        };
        let (char, used) = unescape(after);
        written += char.encode_utf8(&mut out[written..]).len();
        rest = &after[used..];
    }
    // SAFETY: `out` holds the runs of `raw`, which is UTF-8, between its `\` bytes, which are
    // ASCII, and the UTF-8 of the character of each escape. `len` is the length of their sum,
    // so they fill `out`.
    Ok(unsafe { core::str::from_utf8_unchecked(out) })
}

impl PartialEq for Str<'_> {
    /// Whether the strings are equal after unescape. It needs no buffer.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self.borrowed(), other.borrowed()) {
            (Some(text), Some(other)) => text == other,
            _ => self.len == other.len && bytes(self.raw.as_bytes()).eq(bytes(other.raw.as_bytes())),
        }
    }
}

impl Eq for Str<'_> {}

impl PartialEq<&str> for Str<'_> {
    /// Whether the string equals `other` after unescape. It needs no buffer.
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        match self.borrowed() {
            Some(text) => text == *other,
            None => self.len() == other.len() && bytes(self.raw.as_bytes()).eq(other.bytes()),
        }
    }
}

/// The index of the first `\` of the 8 bytes of `word` in little-endian order, if it has one: a
/// byte of `word ^ 0x5c…` is zero exactly at a `\`, and the lowest bit of
/// `(x - 0x01…) & !x & 0x80…` is in the first zero byte of `x`, because a borrow moves only up from
/// a zero byte.
#[inline(always)]
fn backslash(word: u64) -> Option<usize> {
    const ONES: u64 = u64::from_ne_bytes([0x01; 8]);
    const HIGH: u64 = u64::from_ne_bytes([0x80; 8]);
    let word = word ^ (ONES * u64::from(b'\\'));
    let zero = word.wrapping_sub(ONES) & !word & HIGH;
    (zero != 0).then_some((zero.trailing_zeros() / 8) as usize)
}

/// Copies the text of `raw` before its first `\` to the start of `out`, and gives its length: the
/// length of `raw` if it has no `\`. While both hold 8 more bytes, each word is one load and one
/// store, with no call. A word can copy bytes after the `\` into `out`: the caller writes the
/// escape and the text after it over them.
#[inline(always)]
fn run(raw: &[u8], out: &mut [u8]) -> usize {
    let mut at = 0;
    while let (Some(word), Some(slot)) = (
        raw.get(at..).and_then(<[u8]>::first_chunk::<8>),
        out.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<8>),
    ) {
        *slot = *word;
        if let Some(index) = backslash(u64::from_le_bytes(*word)) {
            return at + index;
        }
        at += 8;
    }
    let rest = &raw[at..];
    let len = rest.iter().position(|&byte| byte == b'\\').unwrap_or(rest.len());
    out[at..at + len].copy_from_slice(&rest[..len]);
    at + len
}

/// The UTF-8 bytes of the string `raw` between its quotes, after unescape. The reader checked the
/// escapes of `raw`. UTF-8 keeps the order of the scalar values, so the order of these bytes is
/// the order of the characters.
pub(super) fn bytes(raw: &[u8]) -> impl Iterator<Item = u8> {
    let mut rest = raw;
    let mut char = [0; 4];
    let mut pending = 0..0;
    core::iter::from_fn(move || {
        if let Some(index) = pending.next() {
            return Some(char[index]);
        }
        let (&byte, after) = rest.split_first()?;
        if byte != b'\\' {
            rest = after;
            return Some(byte);
        }
        let (value, len) = unescape(after);
        rest = &after[len..];
        pending = 1..value.encode_utf8(&mut char).len();
        Some(char[0])
    })
}

/// The character of the escape whose bytes after its `\` start `after`, and the number of these
/// bytes. The reader checked the escape: a surrogate escape has its pair.
#[inline]
fn unescape(after: &[u8]) -> (char, usize) {
    // `u32::MAX` is no scalar value, so an escape that the reader did not check fails the assertion.
    let (value, len) = match after.split_first() {
        Some((b'u', digits)) => match hex4(digits) {
            Ok(high @ 0xd800..0xdc00) => {
                let low = digits.get(6..).map_or(Err(0), hex4);
                let value = match low {
                    Ok(low @ 0xdc00..0xe000) => 0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00),
                    _ => u32::MAX,
                };
                (value, 11)
            }
            Ok(value) => (value, 5),
            Err(_) => (u32::MAX, 5),
        },
        Some((b'b', _)) => (0x08, 1),
        Some((b'f', _)) => (0x0c, 1),
        Some((b'n', _)) => (0x0a, 1),
        Some((b'r', _)) => (0x0d, 1),
        Some((b't', _)) => (0x09, 1),
        Some((&byte, _)) => (u32::from(byte), 1),
        None => (u32::MAX, 0),
    };
    let char = char::from_u32(value).expect("the reader checks each escape before a string exists");
    (char, len)
}

/// The value of the 4 hex digits at the start of `digits`, or the index of the first byte that is
/// not a hex digit, which can be the end of `digits`.
#[inline]
pub(super) fn hex4(digits: &[u8]) -> Result<u32, usize> {
    /// The value of each hex digit, and 0x10 for each other byte.
    const VALUES: [u8; 256] = {
        let mut values = [0x10; 256];
        let mut byte = 0_u8;
        loop {
            values[byte as usize] = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => 0x10,
            };
            if byte == u8::MAX {
                break values;
            }
            byte += 1;
        }
    };
    let Some(&four) = digits.first_chunk::<4>() else {
        let at = digits.iter().position(|&byte| VALUES[usize::from(byte)] > 0xf);
        return Err(at.unwrap_or(digits.len()));
    };
    let values = four.map(|byte| VALUES[usize::from(byte)]);
    if values.iter().fold(0, |any, &value| any | value) > 0xf {
        return Err(values.iter().position(|&value| value > 0xf).unwrap_or(0));
    }
    Ok(values.iter().fold(0, |value, &digit| value << 4 | u32::from(digit)))
}

/// Whether the scalar value `value` is a noncharacter: U+FDD0 to U+FDEF, and the last two code
/// points of each plane. Unicode has 66.
#[inline]
pub(super) const fn noncharacter(value: u32) -> bool {
    matches!(value, 0xfdd0..=0xfdef) || value & 0xfffe == 0xfffe
}

/// The offset in `text` of the first noncharacter, if `text` has one. `text` is UTF-8 that starts
/// at a character.
#[inline]
pub(super) fn first_noncharacter(text: &[u8]) -> Option<usize> {
    if text.is_ascii() {
        return None;
    }
    // In UTF-8, U+FDD0 to U+FDEF are `ef b7 90` to `ef b7 af`, and U+FFFE and U+FFFF are `ef bf be`
    // and `ef bf bf`. The last two code points of planes 1 to 16 have a second byte of the form
    // `10xx1111`, then `bf be` or `bf bf`. Only a 4-byte sequence has three continuation bytes, so
    // no other character ends in these bytes.
    let window = text.array_windows().position(|&[first, second, third]| {
        (first == 0xef && second == 0xb7 && matches!(third, 0x90..=0xaf))
            || (second == 0xbf && matches!(third, 0xbe | 0xbf) && (first == 0xef || first & 0xcf == 0x8f))
    })?;
    // A window that starts at a continuation byte starts at the second byte of its sequence.
    Some(if text[window] == 0xef { window } else { window - 1 })
}

/// The length in UTF-8 of the scalar value `value` of the Basic Multilingual Plane.
#[inline]
pub(super) const fn utf8_len(value: u32) -> usize {
    match value {
        0..0x80 => 1,
        0x80..0x800 => 2,
        _ => 3,
    }
}
