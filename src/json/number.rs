//! Numbers: the borrowed number token, its exact accessors, the grammar of RFC 8259 section 6, and
//! the number rule of I-JSON.

use core::ops::Range;

use super::{
    Error, ErrorKind, Limit, float,
    reader::{index, text, unexpected},
};

#[cfg(test)]
mod tests;

/// A number of the input, as written. The reader checked its grammar. It borrows the input. `==`
/// compares the text, so `1.0` and `1.00` differ.
///
/// Each accessor reads the value exactly or refuses: [`Number::to_i64`] and [`Number::to_u64`]
/// read an integer, [`Number::to_decimal`] reads each number of up to 38 significant digits, and
/// [`Number::to_f64`] reads each number to the nearest `f64`. A value that does not fit gives
/// [`ErrorKind::Range`] at the offset of the number.
///
/// ```
/// use rscodec::json::{self, Decimal, ErrorKind, Limits, Profile, Slot};
///
/// const ORDER: Limits = Limits {
///     input_bytes_max: 8192,
///     depth_max: 8,
///     string_bytes_max: 4096,
///     members_max: 32,
///     elements_max: 32,
///     number_digits_max: 32,
/// };
///
/// let order = br#"{"expires":1767225600,"amount":12.50,"lat":52.5163}"#;
/// let mut scratch = [Slot::EMPTY; 8];
/// let (expires, amount, lat) = json::read(order, &ORDER, Profile::Rfc8259, &mut scratch, #[inline(always)] |reader| {
///     let (mut expires, mut amount, mut lat) = (None, None, None);
///     let mut members = reader.value().members()?;
///     while let Some((key, value)) = members.next()? {
///         if key == "expires" {
///             expires = Some(value.number()?.to_u64()?);
///         } else if key == "amount" {
///             amount = Some(value.number()?.to_decimal()?);
///         } else if key == "lat" {
///             lat = Some(value.number()?.to_f64()?);
///         }
///     }
///     Ok((expires, amount, lat))
/// })?;
/// assert_eq!(expires, Some(1_767_225_600));
/// assert_eq!(amount, Some(Decimal { negative: false, coefficient: 1250, exponent: -2 }));
/// assert_eq!(lat, Some(52.5163));
///
/// // An integer accessor refuses a fraction, at the offset of the number.
/// let late = br#"{"expires":1.5}"#;
/// let error = json::read(late, &ORDER, Profile::Rfc8259, &mut scratch, |reader| {
///     let mut members = reader.value().members()?;
///     members.next()?.map(|(_, value)| value.number()?.to_u64()).transpose()
/// })
/// .unwrap_err();
/// assert_eq!((error.kind(), error.offset()), (ErrorKind::Type, 11));
/// # Ok::<(), json::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Number<'a> {
    raw: &'a str,
    /// Three parts in one word, so that each path of the scan writes the word with one store and
    /// the reader reads it with one load. A load of a word that two or more stores wrote waits
    /// until the stores leave the core. The parts:
    ///
    /// - Bits 0 to 31: the offset of the number in the input, for the errors of the accessors. The
    ///   reader refuses an input of more than `u32::MAX` bytes.
    /// - Bits 32 to 47: the digits before the fraction, from the scan of the reader. The digits of
    ///   a number are at most [`Limits::number_digits_max`](super::Limits::number_digits_max), a
    ///   `u16`.
    /// - Bits 48 to 63: the digits of the fraction, or 0 if the number has no fraction.
    place: u64,
    /// The value of the digits, from the scan of the reader, if the number has no exponent and at
    /// most 19 digits, so that a `u64` holds each value, else [`EXACT`]. Thus the accessors read no
    /// digit again, and [`Number::to_f64`] selects the common number with one test.
    digits: u64,
}

/// [`Number::digits`] of a number with an exponent or with more than 19 digits. No value of 19
/// digits is `u64::MAX`.
const EXACT: u64 = u64::MAX;

impl PartialEq for Number<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for Number<'_> {}

/// The form of a number, as written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumberClass {
    /// No fraction and no exponent, for example `-12`. [`Number::to_i64`] and
    /// [`Number::to_u64`] read only this class.
    Integer,
    /// A fraction and no exponent, for example `1.50`.
    Decimal,
    /// An exponent, for example `1e2` or `-1.5E-3`.
    Float,
}

/// A decimal value: `coefficient × 10^exponent`, negative if `negative` is set. It keeps the scale
/// of the text, so `1.50` is `150 × 10^−2`, and `==` compares the parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Decimal {
    /// Whether the text starts with `-`. Thus `-0.0` gives a negative zero.
    pub negative: bool,
    /// The digits of the integer and the fraction, under 10^38.
    pub coefficient: u128,
    /// The exponent of the text, less the digits of the fraction.
    pub exponent: i32,
}

impl<'a> Number<'a> {
    #[inline(always)]
    pub(super) const fn new(raw: &'a str, offset: u32, (integer, fraction, digits): (u16, u16, u64)) -> Self {
        Self {
            raw,
            place: offset as u64 | (integer as u64) << 32 | (fraction as u64) << 48,
            digits,
        }
    }

    /// The number as written, for example `-1.5e3`. It borrows the input.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'a str {
        self.raw
    }

    /// The form of the number, as written.
    #[inline]
    #[must_use]
    pub fn class(&self) -> NumberClass {
        if self.parts().exponent.is_some() {
            NumberClass::Float
        } else if self.fraction_digits() == 0 {
            NumberClass::Integer
        } else {
            NumberClass::Decimal
        }
    }

    /// The value of an integer. `-0` gives 0.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the number has a fraction or an exponent, and [`ErrorKind::Range`] if
    /// the value is outside `i64`, each at the offset of the number. The recovery is
    /// [`Number::to_decimal`] or [`Number::to_f64`].
    #[inline]
    pub fn to_i64(&self) -> Result<i64, Error> {
        let (negative, magnitude) = self.integer()?;
        let value = if negative {
            0_i64.checked_sub_unsigned(magnitude)
        } else {
            i64::try_from(magnitude).ok()
        };
        value.ok_or_else(|| self.range())
    }

    /// The value of an integer. `-0` gives 0.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the number has a fraction or an exponent, and [`ErrorKind::Range`] if
    /// the value is outside `u64`, each at the offset of the number. The recovery is
    /// [`Number::to_decimal`] or [`Number::to_f64`].
    #[inline]
    pub fn to_u64(&self) -> Result<u64, Error> {
        match self.integer()? {
            (true, 1..) => Err(self.range()),
            (_, magnitude) => Ok(magnitude),
        }
    }

    /// The exact value, with the scale of the text: `1.50` gives `150 × 10^−2`, and `1.5e3` gives
    /// `15 × 10^2`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Range`] at the offset of the number if it has more than 38 significant digits,
    /// from its first nonzero digit to its last digit, or if the exponent is outside `i32`. The
    /// recovery is [`Number::to_f64`], or the text of [`Number::as_str`].
    #[inline]
    pub fn to_decimal(&self) -> Result<Decimal, Error> {
        /// The most significant digits of a [`Decimal`]: each such coefficient is under 10^38.
        const DIGITS_MAX: usize = 38;
        let bytes = self.raw.as_bytes();
        let parts = self.parts();
        let digits = bytes[parts.integer].iter().chain(&bytes[parts.fraction]);
        if digits.clone().skip_while(|&&digit| digit == b'0').count() > DIGITS_MAX {
            return Err(self.range());
        }
        let coefficient = digits.fold(0, |value, &digit| value * 10 + u128::from(digit - b'0'));
        let exponent = parts.exponent.map_or(0, |text| exponent(&bytes[text]));
        let exponent = exponent
            .checked_sub(i64::from(self.fraction_digits()))
            .and_then(|exponent| i32::try_from(exponent).ok())
            .ok_or_else(|| self.range())?;
        Ok(Decimal {
            negative: parts.negative,
            coefficient,
            exponent,
        })
    }

    /// The `f64` nearest to the value, with a tie to the even significand. `-0` and each negative
    /// value that rounds to zero give `-0.0`. A nonzero value that rounds to zero gives zero.
    ///
    /// Up to 19 significant digits, it reads with unrounded scaling (Cox, 2026), with no fallback.
    /// A number of more digits reads with the parser of `core`, whose work
    /// [`Limits::number_digits_max`](super::Limits::number_digits_max) bounds.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Range`] at the offset of the number if its magnitude rounds to infinity. The
    /// recovery is [`Number::to_decimal`], or the text of [`Number::as_str`].
    // The common number reads here, and each other number in `f64_exact`. On aarch64 this function
    // inlines into the read loop of its caller (C211 round 4: canada 0.91). On x86_64, with 16
    // registers, the inlined code moved state of the reader to the stack (native b1 1.12), so it
    // stays a call.
    #[cfg_attr(target_arch = "aarch64", inline(always))]
    #[cfg_attr(target_arch = "x86_64", inline(never))]
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), inline)]
    pub fn to_f64(&self) -> Result<f64, Error> {
        // The common number: no exponent, and at most 19 digits, whose value the scan gave.
        if self.digits != EXACT {
            let value = float::read(self.digits, -i32::from(self.fraction_digits())).ok_or_else(|| self.range())?;
            return Ok(if self.raw.as_bytes().first() == Some(&b'-') {
                -value
            } else {
                value
            });
        }
        self.f64_exact()
    }

    /// [`Number::to_f64`] of a number with an exponent or of more than 19 digits. It does not
    /// inline, so the read loop keeps only the common number.
    #[inline(never)]
    fn f64_exact(&self) -> Result<f64, Error> {
        let bytes = self.raw.as_bytes();
        let parts = self.parts();
        let (integer, fraction) = (&bytes[parts.integer.clone()], &bytes[parts.fraction.clone()]);
        let count = integer.len() + fraction.len();
        // Only `0.` before a fraction gives leading zeros, because an integer part does not start
        // with 0.
        let leading = || match integer {
            b"0" => 1 + fraction.iter().take_while(|&&digit| digit == b'0').count(),
            _ => 0,
        };
        let value = if count <= DIGITS_MAX || count - leading() <= DIGITS_MAX {
            // Past 19 digits, the integer part is `0`, so the factor of the fraction changes
            // nothing. The wrapping sum is exact, because the value is under 10^19.
            let scale = POW10.get(fraction.len()).copied().unwrap_or(0);
            let digits = run(bytes, parts.integer)
                .wrapping_mul(scale)
                .wrapping_add(run(bytes, parts.fraction));
            let exponent = parts.exponent.map_or(0, |text| exponent(&bytes[text]));
            let power = exponent.saturating_sub(i64::from(self.fraction_digits()));
            // A power outside `i32` is far outside the table of `read`.
            let power = i32::try_from(power).unwrap_or(if power < 0 { i32::MIN } else { i32::MAX });
            float::read(digits, power).map(|value| if parts.negative { -value } else { value })
        } else {
            // `core` reads each number of RFC 8259, so the parse does not fail.
            self.raw.parse::<f64>().ok().filter(|value| value.is_finite())
        };
        value.ok_or_else(|| self.range())
    }

    /// The sign and the magnitude of an integer.
    #[inline(always)]
    fn integer(&self) -> Result<(bool, u64), Error> {
        let bytes = self.raw.as_bytes();
        let parts = self.parts();
        if parts.integer.end < bytes.len() {
            return Err(Error::new(ErrorKind::Type, self.offset()));
        }
        // No integer has a leading zero, so the order of equal lengths is the order of the values.
        let digits = &bytes[parts.integer.clone()];
        match digits.len() {
            ..20 => Ok((parts.negative, self.digits)),
            20 if digits <= b"18446744073709551615".as_slice() => Ok((parts.negative, run(bytes, parts.integer))),
            _ => Err(self.range()),
        }
    }

    /// The parts of the number, from the digit counts of the scan.
    #[inline(always)]
    fn parts(&self) -> Parts {
        let len = self.raw.len();
        let negative = self.raw.as_bytes().first() == Some(&b'-');
        let start = usize::from(negative);
        let integer = start..start + usize::from(self.integer_digits());
        let fraction = match self.fraction_digits() {
            0 => integer.end..integer.end,
            digits => integer.end + 1..integer.end + 1 + usize::from(digits),
        };
        Parts {
            negative,
            exponent: (fraction.end < len).then(|| fraction.end + 1..len),
            integer,
            fraction,
        }
    }

    #[inline(always)]
    const fn range(&self) -> Error {
        Error::new(ErrorKind::Range, self.offset())
    }

    /// The offset of the number in the input.
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the offset is the low 32 bits of `place`"
    )]
    const fn offset(&self) -> usize {
        self.place as u32 as usize
    }

    /// The count of the digits before the fraction.
    #[inline(always)]
    #[expect(clippy::cast_possible_truncation, reason = "the count is bits 32 to 47 of `place`")]
    const fn integer_digits(&self) -> u16 {
        (self.place >> 32) as u16
    }

    /// The count of the digits of the fraction, or 0 if the number has no fraction.
    #[inline(always)]
    const fn fraction_digits(&self) -> u16 {
        (self.place >> 48) as u16
    }
}

/// The most significant digits that a `u64` holds for each value.
const DIGITS_MAX: usize = 19;

/// `10^n` at index `n`, for each `n` of at most [`DIGITS_MAX`].
const POW10: [u64; DIGITS_MAX + 1] = {
    let mut table = [1; DIGITS_MAX + 1];
    let mut index = 1;
    while index < table.len() {
        table[index] = table[index - 1] * 10;
        index += 1;
    }
    table
};

/// `b'0'` in each byte.
const ASCII_ZEROS: u64 = u64::from_ne_bytes([b'0'; 8]);

/// The value of the ASCII digits of `bytes` at `digits`, in wrapping arithmetic, so it is exact
/// when the value is under 2^64. It reads 8 digits at a time, and the digits after the last group
/// of 8 from one word of `bytes` that holds them, with no loop for each digit.
#[inline(always)]
fn run(bytes: &[u8], digits: Range<usize>) -> u64 {
    let (words, rest) = bytes[digits.clone()].as_chunks::<8>();
    let value = words.iter().fold(0, |value: u64, word| {
        value
            .wrapping_mul(100_000_000)
            .wrapping_add(eight_digits(u64::from_le_bytes(*word)))
    });
    let part = rest.len();
    if part == 0 {
        return value;
    }
    let Some(last) = bytes.len().checked_sub(8) else {
        // A number of fewer than 8 bytes has at most 7 digits, so no step overflows.
        return rest.iter().fold(value, |value, &digit| {
            value.wrapping_mul(10).wrapping_add(u64::from(digit.wrapping_sub(b'0')))
        });
    };
    // The word of 8 bytes that holds the last `part` digits: from their first digit, or the last
    // word of the number.
    let from = digits.end - part;
    let window = from.min(last);
    let word = u64::from_le_bytes(*bytes[window..].first_chunk().expect("a window is in the number"));
    // The digits move to the high bytes, and zeros fill the low bytes: they add nothing.
    let shift = 8 * (8 - part);
    let word = (word >> (8 * (from - window))) << shift | ASCII_ZEROS >> (64 - shift);
    value.wrapping_mul(POW10[part]).wrapping_add(eight_digits(word))
}

/// The value of the 8 ASCII digits of `word`, the first digit in the low byte.
///
/// Each step sets each lane to its value times 10, 100, or 10^4, plus the value of the next lane,
/// and a mask keeps every other lane: the bytes give pairs, the pairs give groups of 4, and the
/// groups give 8 digits. No lane carries into the next: 9 × 10 + 9, 99 × 100 + 99, and
/// 9999 × 10^4 + 9999 each fit their lane. Thus no step overflows the word, and the wrapping
/// operations give the same value with no overflow check.
#[inline(always)]
const fn eight_digits(word: u64) -> u64 {
    digit_values(word.wrapping_sub(ASCII_ZEROS))
}

/// [`eight_digits`] of the word whose bytes are the values of its digits, from 0 to 9.
#[inline(always)]
const fn digit_values(digits: u64) -> u64 {
    let pairs = digits.wrapping_mul(10).wrapping_add(digits >> 8) & 0x00ff_00ff_00ff_00ff;
    let groups = pairs.wrapping_mul(100).wrapping_add(pairs >> 16) & 0x0000_ffff_0000_ffff;
    groups.wrapping_mul(10_000).wrapping_add(groups >> 32) & 0xffff_ffff
}

/// The parts of a number, each a range of its text: the rule of I-JSON and the accessors read them.
#[derive(Clone, Debug)]
struct Parts {
    /// Whether the number starts with `-`.
    negative: bool,
    /// The digits before the fraction.
    integer: Range<usize>,
    /// The digits of the fraction, or an empty range.
    fraction: Range<usize>,
    /// The exponent after its `e` or `E`, with its sign, if the number has one (see [`exponent`]).
    exponent: Option<Range<usize>>,
}

/// The value of an exponent after its `e`: its sign and its digits. It saturates, which changes no
/// result: a fraction has at most `u16::MAX` digits, so an exponent past `i64` is still far outside
/// binary64 and `i32` after the digits of the fraction.
#[inline(always)]
fn exponent(text: &[u8]) -> i64 {
    let (negative, digits) = match text.split_first() {
        Some((b'-', digits)) => (true, digits),
        Some((b'+', digits)) => (false, digits),
        _ => (false, text),
    };
    let value = digits.iter().fold(0_i64, |value, &digit| {
        value.saturating_mul(10).saturating_add(i64::from(digit - b'0'))
    });
    if negative { -value } else { value }
}

/// The number of ASCII digits at the start of `bytes`, from 8 bytes at a time.
///
/// A byte is a digit exactly when its high nibble is 3 and its high nibble after an add of 6 is
/// still 3. An add carries into the next byte only from a byte of 0xfa or more, so each digit
/// before the first other byte gives a 0, and the lowest set bit is in that byte.
#[inline(always)]
fn leading_digits(bytes: &[u8]) -> usize {
    let mut len = 0;
    while let Some(word) = bytes.get(len..).and_then(<[u8]>::first_chunk) {
        let other = non_digits(u64::from_le_bytes(*word));
        if other != 0 {
            return len + (other.trailing_zeros() / 8) as usize;
        }
        len += 8;
    }
    len + bytes[len..].iter().take_while(|byte| byte.is_ascii_digit()).count()
}

/// A nonzero byte for each byte of `word` that is not an ASCII digit, and a zero byte for each
/// digit before the first other byte (see [`leading_digits`]).
#[inline(always)]
const fn non_digits(word: u64) -> u64 {
    const HIGH_NIBBLES: u64 = u64::from_ne_bytes([0xf0; 8]);
    const THREES: u64 = u64::from_ne_bytes([0x30; 8]);
    const SIXES: u64 = u64::from_ne_bytes([0x06; 8]);
    (word & HIGH_NIBBLES ^ THREES) | (word.wrapping_add(SIXES) & HIGH_NIBBLES ^ THREES)
}

/// The count of the digits at `from`, at least one, and at most `budget`, which it decreases.
#[inline(always)]
fn digits(input: &[u8], from: usize, budget: &mut u16) -> Result<u16, Error> {
    let len = leading_digits(&input[from..]);
    if len == 0 {
        return Err(unexpected(input, from));
    }
    let Some(count) = u16::try_from(len).ok().filter(|&count| count <= *budget) else {
        return Err(Error::new(
            ErrorKind::Limit(Limit::NumberDigits),
            from + usize::from(*budget),
        ));
    };
    *budget -= count;
    Ok(count)
}

/// The bytes from the start of a number that [`common`] reads: a sign, 19 digits, a point, and
/// three words of 8 bytes.
const WINDOW: usize = 48;

/// The number at `at`. Each byte of a number is ASCII. A byte that is not structural, whitespace,
/// or a quote must not follow it.
///
/// [`common`] reads the common number with one bound check, if [`WINDOW`] bytes of the input
/// start at `at`. [`scan_exact`] reads each other number out of line, and it gives each refusal at
/// the first byte that the grammar refuses. Each path gives the whole [`Number`], so that the
/// caller reads each of its words as the path wrote it.
///
/// # Errors
///
/// The first byte that the grammar does not allow, or the first digit past `budget`.
#[inline(always)]
pub(super) fn scan(input: &[u8], at: usize, budget: u16) -> Result<Number<'_>, Error> {
    if let Some(window) = input.get(at..).and_then(<[u8]>::first_chunk)
        && let Some((len, parts)) = common(window, budget)
    {
        return Ok(Number::new(text(input, at, at + len), index(at), parts));
    }
    scan_exact(input, at, budget)
}

/// The length and the parts of the common number at the start of `window`, or `None` for each
/// other text. The common number has digits, perhaps a fraction, no exponent, and at most 19 and at
/// most `budget` digits.
#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "each count is at most 19")]
fn common(window: &[u8; WINDOW], budget: u16) -> Option<(usize, (u16, u16, u64))> {
    let start = usize::from(window[0] == b'-');
    let (integer, mut digits, mut next) = digits_at(window, start)?;
    // A number that starts with 0 has no other digit before its fraction.
    if integer == 0 || integer > DIGITS_MAX || integer > 1 && window[start] == b'0' {
        return None;
    }
    let mut len = start + integer;
    let mut fraction = 0;
    if next == b'.' {
        let (count, value, after) = digits_at(window, len + 1)?;
        // A fraction has a digit.
        if count == 0 || integer + count > DIGITS_MAX {
            return None;
        }
        fraction = count;
        digits = digits.wrapping_mul(POW10[count]).wrapping_add(value);
        next = after;
        len += 1 + count;
    }
    let ends = matches!(
        next,
        b' ' | b'\t' | b'\n' | b'\r' | b',' | b':' | b'[' | b']' | b'{' | b'}' | b'"'
    );
    (ends && integer + fraction <= usize::from(budget)).then_some((len, (integer as u16, fraction as u16, digits)))
}

/// The count of the ASCII digits of `window` at `from`, at most 23, their value in wrapping
/// arithmetic, and the byte after them. It reads at most three words of 8 bytes, and it gives `None`
/// if each byte of them is a digit or if they pass the end of `window`.
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a shift of a word by a multiple of 8 keeps one byte"
)]
fn digits_at(window: &[u8; WINDOW], from: usize) -> Option<(usize, u64, u8)> {
    let mut value = 0_u64;
    for index in 0..3 {
        let bytes = window.get(from + 8 * index..).and_then(<[u8]>::first_chunk)?;
        let word = u64::from_le_bytes(*bytes);
        let other = non_digits(word);
        if other != 0 {
            // 8 times the count of the digits before the first other byte.
            let bits = other.trailing_zeros() & 56;
            // The digits move to the high bytes, and zeros fill the low bytes. A borrow of the
            // subtraction moves only into the bytes after the digits, which the shifts drop. Two
            // shifts drop each byte of a word with no digit.
            let digits = word.wrapping_sub(ASCII_ZEROS) << 8 << (56 - bits);
            let count = (bits / 8) as usize;
            let value = value.wrapping_mul(POW10[count]).wrapping_add(digit_values(digits));
            return Some((8 * index + count, value, (word >> bits) as u8));
        }
        value = value.wrapping_mul(100_000_000).wrapping_add(eight_digits(word));
    }
    None
}

/// [`scan`] of each number that [`common`] does not read: a number with an exponent, of more than
/// 19 digits, past the budget, in the last [`WINDOW`] bytes of the input, or a refusal. It does not
/// inline, so the read loop keeps only the common number.
#[inline(never)]
fn scan_exact(input: &[u8], at: usize, mut budget: u16) -> Result<Number<'_>, Error> {
    let start = at + usize::from(input[at] == b'-');
    // A number that starts with 0 has no other digit before its fraction.
    let integer = if input.get(start) == Some(&b'0') {
        if budget == 0 {
            return Err(Error::new(ErrorKind::Limit(Limit::NumberDigits), start));
        }
        budget -= 1;
        1
    } else {
        digits(input, start, &mut budget)?
    };
    let mut end = start + usize::from(integer);
    let fraction = if input.get(end) == Some(&b'.') {
        let fraction = digits(input, end + 1, &mut budget)?;
        end += 1 + usize::from(fraction);
        fraction
    } else {
        0
    };
    if matches!(input.get(end), Some(b'e' | b'E')) {
        let sign = end + 1;
        let first = sign + usize::from(matches!(input.get(sign), Some(b'+' | b'-')));
        end = first + usize::from(digits(input, first, &mut budget)?);
    }
    match input.get(end) {
        None | Some(b' ' | b'\t' | b'\n' | b'\r' | b',' | b':' | b'[' | b']' | b'{' | b'}' | b'"') => {}
        Some(_) => return Err(unexpected(input, end)),
    }
    let integer_end = start + usize::from(integer);
    let digits_end = if fraction == 0 {
        integer_end
    } else {
        integer_end + 1 + usize::from(fraction)
    };
    let digits = if digits_end < end || usize::from(integer) + usize::from(fraction) > DIGITS_MAX {
        EXACT
    } else {
        let digits = digit_run(&input[start..integer_end]).1;
        if fraction == 0 {
            digits
        } else {
            digits
                .wrapping_mul(POW10[usize::from(fraction)])
                .wrapping_add(digit_run(&input[integer_end + 1..digits_end]).1)
        }
    };
    Ok(Number::new(
        text(input, at, end),
        index(at),
        (integer, fraction, digits),
    ))
}

/// The count of the ASCII digits at the start of `bytes`, and their value in wrapping arithmetic,
/// which is exact for at most 19 digits. It reads 8 bytes at a time: each word of digits adds its
/// value, and the word of the first other byte adds the value of its digits before that byte.
#[inline(always)]
fn digit_run(bytes: &[u8]) -> (usize, u64) {
    let mut len = 0;
    let mut value = 0_u64;
    while let Some(word) = bytes.get(len..).and_then(<[u8]>::first_chunk::<8>) {
        let word = u64::from_le_bytes(*word);
        let other = non_digits(word);
        if other != 0 {
            let count = (other.trailing_zeros() / 8) as usize;
            if count != 0 {
                // The digits move to the high bytes, and zeros fill the low bytes.
                let shift = 8 * (8 - count);
                let digits = word << shift | ASCII_ZEROS >> (64 - shift);
                value = value.wrapping_mul(POW10[count]).wrapping_add(eight_digits(digits));
            }
            return (len + count, value);
        }
        value = value.wrapping_mul(100_000_000).wrapping_add(eight_digits(word));
        len += 8;
    }
    for &byte in &bytes[len..] {
        if !byte.is_ascii_digit() {
            break;
        }
        value = value.wrapping_mul(10).wrapping_add(u64::from(byte - b'0'));
        len += 1;
    }
    (len, value)
}

/// Whether binary64 holds the number of `parts`, in the rule of [`Profile::IJson`] (selected on
/// 2026-11-18, from RFC 7493 section 2.2):
///
/// - An integer, which has no fraction and no exponent, is within ±(2^53 − 1).
/// - Another number has at most 17 significant digits, from its first nonzero digit to its last.
///   Its magnitude does not round to infinity. If it is not zero, it does not round to zero.
///
/// [`Profile::IJson`]: super::Profile::IJson
pub(super) fn binary64_holds(number: &Number<'_>) -> bool {
    /// The power of 10 and the 17 digits of 2^1024 − 2^970, rounded up: the least such pair that
    /// rounds to infinity, because a tie rounds to the even significand.
    const INFINITY: (i64, u64) = (292, 17_976_931_348_623_159);
    /// The power of 10 and the 17 digits of 2^−1075, rounded down: the greatest such pair that
    /// rounds to zero, because a tie rounds to the even significand.
    const ZERO: (i64, u64) = (-340, 24_703_282_292_062_327);
    let input = number.raw.as_bytes();
    let parts = number.parts();
    let integer = &input[parts.integer.clone()];
    let fraction = &input[parts.fraction.clone()];
    if fraction.is_empty() && parts.exponent.is_none() {
        // No integer has a leading zero, so the order of equal lengths is the order of the values.
        return integer.len() < 16 || integer.len() == 16 && integer <= b"9007199254740991".as_slice();
    }
    let digits = || integer.iter().chain(fraction);
    let Some(first) = digits().position(|&digit| digit != b'0') else {
        return true;
    };
    let total = integer.len() + fraction.len();
    let after_last = total - digits().rev().position(|&digit| digit != b'0').unwrap_or(0);
    let count = after_last - first;
    if count > 17 {
        return false;
    }
    // The value is `significand × 10^(integer.len() − after_last + exponent)`. The pair of 17
    // digits multiplies the significand by `10^(17 − count)`. Thus the order of the pairs is the
    // order of the values.
    let significand = digits()
        .skip(first)
        .take(count)
        .fold(0, |value, &digit| value * 10 + u64::from(digit - b'0'));
    let scaled = (count..17).fold(significand, |value, _| value * 10);
    let wide = |len: usize| i64::try_from(len).unwrap_or(i64::MAX);
    let power = parts
        .exponent
        .map_or(0, |text| exponent(&input[text]))
        .saturating_add(wide(integer.len()))
        .saturating_sub(wide(after_last))
        .saturating_sub(wide(17 - count));
    (power, scaled) < INFINITY && (power, scaled) > ZERO
}
