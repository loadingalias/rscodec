//! Numbers: the borrowed number token, its exact accessors, the grammar of RFC 8259 section 6, and
//! the number rule of I-JSON.

use core::ops::Range;

use super::{Error, ErrorKind, Limit, float, reader::unexpected};

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
/// assert_eq!((error.kind, error.offset), (ErrorKind::Type, 11));
/// # Ok::<(), json::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Number<'a> {
    raw: &'a str,
    /// The offset of the number in the input, for the errors of the accessors.
    offset: usize,
}

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
    pub(super) const fn new(raw: &'a str, offset: usize) -> Self {
        Self { raw, offset }
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
        let parts = parts(self.raw.as_bytes());
        if parts.exponent.is_some() {
            NumberClass::Float
        } else if parts.fraction.is_empty() {
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
        let parts = parts(bytes);
        let fraction = &bytes[parts.fraction.clone()];
        let digits = bytes[parts.integer].iter().chain(fraction);
        if digits.clone().skip_while(|&&digit| digit == b'0').count() > DIGITS_MAX {
            return Err(self.range());
        }
        let coefficient = digits.fold(0, |value, &digit| value * 10 + u128::from(digit - b'0'));
        let fraction = i64::try_from(fraction.len()).ok();
        let exponent = fraction.and_then(|fraction| parts.exponent.unwrap_or(0).checked_sub(fraction));
        let exponent = exponent
            .and_then(|exponent| i32::try_from(exponent).ok())
            .ok_or_else(|| self.range())?;
        Ok(Decimal {
            negative: bytes.first() == Some(&b'-'),
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
    #[inline]
    pub fn to_f64(&self) -> Result<f64, Error> {
        /// The most significant digits that a `u64` holds for each value.
        const DIGITS_MAX: usize = 19;
        let bytes = self.raw.as_bytes();
        let parts = parts(bytes);
        let (integer, fraction) = (&bytes[parts.integer], &bytes[parts.fraction]);
        let count = integer.len() + fraction.len();
        // Only `0.` before a fraction gives leading zeros, because an integer part does not start
        // with 0.
        let leading = || match integer {
            b"0" => 1 + fraction.iter().take_while(|&&digit| digit == b'0').count(),
            _ => 0,
        };
        let value = if count <= DIGITS_MAX || count - leading() <= DIGITS_MAX {
            // Each leading zero adds 0, so the wrapping sum is exact.
            let digits = accumulate(accumulate(0, integer), fraction);
            let fraction = i64::try_from(fraction.len()).unwrap_or(i64::MAX);
            let power = parts.exponent.unwrap_or(0).saturating_sub(fraction);
            // A power outside `i32` is far outside the table of `read`.
            let power = i32::try_from(power).unwrap_or(if power < 0 { i32::MIN } else { i32::MAX });
            let negative = bytes.first() == Some(&b'-');
            float::read(digits, power).map(|value| if negative { -value } else { value })
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
        let negative = bytes.first() == Some(&b'-');
        let digits = &bytes[usize::from(negative)..];
        let len = leading_digits(digits);
        if len < digits.len() {
            return Err(Error::new(ErrorKind::Type, self.offset));
        }
        // No integer has a leading zero, so the order of equal lengths is the order of the values.
        match len {
            ..20 => Ok((negative, accumulate(0, digits))),
            20 if digits <= b"18446744073709551615".as_slice() => Ok((negative, accumulate(0, digits))),
            _ => Err(self.range()),
        }
    }

    #[inline(always)]
    const fn range(&self) -> Error {
        Error::new(ErrorKind::Range, self.offset)
    }
}

/// `value` with the ASCII digits of `digits` after its digits, in wrapping arithmetic: 8 digits at
/// a time, then one at a time.
#[inline(always)]
fn accumulate(value: u64, digits: &[u8]) -> u64 {
    let (words, rest) = digits.as_chunks::<8>();
    let value = words.iter().fold(value, |value, word| {
        value
            .wrapping_mul(100_000_000)
            .wrapping_add(eight_digits(u64::from_le_bytes(*word)))
    });
    rest.iter().fold(value, |value, &digit| {
        value.wrapping_mul(10).wrapping_add(u64::from(digit - b'0'))
    })
}

/// The value of the 8 ASCII digits of `word`, the first digit in the low byte.
///
/// Each step sets each lane to its value times 10, 100, or 10^4, plus the value of the next lane,
/// and a mask keeps every other lane: the bytes give pairs, the pairs give groups of 4, and the
/// groups give 8 digits. No lane carries into the next: 9 × 10 + 9, 99 × 100 + 99, and
/// 9999 × 10^4 + 9999 each fit their lane.
#[inline(always)]
const fn eight_digits(word: u64) -> u64 {
    let digits = word - u64::from_ne_bytes([b'0'; 8]);
    let pairs = (digits * 10 + (digits >> 8)) & 0x00ff_00ff_00ff_00ff;
    let groups = (pairs * 100 + (pairs >> 16)) & 0x0000_ffff_0000_ffff;
    (groups * 10_000 + (groups >> 32)) & 0xffff_ffff
}

/// The parts of a number: the rule of I-JSON and the accessors read them.
#[derive(Clone, Debug)]
pub(super) struct Parts {
    /// The digits before the fraction.
    integer: Range<usize>,
    /// The digits of the fraction, or an empty range.
    fraction: Range<usize>,
    /// The value of the exponent, if the number has one (see [`exponent`]).
    exponent: Option<i64>,
}

/// The parts of the number `bytes`, which the reader checked, with each range in `bytes`.
#[inline(always)]
fn parts(bytes: &[u8]) -> Parts {
    let start = usize::from(bytes.first() == Some(&b'-'));
    let integer = start..start + leading_digits(&bytes[start..]);
    let mut end = integer.end;
    let fraction = if bytes.get(end) == Some(&b'.') {
        let from = end + 1;
        end = from + leading_digits(&bytes[from..]);
        from..end
    } else {
        end..end
    };
    Parts {
        integer,
        fraction,
        exponent: bytes.get(end + 1..).map(exponent),
    }
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
    const HIGH_NIBBLES: u64 = u64::from_ne_bytes([0xf0; 8]);
    const THREES: u64 = u64::from_ne_bytes([0x30; 8]);
    const SIXES: u64 = u64::from_ne_bytes([0x06; 8]);
    let mut len = 0;
    while let Some(word) = bytes.get(len..).and_then(<[u8]>::first_chunk) {
        let word = u64::from_le_bytes(*word);
        let other = (word & HIGH_NIBBLES ^ THREES) | (word.wrapping_add(SIXES) & HIGH_NIBBLES ^ THREES);
        if other != 0 {
            return len + (other.trailing_zeros() / 8) as usize;
        }
        len += 8;
    }
    len + bytes[len..].iter().take_while(|byte| byte.is_ascii_digit()).count()
}

/// The end of the digits at `from`, at least one, and at most `budget`, which it decreases.
#[inline(always)]
fn digits(input: &[u8], from: usize, budget: &mut usize) -> Result<usize, Error> {
    let len = leading_digits(&input[from..]);
    if len == 0 {
        return Err(unexpected(input, from));
    }
    if len > *budget {
        return Err(Error::new(ErrorKind::Limit(Limit::NumberDigits), from + *budget));
    }
    *budget -= len;
    Ok(from + len)
}

/// The end of the number at `at`, and its parts. Each byte of the number is ASCII. A byte that is
/// not structural, whitespace, or a quote must not follow it.
///
/// # Errors
///
/// The first byte that the grammar does not allow, or the first digit past `digits_max`.
#[inline(always)]
pub(super) fn scan(input: &[u8], at: usize, digits_max: u16) -> Result<(usize, Parts), Error> {
    let budget = &mut usize::from(digits_max);
    let start = at + usize::from(input[at] == b'-');
    // A number that starts with 0 has no other digit before its fraction.
    let end = if input.get(start) == Some(&b'0') {
        if *budget == 0 {
            return Err(Error::new(ErrorKind::Limit(Limit::NumberDigits), start));
        }
        *budget -= 1;
        start + 1
    } else {
        digits(input, start, budget)?
    };
    let integer = start..end;
    let (fraction, end) = if input.get(end) == Some(&b'.') {
        let fraction_end = digits(input, end + 1, budget)?;
        (end + 1..fraction_end, fraction_end)
    } else {
        (end..end, end)
    };
    let (exponent, end) = if matches!(input.get(end), Some(b'e' | b'E')) {
        let sign = end + 1;
        let first = sign + usize::from(matches!(input.get(sign), Some(b'+' | b'-')));
        let exponent_end = digits(input, first, budget)?;
        (Some(exponent(&input[sign..exponent_end])), exponent_end)
    } else {
        (None, end)
    };
    match input.get(end) {
        None | Some(b' ' | b'\t' | b'\n' | b'\r' | b',' | b':' | b'[' | b']' | b'{' | b'}' | b'"') => {}
        Some(_) => return Err(unexpected(input, end)),
    }
    let parts = Parts {
        integer,
        fraction,
        exponent,
    };
    Ok((end, parts))
}

/// Whether binary64 holds the number of `parts`, in the rule of [`Profile::IJson`] (selected on
/// 2026-11-18, from RFC 7493 section 2.2):
///
/// - An integer, which has no fraction and no exponent, is within ±(2^53 − 1).
/// - Another number has at most 17 significant digits, from its first nonzero digit to its last.
///   Its magnitude does not round to infinity. If it is not zero, it does not round to zero.
///
/// [`Profile::IJson`]: super::Profile::IJson
pub(super) fn binary64_holds(input: &[u8], parts: &Parts) -> bool {
    /// The power of 10 and the 17 digits of 2^1024 − 2^970, rounded up: the least such pair that
    /// rounds to infinity, because a tie rounds to the even significand.
    const INFINITY: (i64, u64) = (292, 17_976_931_348_623_159);
    /// The power of 10 and the 17 digits of 2^−1075, rounded down: the greatest such pair that
    /// rounds to zero, because a tie rounds to the even significand.
    const ZERO: (i64, u64) = (-340, 24_703_282_292_062_327);
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
        .unwrap_or(0)
        .saturating_add(wide(integer.len()))
        .saturating_sub(wide(after_last))
        .saturating_sub(wide(17 - count));
    (power, scaled) < INFINITY && (power, scaled) > ZERO
}
