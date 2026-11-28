use core::ops::Range;

use super::{
    Error, ErrorKind, Limit, float,
    reader::{index, text, unexpected},
};

#[cfg(test)]
mod tests;

/// A validated number that borrows its original text. Equality compares the spelling, including trailing zeros.
/// Integer accessors require integer syntax. [`Number::to_decimal`] preserves the decimal scale.
/// [`Number::to_f64`] rounds to the nearest binary64 value.
///
/// ```
/// use rscodec::json::{self, Decimal, Limits, Profile};
///
/// let limits = Limits {
///     input_bytes_max: 32, depth_max: 0, string_bytes_max: 0,
///     members_max: 0, elements_max: 0, number_digits_max: 8,
/// };
/// let amount = json::read(b"12.50", &limits, Profile::Rfc8259, &mut [], |reader| {
///     reader.value().number()?.to_decimal()
/// })?;
/// assert_eq!(amount, Decimal { negative: false, coefficient: 1250, exponent: -2 });
/// # Ok::<(), json::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Number<'a> {
    raw: &'a str,
    place: u64,
    digits: u64,
}

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
    // Inlining improved canada.json by 9% on AArch64. On x86_64 it spilled reader state to the stack.
    #[cfg_attr(target_arch = "aarch64", inline(always))]
    #[cfg_attr(target_arch = "x86_64", inline(never))]
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), inline)]
    pub fn to_f64(&self) -> Result<f64, Error> {
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

    #[inline(never)]
    fn f64_exact(&self) -> Result<f64, Error> {
        let bytes = self.raw.as_bytes();
        let parts = self.parts();
        let (integer, fraction) = (&bytes[parts.integer.clone()], &bytes[parts.fraction.clone()]);
        let count = integer.len() + fraction.len();
        let leading = || match integer {
            b"0" => 1 + fraction.iter().take_while(|&&digit| digit == b'0').count(),
            _ => 0,
        };
        let value = if count <= DIGITS_MAX || count - leading() <= DIGITS_MAX {
            let scale = POW10.get(fraction.len()).copied().unwrap_or(0);
            let digits = run(bytes, parts.integer)
                .wrapping_mul(scale)
                .wrapping_add(run(bytes, parts.fraction));
            let exponent = parts.exponent.map_or(0, |text| exponent(&bytes[text]));
            let power = exponent.saturating_sub(i64::from(self.fraction_digits()));
            let power = i32::try_from(power).unwrap_or(if power < 0 { i32::MIN } else { i32::MAX });
            float::read(digits, power).map(|value| if parts.negative { -value } else { value })
        } else {
            self.raw.parse::<f64>().ok().filter(|value| value.is_finite())
        };
        value.ok_or_else(|| self.range())
    }

    #[inline(always)]
    fn integer(&self) -> Result<(bool, u64), Error> {
        let bytes = self.raw.as_bytes();
        let parts = self.parts();
        if parts.integer.end < bytes.len() {
            return Err(Error::new(ErrorKind::Type, self.offset()));
        }
        let digits = &bytes[parts.integer.clone()];
        match digits.len() {
            ..20 => Ok((parts.negative, self.digits)),
            20 if digits <= b"18446744073709551615".as_slice() => Ok((parts.negative, run(bytes, parts.integer))),
            _ => Err(self.range()),
        }
    }

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

    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the offset is the low 32 bits of `place`"
    )]
    const fn offset(&self) -> usize {
        self.place as u32 as usize
    }

    #[inline(always)]
    #[expect(clippy::cast_possible_truncation, reason = "the count is bits 32 to 47 of `place`")]
    const fn integer_digits(&self) -> u16 {
        (self.place >> 32) as u16
    }

    #[inline(always)]
    const fn fraction_digits(&self) -> u16 {
        (self.place >> 48) as u16
    }
}

const DIGITS_MAX: usize = 19;

const POW10: [u64; DIGITS_MAX + 1] = {
    let mut table = [1; DIGITS_MAX + 1];
    let mut index = 1;
    while index < table.len() {
        table[index] = table[index - 1] * 10;
        index += 1;
    }
    table
};

const ASCII_ZEROS: u64 = u64::from_ne_bytes([b'0'; 8]);

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
        return rest.iter().fold(value, |value, &digit| {
            value.wrapping_mul(10).wrapping_add(u64::from(digit.wrapping_sub(b'0')))
        });
    };
    let from = digits.end - part;
    let window = from.min(last);
    let word = u64::from_le_bytes(*bytes[window..].first_chunk().expect("a window is in the number"));
    let shift = 8 * (8 - part);
    let word = (word >> (8 * (from - window))) << shift | ASCII_ZEROS >> (64 - shift);
    value.wrapping_mul(POW10[part]).wrapping_add(eight_digits(word))
}

#[inline(always)]
const fn eight_digits(word: u64) -> u64 {
    digit_values(word.wrapping_sub(ASCII_ZEROS))
}

#[inline(always)]
const fn digit_values(digits: u64) -> u64 {
    let pairs = digits.wrapping_mul(10).wrapping_add(digits >> 8) & 0x00ff_00ff_00ff_00ff;
    let groups = pairs.wrapping_mul(100).wrapping_add(pairs >> 16) & 0x0000_ffff_0000_ffff;
    groups.wrapping_mul(10_000).wrapping_add(groups >> 32) & 0xffff_ffff
}

#[derive(Clone, Debug)]
struct Parts {
    negative: bool,
    integer: Range<usize>,
    fraction: Range<usize>,
    exponent: Option<Range<usize>>,
}

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

#[inline(always)]
const fn non_digits(word: u64) -> u64 {
    const HIGH_NIBBLES: u64 = u64::from_ne_bytes([0xf0; 8]);
    const THREES: u64 = u64::from_ne_bytes([0x30; 8]);
    const SIXES: u64 = u64::from_ne_bytes([0x06; 8]);
    (word & HIGH_NIBBLES ^ THREES) | (word.wrapping_add(SIXES) & HIGH_NIBBLES ^ THREES)
}

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

const WINDOW: usize = 48;

#[inline(always)]
pub(super) fn scan(input: &[u8], at: usize, budget: u16) -> Result<Number<'_>, Error> {
    if let Some(window) = input.get(at..).and_then(<[u8]>::first_chunk)
        && let Some((len, parts)) = common(window, budget)
    {
        return Ok(Number::new(text(input, at, at + len), index(at), parts));
    }
    scan_exact(input, at, budget)
}

#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "each count is at most 19")]
fn common(window: &[u8; WINDOW], budget: u16) -> Option<(usize, (u16, u16, u64))> {
    let start = usize::from(window[0] == b'-');
    let (integer, mut digits, mut next) = digits_at(window, start)?;
    if integer == 0 || integer > DIGITS_MAX || integer > 1 && window[start] == b'0' {
        return None;
    }
    let mut len = start + integer;
    let mut fraction = 0;
    if next == b'.' {
        let (count, value, after) = digits_at(window, len + 1)?;
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
            let bits = other.trailing_zeros() & 56;
            let digits = word.wrapping_sub(ASCII_ZEROS) << 8 << (56 - bits);
            let count = (bits / 8) as usize;
            let value = value.wrapping_mul(POW10[count]).wrapping_add(digit_values(digits));
            return Some((8 * index + count, value, (word >> bits) as u8));
        }
        value = value.wrapping_mul(100_000_000).wrapping_add(eight_digits(word));
    }
    None
}

#[inline(never)]
fn scan_exact(input: &[u8], at: usize, mut budget: u16) -> Result<Number<'_>, Error> {
    let start = at + usize::from(input[at] == b'-');
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

// RFC 7493 §2.2 recommends exact integers and finite binary64 values. Profile::IJson enforces the documented ranges.
pub(super) fn binary64_holds(number: &Number<'_>) -> bool {
    const INFINITY: (i64, u64) = (292, 17_976_931_348_623_159);
    const ZERO: (i64, u64) = (-340, 24_703_282_292_062_327);
    let input = number.raw.as_bytes();
    let parts = number.parts();
    let integer = &input[parts.integer.clone()];
    let fraction = &input[parts.fraction.clone()];
    if fraction.is_empty() && parts.exponent.is_none() {
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
