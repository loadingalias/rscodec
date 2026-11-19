//! Numbers: the borrowed number token, the grammar of RFC 8259 section 6, and the number rule of
//! I-JSON.

use core::ops::Range;

use super::{Error, ErrorKind, Limit, reader::unexpected};

/// A number of the input, as written. The reader checked its grammar. It borrows the input. `==`
/// compares the text, so `1.0` and `1.00` differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Number<'a> {
    raw: &'a str,
}

impl<'a> Number<'a> {
    #[inline(always)]
    pub(super) const fn new(raw: &'a str) -> Self {
        Self { raw }
    }

    /// The number as written, for example `-1.5e3`. It borrows the input.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'a str {
        self.raw
    }
}

/// The parts of a number that the rule of I-JSON reads.
#[derive(Clone, Debug)]
pub(super) struct Parts {
    /// The digits before the fraction.
    integer: Range<usize>,
    /// The digits of the fraction, or an empty range.
    fraction: Range<usize>,
    /// The value of the exponent, if the number has one. It saturates, which changes no result of
    /// the rule: a number with an exponent past `i64` is far outside binary64.
    exponent: Option<i64>,
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
        let value = input[first..exponent_end].iter().fold(0_i64, |value, &digit| {
            value.saturating_mul(10).saturating_add(i64::from(digit - b'0'))
        });
        let value = if input.get(sign) == Some(&b'-') { -value } else { value };
        (Some(value), exponent_end)
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
