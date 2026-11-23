//! The shortest decimal form of a binary64 value that reads back exactly, in the layout of `{:?}`
//! of `core::fmt`.
//!
//! The method is the one of zmij (Zverovich, 2025), which comes from Schubfach (Giulietti). A
//! value `v = c × 2^q` has the rounding interval of half a unit in the last place on each side.
//! For `k = ⌊log10(width of the interval)⌋`, one 128-bit power of ten scales `v` to
//! `X = v × 10^(−k−1)`. The integer part of `X` holds the 15 or 16 leading digits, and the
//! interval is less than 1 wide at that scale. Thus at most one integer is in the interval, and if
//! one is, it is the shortest form. Otherwise the interval holds an integer at the scale `10^k`,
//! and the digit after the integer part of `X`, rounded to the nearest with a tie up as
//! `core::fmt` does, gives the nearest such integer. No loop and no division by 10 run.
//!
//! The decisions follow zmij with no margin and no second path (C216). Schubfach proves that the
//! 128-bit table of [`scale`] decides each bound of the interval exactly, with the parity of `c` for
//! a bound that is a candidate, and that no value is closer to a threshold than the error of the
//! approximations unless it is on it. The 17th digit takes the bias of zmij against the floor of the
//! fraction. A tie of that digit, which only a fraction of exactly a quarter gives, rounds up, as
//! `core::fmt` does; zmij rounds it down. The tests compare each form with `core::fmt`.
//!
//! [`read`] goes the other way: the binary64 value nearest to a decimal of up to 64 bits. It is
//! the parse of unrounded scaling (Cox, 2026, "Floating-Point Printing and Parsing Can Be Simple
//! And Fast"), which Cox proves with Ivy ("Fast Unrounded Scaling: Proof by Ivy", Theorem 1).

#[cfg(test)]
mod tests;

/// The most bytes of the form of an `f64`: `-2.2250738585072014e-308`.
pub(super) const BYTES_MAX: usize = 24;

/// The bytes that [`write`] can store, more than it keeps: it stores whole 8-byte words, the last
/// one from byte 29 at most.
pub(super) const SPAN: usize = 32;

/// Half a unit of `2^−64`, and the bias of zmij for the 17th digit of a regular value: the fraction
/// of [`scale`] is below its exact value by less than 1.1 units, so ten times it by less than 11.
const BIASED_HALF: u64 = (1 << 63) + 6;

/// The least and the greatest `n` of the table of `10^n`: `n = −k − 1` for the `k` of each finite
/// binary64 value.
const POW10_MIN: i32 = -293;
const POW10_MAX: i32 = 324;

/// The 128-bit significand of `10^n` for each `n` from [`POW10_MIN`] to [`POW10_MAX`]: the high and
/// the low 64 bits of `⌊10^n × 2^(127 − ⌊n × log2 10⌋)⌋`.
static POW10: [(u64, u64); 618] = pow10_table(POW10_MIN, POW10_MAX, Rounding::Down);

/// The least and the greatest power of 10 of the table of [`read`]. Each `u64` of digits rounds to
/// zero below it, and each nonzero one rounds to infinity above it.
const POW10_UP_MIN: i32 = -343;
const POW10_UP_MAX: i32 = 308;

/// The table of Cox for [`read`]: for each `p` from [`POW10_UP_MIN`] to [`POW10_UP_MAX`],
/// `pm = ⌈10^p × 2^(127 − ⌊p × log2 10⌋)⌉` as the pair `(hi, lo)` with `pm = hi × 2^64 − lo`. Thus
/// the high word is rounded up.
static POW10_UP: [(u64, u64); 652] = pow10_table(POW10_UP_MIN, POW10_UP_MAX, Rounding::Up);

/// The rounding of the 128-bit significands of a table of powers of 10.
#[derive(Clone, Copy)]
enum Rounding {
    /// The floor, as the high and the low word.
    Down,
    /// The ceiling, as `(hi, lo)` with the value `hi × 2^64 − lo`.
    Up,
}

/// A finite nonzero value without its sign: `c × 2^q`. `irregular` marks a power of two above
/// the subnormals, whose lower neighbor is half as far as its upper neighbor.
#[derive(Clone, Copy, Debug)]
struct Binary {
    c: u64,
    q: i32,
    irregular: bool,
}

/// A decimal value `(digits × 10 + extra) × 10^exponent`: 16 digits, with zeros before them when
/// the value has fewer, and a 17th digit, or 0 when the value has 16 digits or fewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decimal {
    digits: u64,
    extra: u8,
    exponent: i32,
}

/// Writes the shortest form of the finite `value` from the start of `dest` in the layout of `{:?}`,
/// and returns its length, at most [`BYTES_MAX`]. The bytes of `dest` after that length are
/// unspecified.
#[inline]
pub(super) fn write(value: f64, dest: &mut [u8; SPAN]) -> usize {
    let bits = value.to_bits();
    let field = bits & ((1 << 52) - 1);
    let biased = u32::try_from(bits >> 52 & 0x7ff).unwrap_or(0);
    dest[0] = b'-';
    let sign = usize::from(value.is_sign_negative());
    let Some(dest) = dest[sign..].first_chunk_mut::<{ SPAN - 1 }>() else {
        return sign;
    };
    if biased == 0 && field == 0 {
        dest[..3].copy_from_slice(b"0.0");
        return sign + 3;
    }
    // The exponent of a subnormal value is that of the least normal value.
    let binary = if biased == 0 {
        Binary {
            c: field,
            q: -1074,
            irregular: false,
        }
    } else {
        Binary {
            c: field | 1 << 52,
            q: biased.cast_signed() - 1075,
            irregular: field == 0,
        }
    };
    sign + layout(shortest(binary), dest)
}

/// The shortest decimal in the rounding interval of `binary`, the nearest to it if several are,
/// with a tie up.
#[inline]
fn shortest(binary: Binary) -> Decimal {
    let Binary { c, q, irregular } = binary;
    let k = if irregular {
        floor_log10_three_quarters_pow2(q)
    } else {
        floor_log10_pow2(q)
    };
    let (integral, fraction, half) = scale(c, q, k);
    // The nearest integer to ten times `fraction`, in units of 2^−64, after the bias `add`.
    let tenfold = |fraction: u64, add: u64| fraction.carrying_mul(10, add).1;
    let (up, down, digit) = if irregular {
        // The lower neighbor of a power of two is half as far, so the lower bound is a quarter unit
        // below. The least digit in the interval is the ceiling of ten times the lower bound.
        let lower = half >> 1;
        let digit = tenfold(fraction, 1 << 63).max(tenfold(fraction.wrapping_sub(lower), u64::MAX));
        (fraction.overflowing_add(half).1, lower > fraction, digit)
    } else {
        // A bound belongs to the interval when `c` is even: one more unit of `half` takes it in.
        let half = half + u64::from(c & 1 == 0);
        (
            fraction.overflowing_add(half).1,
            half > fraction,
            tenfold(fraction, BIASED_HALF),
        )
    };
    // With neither bound past an integer, the interval holds no shorter form, so the 17th digit
    // is not 0.
    let longer = !up && !down;
    Decimal {
        digits: integral + u64::from(up),
        extra: if longer { u8::try_from(digit).unwrap_or(0) } else { 0 },
        exponent: k,
    }
}

/// `X = c × 2^q × 10^(−k−1)` as its integer part and its fraction in units of `2^−64`, and half a
/// unit in the last place of `c × 2^q` at that scale, in the same units.
///
/// Bounds: the table holds `g = ⌊G⌋` for the exact significand `G` of `10^(−k−1)`. The product
/// with `c' = c × 2^shift < 2^60` is floored at bit 64, so it is below `c' × G / 2^64` by less
/// than 2 units of `2^−70`, and the fraction is below its exact value by less than 1.1 units of
/// `2^−64`. Half a unit is `G × 2^(shift − 71)`, and its floor from the high word of `g` is below
/// it by less than 1.1 units.
#[inline(always)]
fn scale(c: u64, q: i32, k: i32) -> (u64, u64, u64) {
    let n = -k - 1;
    let (high, low) = POW10[(n - POW10_MIN).cast_unsigned() as usize];
    // `10^n = g × 2^(⌊n × log2 10⌋ − 127)`, and `X × 2^70 = (c × 2^shift) × g / 2^64`.
    let shift = (q + floor_log2_pow10(n) + 7).cast_unsigned();
    let scaled = c << shift;
    let (low_product, high_product) = scaled.carrying_mul(high, scaled.carrying_mul(low, 0).1);
    let (integral, fraction) = (high_product >> 6, high_product << 58 | low_product >> 6);
    (integral, fraction, high >> (7 - shift))
}

/// The binary64 value nearest to `digits × 10^power`, with a tie to the even significand, or
/// `None` if its magnitude rounds to infinity. Each `u64` of digits is exact.
///
/// With `e` such that `v = digits × 2^e × 10^power` is in `[2^52, 2^54)`, or `e = 1074` for a
/// subnormal value, one scaling gives `v` unrounded: its integer part, a half bit, and a sticky bit
/// that is set when a lower bit is set. Then one step rounds it to the nearest with a tie to even.
///
/// It inlines into each caller, so [`Number::to_f64`](super::Number::to_f64), which a reader calls
/// out of line, makes no second call.
#[inline(always)]
pub(super) fn read(digits: u64, power: i32) -> Option<f64> {
    let index = usize::try_from(power.saturating_sub(POW10_UP_MIN)).ok();
    let Some(&scale) = index.and_then(|index| POW10_UP.get(index)) else {
        return (digits == 0 || power < 0).then_some(0.0);
    };
    if digits == 0 {
        return Some(0.0);
    }
    let lead = digits.leading_zeros().cast_signed();
    let log2 = floor_log2_pow10(power);
    // `digits` is in `[2^(63 − lead), 2^(64 − lead))` and `10^power` is in `[2^log2, 2^(log2 + 1))`,
    // so this `e` puts `v` in `[2^52, 2^54)`.
    let e = (lead - 11 - log2).min(1074);
    // The shift of the high word of the product keeps two bits below the integer part. It is 8
    // when `e` is not at its cap, and more when it is.
    let shift = lead - e - log2 - 3;
    if shift >= 64 {
        // `v` is under a quarter, so it rounds to zero.
        return Some(0.0);
    }
    let unrounded = uscale(digits << lead, scale, shift.cast_unsigned());
    // A value that rounds to 2^53 or more has one bit too many: `4 × 2^53 − 2` is the least
    // unrounded value that rounds to 2^53.
    let over = unrounded >= (1 << 55) - 2;
    let unrounded = unrounded >> u32::from(over) | unrounded & 1;
    let e = e - i32::from(over);
    let significand = (unrounded + 1 + (unrounded >> 2 & 1)) >> 2;
    // The significand is under 2^53. Under 2^52, the value is subnormal and `e` is 1074, so the
    // biased exponent is 0.
    if significand < 1 << 52 {
        return Some(f64::from_bits(significand));
    }
    let biased = u64::try_from(1075 - e).ok().filter(|&biased| biased < 0x7ff)?;
    Some(f64::from_bits(significand & ((1 << 52) - 1) | biased << 52))
}

/// `x × pm / 2^(128 + shift)` unrounded, for `x` with its high bit set and `pm = hi × 2^64 − lo`
/// from [`POW10_UP`]: the integer part with two more bits, and the sticky bit in the low bit.
///
/// The function of Cox. The high word of `x × hi` is at most one above the high word of the exact
/// product. A set bit of it below `shift` stops the borrow of `x × lo`, so the high bits are
/// final and the result is inexact. Only otherwise does the second product run.
#[inline(always)]
fn uscale(x: u64, (hi, lo): (u64, u64), shift: u32) -> u64 {
    let (middle, mut top) = x.carrying_mul(hi, 0);
    let mut sticky = 1;
    if top & ((1 << shift) - 1) == 0 {
        let (_, borrow) = x.carrying_mul(lo, 0);
        sticky = u64::from(middle.wrapping_sub(borrow) > 1);
        // `top` is at least 2^62, because `x` and `hi` are each at least 2^63.
        top -= u64::from(middle < borrow);
    }
    top >> shift | sticky
}

/// The high and the low 64 bits of `value`. The `const` tables need it: `u64::carrying_mul` is
/// not stable in `const` code.
#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "each half keeps exactly its 64 bits")]
const fn halves(value: u128) -> (u64, u64) {
    ((value >> 64) as u64, value as u64)
}

/// `⌊q × log10 2⌋` for `q` from −1074 to 971.
#[inline(always)]
const fn floor_log10_pow2(q: i32) -> i32 {
    (q * 315_653) >> 20
}

/// `⌊log10(3/4 × 2^q)⌋` for `q` from −1074 to 971.
#[inline(always)]
const fn floor_log10_three_quarters_pow2(q: i32) -> i32 {
    (q * 315_653 - 131_072) >> 20
}

/// `⌊n × log2 10⌋` for `n` from [`POW10_UP_MIN`] to [`POW10_MAX`].
#[inline(always)]
const fn floor_log2_pow10(n: i32) -> i32 {
    (n * 217_706) >> 16
}

/// The table of `10^n` for each `n` from `min` to `max`, a negative `min`, with long arithmetic on
/// integers of [`TABLE_LIMBS`] limbs at compile time.
const fn pow10_table<const LEN: usize>(min: i32, max: i32, rounding: Rounding) -> [(u64, u64); LEN] {
    let mut table = [(0, 0); LEN];
    // 10^n for n from 0, exactly.
    let mut power = [0_u64; TABLE_LIMBS];
    power[0] = 1;
    let mut n = 0;
    while n <= max {
        let (floor, inexact) = high_128(&power);
        table[(n - min).cast_unsigned() as usize] = rounded(floor, inexact, rounding);
        let mut carry = 0;
        let mut index = 0;
        while index < TABLE_LIMBS {
            let product = power[index] as u128 * 10 + carry;
            power[index] = halves(product).1;
            carry = halves(product).0 as u128;
            index += 1;
        }
        n += 1;
    }
    // ⌊2^1279 / 10^m⌋ for m from 1: each division by 10 floors, and ⌊⌊x / a⌋ / b⌋ = ⌊x / ab⌋. No
    // 10^−m times a power of 2 is an integer, so each of these significands is inexact.
    let mut quotient = [0_u64; TABLE_LIMBS];
    quotient[TABLE_LIMBS - 1] = 1 << 63;
    let mut m = 1;
    while -m >= min {
        let mut remainder = 0_u128;
        let mut index = TABLE_LIMBS;
        while index > 0 {
            index -= 1;
            let dividend = remainder << 64 | quotient[index] as u128;
            quotient[index] = halves(dividend / 10).1;
            remainder = dividend % 10;
        }
        table[(-m - min).cast_unsigned() as usize] = rounded(high_128(&quotient).0, true, rounding);
        m += 1;
    }
    table
}

/// The significand whose floor is `(high, low)`, in the form of `rounding`. `inexact` is set if
/// the exact significand is not an integer.
const fn rounded((high, low): (u64, u64), inexact: bool, rounding: Rounding) -> (u64, u64) {
    match rounding {
        Rounding::Down => (high, low),
        Rounding::Up => {
            // The ceiling `high × 2^64 + low`, as `(high + 1) × 2^64 − (2^64 − low)` when `low` is
            // not 0.
            let (low, carry) = low.overflowing_add(inexact as u64);
            let high = high + carry as u64;
            (high + (low != 0) as u64, low.wrapping_neg())
        }
    }
}

/// Enough limbs for 10^324 and for 2^1279 / 10^343 with 128 bits.
const TABLE_LIMBS: usize = 20;

/// The 128 bits of `value` from its highest set bit down, with zeros after its lowest bit, and
/// whether a set bit of `value` is below them.
const fn high_128(value: &[u64; TABLE_LIMBS]) -> ((u64, u64), bool) {
    let mut top = TABLE_LIMBS - 1;
    while value[top] == 0 {
        top -= 1;
    }
    let second = if top >= 1 { value[top - 1] } else { 0 };
    let third = if top >= 2 { value[top - 2] } else { 0 };
    let wide = (value[top] as u128) << 64 | second as u128;
    let lead = value[top].leading_zeros();
    let aligned = if lead == 0 {
        wide
    } else {
        wide << lead | (third >> (64 - lead)) as u128
    };
    // The bits of `third` below the 128, and each limb below `third`.
    let mut inexact = third << lead != 0;
    let mut index = 0;
    while index + 3 <= top {
        inexact |= value[index] != 0;
        index += 1;
    }
    (halves(aligned), inexact)
}

/// Writes `decimal` from the start of `dest` in the layout of `{:?}` and returns its length.
///
/// The value is fixed-point from 1e-4 up to under 1e16, with at least one digit after the point,
/// and otherwise in exponent form. The bounds are on the value, and a value is under such a bound
/// exactly when its shortest form is, because the bound is in the rounding interval of the nearest
/// float to it and not in another one.
#[inline(always)]
fn layout(decimal: Decimal, dest: &mut [u8; SPAN - 1]) -> usize {
    let Decimal {
        digits,
        extra,
        exponent,
    } = decimal;
    // 16 digits with a first digit that is not 0, the 17th digit or 0, and the power of 10 of the
    // first digit. The integer part of a normal binary64 value has 15 or 16 digits, so 15 digits
    // take the 17th digit in with no branch. The fewer digits of a subnormal value take zeros
    // after them in [`widen`].
    let short = digits < POWERS[15];
    let (mut digits, mut extra, mut point) = if short {
        (digits * 10 + u64::from(extra), 0, exponent + 15)
    } else {
        (digits, extra, exponent + 16)
    };
    if digits < POWERS[15] {
        (digits, extra, point) = widen(digits, point);
    }
    match point {
        0..16 => {
            // `point + 1` digits, a point, and the rest of the digits or one zero.
            let whole = point.cast_unsigned() as usize + 1;
            let count = mantissa(digits, extra, whole, dest);
            // `count + 1` when a digit is after the point, else `whole + 2`, with no branch.
            (count + 1).max(whole + 2)
        }
        -4..0 => {
            // `0.`, `−point − 1` zeros, and the digits.
            let at = point.unsigned_abs() as usize + 1;
            dest[..8].copy_from_slice(b"0.000000");
            at + plain(digits, extra, at, dest)
        }
        _ => {
            // The first digit, a point, the other digits, and the exponent. The point stays only
            // before other digits.
            let count = mantissa(digits, extra, 1, dest);
            let at = count + usize::from(count > 1);
            let exponent = EXPONENTS[(point + EXPONENT_BIAS).cast_unsigned() as usize];
            dest[at..at + 8].copy_from_slice(&exponent.to_le_bytes());
            at + (exponent >> 56) as usize
        }
    }
}

// The text of the digits. `mantissa` writes the 16 digits of `digits`, from 10^15 up and under
// 10^16, and `extra`, the 17th digit or 0, from the start of `dest`, with a point after the first
// `whole` digits, from 1 to 16. `plain` writes them from `at`, at most 5, with no point. Each gives
// the count of the digits without the zeros after the last one, or 17 when `extra` is not 0. The
// bytes after the 17th digit are unspecified.
cfg_select! {
    all(target_arch = "aarch64", target_feature = "neon") => {
        use super::aarch64::digits::{mantissa, plain};
    }
    target_arch = "x86_64" => {
        use super::x86::digits::{mantissa, plain};
    }
    _ => {
        use core::hint::select_unpredictable;

        use super::integer::sixteen_digits;

        #[inline(always)]
        fn mantissa(digits: u64, extra: u8, whole: usize, dest: &mut [u8]) -> usize {
            let (high, low, extra, count) = words(digits, extra);
            let words = match whole {
                1..8 => {
                    let (first, spill) = insert(high, whole, b'.');
                    [first, spill | low << 8, low >> 56 | extra << 8]
                }
                8..16 => {
                    let (second, spill) = insert(low, whole - 8, b'.');
                    [high, second, spill | extra << 8]
                }
                _ => [high, low, u64::from(b'.') | extra << 8],
            };
            store(dest, 0, words);
            count
        }

        #[inline(always)]
        fn plain(digits: u64, extra: u8, at: usize, dest: &mut [u8]) -> usize {
            let (high, low, extra, count) = words(digits, extra);
            store(dest, at, [high, low, extra]);
            count
        }

        /// The text of the 16 digits in two words, of the 17th digit, and their count. The
        /// digits stay in words, and each word goes to `dest` with one store, so no load reads a
        /// stored byte back.
        #[inline(always)]
        fn words(digits: u64, extra: u8) -> (u64, u64, u64, usize) {
            const ZEROS: u64 = u64::from_ne_bytes([b'0'; 8]);
            let (high, low) = sixteen_digits(digits);
            let zeros = if low != 0 {
                low.leading_zeros() / 8
            } else {
                8 + high.leading_zeros() / 8
            };
            // A select, not a branch: the count of digits varies from value to value.
            let count = select_unpredictable(extra != 0, 17, 16 - zeros as usize);
            (high | ZEROS, low | ZEROS, u64::from(b'0' + extra), count)
        }

        /// Stores the three little-endian `words` into `dest` from `at`.
        #[inline(always)]
        fn store(dest: &mut [u8], at: usize, words: [u64; 3]) {
            if let Some(dest) = dest.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<24>) {
                let (chunks, _) = dest.as_chunks_mut::<8>();
                for (chunk, word) in chunks.iter_mut().zip(words) {
                    *chunk = word.to_le_bytes();
                }
            }
        }
    }
}

/// `digits`, under 10^15 and not 0, with zeros after it to 16 digits, no 17th digit, and the
/// power of 10 of its first digit, from `point`, the power of 10 of its digit 15.
#[cold]
fn widen(digits: u64, point: i32) -> (u64, u8, i32) {
    let count = digits.checked_ilog10().unwrap_or(0) + 1;
    (
        digits * POWERS[16 - count as usize],
        0,
        point - 16 + count.cast_signed(),
    )
}

/// `word` with `byte` at index `at`, under 8, and the bytes from `at` one place higher, and the
/// byte that leaves the word at the top.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline(always)]
pub(super) const fn insert(word: u64, at: usize, byte: u8) -> (u64, u64) {
    let below = (1 << (8 * at)) - 1;
    (
        word & below | (byte as u64) << (8 * at) | (word & !below) << 8,
        word >> 56,
    )
}

/// `10^n` for each `n` up to 17.
const POWERS: [u64; 18] = {
    let mut powers = [1; 18];
    let mut n = 1;
    while n < 18 {
        powers[n] = powers[n - 1] * 10;
        n += 1;
    }
    powers
};

/// The least power of 10 of the first digit of a finite binary64 value: `5e-324`.
const EXPONENT_BIAS: i32 = 324;

/// The exponent text of each power of 10 from −324 to 308, for example `e-5` and `e100`, in the low
/// bytes, and its length in the high byte.
static EXPONENTS: [u64; 633] = {
    let mut table = [0; 633];
    let mut power = -EXPONENT_BIAS;
    while power <= 308 {
        let mut text = [0_u8; 8];
        text[0] = b'e';
        let mut len = 1_u8;
        if power < 0 {
            text[1] = b'-';
            len = 2;
        }
        let magnitude = power.unsigned_abs();
        let mut place = if magnitude >= 100 {
            100
        } else if magnitude >= 10 {
            10
        } else {
            1
        };
        while place > 0 {
            text[len as usize] = b'0' + (magnitude / place % 10).to_le_bytes()[0];
            len += 1;
            place /= 10;
        }
        text[7] = len;
        table[(power + EXPONENT_BIAS).cast_unsigned() as usize] = u64::from_le_bytes(text);
        power += 1;
    }
    table
};
