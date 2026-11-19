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
//! The fast path decides from approximations: each is below its exact value by less than 1.1 units
//! of `2^−64` (see [`scale`]). When a decision is closer than [`MARGIN`] to its threshold, the exact
//! path decides it with integer arithmetic. Thus each result is exact, with no tuned bias.

use core::cmp::Ordering;

#[cfg(test)]
mod tests;

use super::integer::sixteen_digits;

/// The most bytes of the form of an `f64`: `-2.2250738585072014e-308`.
pub(super) const BYTES_MAX: usize = 24;

/// The bytes that [`write`] can store, more than it keeps: it stores whole 8-byte words, the last
/// one from byte 29 at most.
pub(super) const SPAN: usize = 32;

/// The distance from a threshold, in units of `2^−64`, under which the fast path does not
/// decide. Each approximation is below its exact value by less than 1.1 units, so 4 is safe for
/// a sum or a difference of two of them.
const MARGIN: u64 = 4;

/// [`MARGIN`] for ten times an approximation.
const TENFOLD_MARGIN: u64 = 16;

/// The least and the greatest `n` of the table of `10^n`: `n = −k − 1` for the `k` of each finite
/// binary64 value.
const POW10_MIN: i32 = -293;
const POW10_MAX: i32 = 324;

/// The 128-bit significand of `10^n` for each `n` from [`POW10_MIN`] to [`POW10_MAX`]: the high and
/// the low 64 bits of `⌊10^n × 2^(127 − ⌊n × log2 10⌋)⌋`.
static POW10: [(u64, u64); 618] = pow10_table();

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
    let lower = if irregular { half >> 1 } else { half };
    // A bound of the interval belongs to it when `c` is even, but off the margin of each threshold
    // the sum is not 0 and `lower` is not `fraction`, so only the exact path needs the parity.
    let (sum, up) = fraction.overflowing_add(half);
    let down = lower > fraction;
    let (mut digit, tenfold) = halves(u128::from(fraction) * 10 + (1 << 63));
    let near = |distance: u64, margin: u64| distance.wrapping_add(margin) <= 2 * margin;
    let longer = !up && !down;
    if near(sum, MARGIN) || near(lower.wrapping_sub(fraction), MARGIN) || longer && near(tenfold, TENFOLD_MARGIN) {
        let (digits, extra) = exact(c, q, irregular, k, integral, fraction);
        return Decimal {
            digits,
            extra,
            exponent: k,
        };
    }
    if longer && irregular {
        // The lower bound is less than half a unit of `10^k` below `v`, so the nearest can be under
        // it. The least digit in the interval is the ceiling of ten times the lower bound.
        let least = u128::from(fraction - lower) * 10;
        if near(halves(least).1, TENFOLD_MARGIN) {
            let (digits, extra) = exact(c, q, irregular, k, integral, fraction);
            return Decimal {
                digits,
                extra,
                exponent: k,
            };
        }
        digit = digit.max(halves(least + u128::from(u64::MAX)).0);
    }
    // `longer` excludes `up`, and the 17th digit is not 0: else the interval holds a shorter form.
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
    let product = u128::from(scaled) * u128::from(high) + ((u128::from(scaled) * u128::from(low)) >> 64);
    let (high_product, low_product) = halves(product);
    let (integral, fraction) = (high_product >> 6, high_product << 58 | low_product >> 6);
    (integral, fraction, high >> (7 - shift))
}

/// The digits and the 17th digit of [`shortest`], with exact integer arithmetic, from the estimates
/// of [`scale`]. It takes and gives scalars, so a call passes no value in memory.
#[cold]
#[inline(never)]
fn exact(c: u64, q: i32, irregular: bool, k: i32, integral: u64, fraction: u64) -> (u64, u8) {
    let n = -k - 1;
    let inclusive = c & 1 == 0;
    let inside = |order: Ordering| order.is_lt() || inclusive && order.is_eq();
    // The estimate of the integer part is its floor, or one below it.
    let floor = integral + u64::from(compare(c, q, n, integral + 1).is_ge());
    let (lower, lower_exponent) = if irregular {
        (4 * c - 1, q - 2)
    } else {
        (2 * c - 1, q - 1)
    };
    if inside(compare(lower, lower_exponent, n, floor)) {
        return (floor, 0);
    }
    if inside(compare(2 * c + 1, q - 1, n, floor + 1).reverse()) {
        return (floor + 1, 0);
    }
    // The nearest integer to `10X` with a tie up: the greatest `d` with `2d − 1 ≤ 20X`. The
    // estimate is within one of it.
    let mut digits = integral * 10 + halves(u128::from(fraction) * 10 + (1 << 63)).0;
    if compare(c, q + 1, n + 1, 2 * digits - 1).is_lt() {
        digits -= 1;
    } else if compare(c, q + 1, n + 1, 2 * digits + 1).is_ge() {
        digits += 1;
    }
    // The lower bound of an irregular value is a third of the interval below it, at least a third
    // of a unit, so one step up reaches the interval.
    if irregular && compare(lower, lower_exponent, n + 1, digits).is_gt() {
        digits += 1;
    }
    (digits / 10, u8::try_from(digits % 10).unwrap_or(0))
}

/// The order of `p × 2^twos × 10^tens` and `y`, exactly.
fn compare(p: u64, twos: i32, tens: i32, y: u64) -> Ordering {
    let (mut left, mut right) = (Big::new(p), Big::new(y));
    if tens >= 0 {
        left.mul_pow5(tens.cast_unsigned());
    } else {
        right.mul_pow5(tens.unsigned_abs());
    }
    let twos = twos + tens;
    if twos >= 0 {
        left.shl(twos.cast_unsigned());
    } else {
        right.shl(twos.unsigned_abs());
    }
    left.0.iter().rev().cmp(right.0.iter().rev())
}

/// The high and the low 64 bits of `value`.
#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "each half keeps exactly its 64 bits")]
const fn halves(value: u128) -> (u64, u64) {
    ((value >> 64) as u64, value as u64)
}

/// An unsigned integer of [`Big::LIMBS`] 64-bit limbs, the least significant first.
struct Big([u64; Big::LIMBS]);

impl Big {
    /// 896 bits: the greatest operand of [`compare`] is `(2^54 + 1) × 5^325`, under 2^810.
    const LIMBS: usize = 14;

    const fn new(value: u64) -> Self {
        let mut limbs = [0; Self::LIMBS];
        limbs[0] = value;
        Self(limbs)
    }

    fn mul_pow5(&mut self, mut power: u32) {
        /// The greatest power of 5 under 2^64.
        const STEP: u32 = 27;
        while power > 0 {
            let factor = 5_u64.pow(power.min(STEP));
            power -= power.min(STEP);
            let mut carry = 0;
            for limb in &mut self.0 {
                let product = u128::from(*limb) * u128::from(factor) + carry;
                let (high, low) = halves(product);
                (*limb, carry) = (low, u128::from(high));
            }
            assert!(carry == 0, "each operand of `compare` fits `Big::LIMBS`");
        }
    }

    fn shl(&mut self, bits: u32) {
        let (limbs, bits) = ((bits / 64) as usize, bits % 64);
        assert!(
            self.0[Self::LIMBS - limbs..].iter().all(|&limb| limb == 0),
            "each operand of `compare` fits `Big::LIMBS`"
        );
        self.0.rotate_right(limbs);
        if bits > 0 {
            let mut carry = 0;
            for limb in &mut self.0 {
                (*limb, carry) = (*limb << bits | carry, *limb >> (64 - bits));
            }
            assert!(carry == 0, "each operand of `compare` fits `Big::LIMBS`");
        }
    }
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

/// `⌊n × log2 10⌋` for `n` from [`POW10_MIN`] to [`POW10_MAX`].
#[inline(always)]
const fn floor_log2_pow10(n: i32) -> i32 {
    (n * 217_706) >> 16
}

/// [`POW10`], with long arithmetic on integers of [`TABLE_LIMBS`] limbs at compile time.
const fn pow10_table() -> [(u64, u64); 618] {
    let mut table = [(0, 0); 618];
    // 10^n for n from 0, exactly.
    let mut power = [0_u64; TABLE_LIMBS];
    power[0] = 1;
    let mut n = 0;
    while n <= POW10_MAX {
        table[(n - POW10_MIN).cast_unsigned() as usize] = high_128(&power);
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
    // ⌊2^1279 / 10^m⌋ for m from 1: each division by 10 floors, and ⌊⌊x / a⌋ / b⌋ = ⌊x / ab⌋.
    let mut quotient = [0_u64; TABLE_LIMBS];
    quotient[TABLE_LIMBS - 1] = 1 << 63;
    let mut m = 1;
    while -m >= POW10_MIN {
        let mut remainder = 0_u128;
        let mut index = TABLE_LIMBS;
        while index > 0 {
            index -= 1;
            let dividend = remainder << 64 | quotient[index] as u128;
            quotient[index] = halves(dividend / 10).1;
            remainder = dividend % 10;
        }
        table[(-m - POW10_MIN).cast_unsigned() as usize] = high_128(&quotient);
        m += 1;
    }
    table
}

/// Enough limbs for 10^324 and for 2^1279 / 10^293 with 128 bits.
const TABLE_LIMBS: usize = 20;

/// The 128 bits of `value` from its highest set bit down, with zeros after its lowest bit.
const fn high_128(value: &[u64; TABLE_LIMBS]) -> (u64, u64) {
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
    halves(aligned)
}

/// Writes `decimal` from the start of `dest` in the layout of `{:?}` and returns its length.
///
/// The value is fixed-point from 1e-4 up to under 1e16, with at least one digit after the point,
/// and otherwise in exponent form. The bounds are on the value, and a value is under such a bound
/// exactly when its shortest form is, because the bound is in the rounding interval of the nearest
/// float to it and not in another one.
///
/// The digits stay in three 64-bit words, and each word goes to `dest` with one store, so no load
/// reads a stored byte back.
#[inline(always)]
fn layout(decimal: Decimal, dest: &mut [u8; SPAN - 1]) -> usize {
    const ZEROS: u64 = u64::from_ne_bytes([b'0'; 8]);
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
    let (high, low) = sixteen_digits(digits);
    // The digits without the zeros after the last one. The 17th digit is not 0.
    let zeros = if low != 0 {
        low.leading_zeros() / 8
    } else {
        8 + high.leading_zeros() / 8
    };
    let count = if extra != 0 { 17 } else { 16 - zeros as usize };
    let (high, low, extra) = (high | ZEROS, low | ZEROS, u64::from(b'0' + extra));
    match point {
        0..16 => {
            // `point + 1` digits, a point, and the rest of the digits or one zero.
            let whole = point.cast_unsigned() as usize + 1;
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
            if count > whole { count + 1 } else { whole + 2 }
        }
        -4..0 => {
            // `0.`, `−point − 1` zeros, and the digits.
            let zeros = point.unsigned_abs() as usize - 1;
            dest[..8].copy_from_slice(b"0.000000");
            store(dest, 2 + zeros, [high, low, extra]);
            2 + zeros + count
        }
        _ => {
            // The first digit, a point, the other digits, and the exponent. The point stays only
            // before other digits.
            let (first, spill) = insert(high, 1, b'.');
            store(dest, 0, [first, spill | low << 8, low >> 56 | extra << 8]);
            let at = if count > 1 { count + 1 } else { 1 };
            let exponent = EXPONENTS[(point + EXPONENT_BIAS).cast_unsigned() as usize];
            dest[at..at + 8].copy_from_slice(&exponent.to_le_bytes());
            at + (exponent >> 56) as usize
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

/// `word` with `byte` at index `at`, under 8, and the bytes from `at` one place higher, and the
/// byte that leaves the word at the top.
#[inline(always)]
const fn insert(word: u64, at: usize, byte: u8) -> (u64, u64) {
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
