//! The printer against `{:?}` of `core::fmt`, which prints the shortest digits that read back,
//! the nearest to the value, with a tie up.

extern crate std;

use core::cmp::Ordering;
use std::{format, string::String};

use super::{Binary, SPAN, floor_log2_pow10, floor_log10_pow2, floor_log10_three_quarters_pow2, scale, shortest};

fn printed(value: f64) -> String {
    let mut dest = [0; SPAN];
    let len = super::write(value, &mut dest);
    String::from_utf8(dest[..len].to_vec()).expect("the printer writes ASCII")
}

#[track_caller]
fn check_f64(value: f64) {
    assert_eq!(
        printed(value),
        format!("{value:?}"),
        "the form of {:#018x}",
        value.to_bits()
    );
}

/// xorshift64*: a fixed seed gives the same values on each run.
fn random(mut state: u64) -> impl Iterator<Item = u64> {
    core::iter::repeat_with(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    })
}

/// The binary form and the digits of `core::fmt` of the finite nonzero `value`, as the pair of the
/// integer of its digits and the power of 10 of its last digit.
fn binary_and_digits(value: f64) -> (Binary, (u64, i32)) {
    let bits = value.to_bits();
    let (biased, field) = ((bits >> 52 & 0x7ff) as i32, bits & ((1 << 52) - 1));
    let binary = if biased == 0 {
        Binary {
            c: field,
            q: -1074,
            irregular: false,
        }
    } else {
        Binary {
            c: field | 1 << 52,
            q: biased - 1075,
            irregular: field == 0,
        }
    };
    let text = format!("{:e}", value.abs());
    let (mantissa, exponent) = text.split_once('e').expect("the exponent form has an exponent");
    let fraction = mantissa.split_once('.').map_or(0, |(_, fraction)| fraction.len());
    let digits = mantissa.replace('.', "").parse().expect("the mantissa is digits");
    let exponent: i32 = exponent.parse().expect("the exponent is an integer");
    (
        binary,
        (digits, exponent - i32::try_from(fraction).expect("a short fraction")),
    )
}

/// `digits × 10^exponent` without its trailing zeros.
fn normal((mut digits, mut exponent): (u64, i32)) -> (u64, i32) {
    while digits % 10 == 0 {
        digits /= 10;
        exponent += 1;
    }
    (digits, exponent)
}

/// Values whose decisions are close to their thresholds: the boundary cases of the tests of zmij,
/// exact ties of the 17th digit (`c` odd and `q = −2`), and the bounds of the layout.
const HARD: &[f64] = &[
    6.62607015e-34,
    5.444310685350916e14,
    1.3588129002659584e-245,
    2.9802322387695312e-8,
    5.960464477539063e-8,
    1.3076622631878654e65,
    9.03725590277404e159,
    9.03725590277404e160,
    9.03725590277404e161,
    -4.932096661796888e-226,
    3.439070283483335e35,
    6.606854224493745e-17,
    6.079537928711555e61,
    1_125_899_906_842_624.3,
    1_125_899_906_842_624.8,
    2_251_799_813_685_247.8,
    1e-4,
    9.999999999999999e-5,
    1e16,
    9_999_999_999_999_998.0,
    1e15,
    0.1,
    0.3,
    2.2250738585072014e-308,
    2.225073858507201e-308,
    5e-324,
    1e-323,
    f64::MAX,
    f64::MIN_POSITIVE,
    f64::EPSILON,
    123_456_789_012_345_680.0,
];

#[test]
fn each_logarithm_formula_holds_on_its_whole_domain() {
    let at_least_one = |p, twos, tens| compare(p, twos, tens, 1) != Ordering::Less;
    for q in -1074..=971 {
        let k = floor_log10_pow2(q);
        assert!(
            at_least_one(1, q, -k) && !at_least_one(1, q, -k - 1),
            "⌊log10 2^{q}⌋ is {k}"
        );
        let k = floor_log10_three_quarters_pow2(q);
        assert!(
            at_least_one(3, q - 2, -k) && !at_least_one(3, q - 2, -k - 1),
            "⌊log10(3/4 × 2^{q})⌋ is {k}"
        );
    }
    for n in super::POW10_UP_MIN..=super::POW10_MAX {
        let f = floor_log2_pow10(n);
        assert!(
            at_least_one(1, -f, n) && !at_least_one(1, -f - 1, n),
            "⌊log2 10^{n}⌋ is {f}"
        );
    }
}

#[test]
fn each_hard_value_prints_as_core_fmt() {
    for &value in HARD {
        check_f64(value);
        check_f64(-value);
    }
}

#[test]
fn each_power_of_two_prints_as_core_fmt() {
    for biased in 1..0x7ff_u64 {
        check_f64(f64::from_bits(biased << 52));
        check_f64(-f64::from_bits(biased << 52));
    }
    // The powers of two of binary32, as binary64 values.
    for biased in 1..0xff_u32 {
        check_f64(f64::from(f32::from_bits(biased << 23)));
        check_f64(-f64::from(f32::from_bits(biased << 23)));
    }
}

#[test]
fn each_exponent_prints_as_core_fmt_with_its_extreme_and_random_significands() {
    let mut bits = random(0x5eed_0001);
    for biased in 0..0x7ff_u64 {
        let extremes = [0, 1, 2, 3, (1 << 52) - 1, (1 << 52) - 2, 1 << 51];
        for field in extremes
            .into_iter()
            .chain(bits.by_ref().take(256).map(|bits| bits >> 12))
        {
            check_f64(f64::from_bits(biased << 52 | field));
        }
    }
    // The binary32 exponents, as binary64 values.
    for biased in 0..0xff_u32 {
        let extremes = [0, 1, 2, 3, (1 << 23) - 1, (1 << 23) - 2, 1 << 22];
        let fields = extremes
            .into_iter()
            .chain(bits.by_ref().take(256).map(|bits| (bits >> 41) as u32));
        for field in fields {
            check_f64(f64::from(f32::from_bits(biased << 23 | field)));
        }
    }
}

#[test]
fn each_value_of_a_fixed_seed_sample_prints_as_core_fmt_and_reads_back() {
    for bits in random(0x5eed_0002).take(1 << 21) {
        let value = f64::from_bits(bits);
        if value.is_finite() {
            check_f64(value);
            assert_eq!(
                printed(value).parse::<f64>().map(f64::to_bits),
                Ok(bits),
                "{bits:#018x} reads back"
            );
        }
    }
}

/// `shortest` gives the digits of `core::fmt` on a sample, on each power of two, and on the hard
/// values, before the layout.
#[test]
fn shortest_gives_the_digits_of_core_fmt() {
    let sample = random(0x5eed_0003).take(1 << 16).map(f64::from_bits);
    let powers = (1..0x7ff_u64).map(|biased| f64::from_bits(biased << 52));
    for value in sample.chain(powers).chain(HARD.iter().copied()) {
        if !value.is_finite() || value == 0.0 {
            continue;
        }
        let (binary, expected) = binary_and_digits(value);
        let decimal = shortest(binary);
        assert_eq!(
            normal((decimal.digits * 10 + u64::from(decimal.extra), decimal.exponent)),
            expected,
            "the digits of {value:e}"
        );
    }
}

/// Sample both sides of each rounding threshold in units of 2⁻⁶⁴.
const BAND: u64 = 4;
const TENFOLD_BAND: u64 = 16;

/// Whether the estimates for `binary` are in a band, so that only an exact decision gives the
/// form of `core::fmt`.
fn in_band(binary: Binary) -> bool {
    let Binary { c, q, irregular } = binary;
    let k = if irregular {
        floor_log10_three_quarters_pow2(q)
    } else {
        floor_log10_pow2(q)
    };
    let (_, fraction, half) = scale(c, q, k);
    let lower = if irregular { half >> 1 } else { half };
    let near = |distance: u64, band: u64| distance.wrapping_add(band) <= 2 * band;
    let (tenfold, _) = fraction.carrying_mul(10, 1 << 63);
    near(fraction.wrapping_add(half), BAND) || near(lower.wrapping_sub(fraction), BAND) || near(tenfold, TENFOLD_BAND)
}

/// Values that random inputs reach about once in 2^60, each built to fall in a band:
///
/// - Each exact tie of the 17th digit: a value `c / 4` with `c` odd, which has a fraction of a
///   quarter, and each power of two in a band, for example 2^−25.
/// - Each bound of an interval that is exactly a decimal of the scale of the 16-digit candidate:
///   `(2c ± 1) × 2^(q − 1)` is a multiple of `10^(k + 1)` when `5^(k + 1)` divides `2c ± 1` and
///   `q ≥ k + 2`. Both parities of `c` occur, so a bound belongs to the interval and does not.
fn band_values() -> std::vec::Vec<f64> {
    let mut values = std::vec::Vec::new();
    let odd = random(0x5eed_0006).map(|bits| (bits >> 12 | 1 << 52) | 1);
    for c in [(1 << 52) + 1, (1 << 52) + 3, (1 << 53) - 1]
        .into_iter()
        .chain(odd.take(1 << 12))
    {
        values.push(Binary {
            c,
            q: -2,
            irregular: false,
        });
    }
    for biased in 1..0x7ff {
        values.push(Binary {
            c: 1 << 52,
            q: biased - 1075,
            irregular: true,
        });
    }
    for q in 3..80 {
        let modulus = 5_u64.pow((floor_log10_pow2(q) + 1).cast_unsigned());
        if modulus >= 1 << 51 {
            break;
        }
        // `c` for `2c + 1` and for `2c − 1` divisible by the modulus, from 2^52 up, and the next one.
        for residue in [(modulus - 1) / 2, modulus.div_ceil(2)] {
            let first = (1 << 52) + (residue + modulus - (1 << 52) % modulus) % modulus;
            values.extend([first, first + modulus].map(|c| Binary { c, q, irregular: false }));
        }
    }
    values
        .into_iter()
        .filter(|&binary| in_band(binary))
        .map(|Binary { c, q, .. }| f64::from_bits(u64::from((q + 1075).cast_unsigned()) << 52 | c & ((1 << 52) - 1)))
        .collect()
}

#[test]
fn each_value_in_a_band_prints_as_core_fmt() {
    let values = band_values();
    // The ties of `c / 4`, the bounds on a decimal, and the powers of two in a band, such as 2^-25.
    assert!(values.len() > 1 << 12, "{} values are in a band", values.len());
    for value in [2.9802322387695312e-8, 1_125_899_906_842_624.3] {
        assert!(values.contains(&value), "{value:e} is in a band");
    }
    for value in values {
        check_f64(value);
        check_f64(-value);
    }
}

/// The order of `p × 2^twos × 10^tens` and `y`, exactly: the oracle of the logarithm formulas.
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
                (*limb, carry) = limb.carrying_mul(factor, carry);
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
