//! The contracts of the number accessors. The expected results come from the grammar of RFC 8259
//! section 6, from the parsers of `core`, from the text, and from Nigel Tao's data, not from the
//! accessors.

extern crate std;

use std::{format, fs, string::String, vec::Vec};

use super::super::{Decimal, Error, ErrorKind, Limits, Number, NumberClass, Profile, Slot, read};

/// Limits that no test number reaches.
const LIMITS: Limits = Limits {
    input_bytes_max: u32::MAX,
    depth_max: 1,
    string_bytes_max: 0,
    members_max: 0,
    elements_max: 1,
    number_digits_max: u16::MAX,
};

/// What the accessors give for one number. `f64` holds the bits of the value.
#[derive(Debug, PartialEq)]
struct Values {
    class: NumberClass,
    i64: Result<i64, Error>,
    u64: Result<u64, Error>,
    decimal: Result<Decimal, Error>,
    f64: Result<u64, Error>,
}

/// The values of the accessors for the document `text`, or `None` if the reader refuses it.
///
/// The scan reads a number in the last 48 bytes of the input on its exact path, and each other
/// number on its common path if it can. Thus each text also reads with 48 spaces after it, and the
/// two reads must agree.
fn values(text: &str) -> Option<Values> {
    let read_values = |document: &[u8]| {
        let mut scratch = [Slot::EMPTY; 1];
        read(document, &LIMITS, Profile::Rfc8259, &mut scratch, |reader| {
            let number = reader.value().number()?;
            Ok(Values {
                class: number.class(),
                i64: number.to_i64(),
                u64: number.to_u64(),
                decimal: number.to_decimal(),
                f64: number.to_f64().map(f64::to_bits),
            })
        })
        .ok()
    };
    let values = read_values(text.as_bytes());
    let padded = format!("{text}{:48}", "");
    assert_eq!(read_values(padded.as_bytes()), values, "{text} with spaces after it");
    values
}

/// The parts of `text` if it is a number in the ABNF of RFC 8259 section 6: the integer, the
/// fraction, and the exponent after its `e` or `E`.
///
/// ```text
/// number = [ minus ] int [ frac ] [ exp ]
/// int = zero / ( digit1-9 *DIGIT )
/// frac = decimal-point 1*DIGIT
/// exp = e [ minus / plus ] 1*DIGIT
/// ```
fn grammar(text: &str) -> Option<(&str, Option<&str>, Option<&str>)> {
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    let (mantissa, exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, None), |(mantissa, exponent)| (mantissa, Some(exponent)));
    let (integer, fraction) = mantissa
        .split_once('.')
        .map_or((mantissa, None), |(integer, fraction)| (integer, Some(fraction)));
    let integer_holds = integer == "0" || digits(integer) && !integer.starts_with('0');
    let exponent_holds = |exponent: &str| digits(exponent.strip_prefix(['-', '+']).unwrap_or(exponent));
    (integer_holds && fraction.is_none_or(digits) && exponent.is_none_or(exponent_holds))
        .then_some((integer, fraction, exponent))
}

/// The values of `text`, a number at offset 0, from the grammar, from the parsers of `core`, and
/// from the digits of the text.
fn oracle(text: &str) -> Option<Values> {
    let (integer, fraction, exponent) = grammar(text)?;
    let refused = |kind| Error::new(kind, 0);
    let class = match (fraction, exponent) {
        (_, Some(_)) => NumberClass::Float,
        (Some(_), None) => NumberClass::Decimal,
        (None, None) => NumberClass::Integer,
    };
    let integer_value = (class == NumberClass::Integer)
        .then(|| text.parse::<i128>().ok())
        .ok_or_else(|| refused(ErrorKind::Type));
    let integer_value = |fits: fn(i128) -> bool| match integer_value {
        Ok(Some(value)) if fits(value) => Ok(value),
        Ok(_) => Err(refused(ErrorKind::Range)),
        Err(error) => Err(error),
    };
    let fraction = fraction.unwrap_or("");
    let significant = format!("{integer}{fraction}");
    let significant = significant.trim_start_matches('0');
    let fraction_len = i128::try_from(fraction.len()).unwrap();
    let exponent = exponent.map_or(Some(0), |exponent| exponent.parse::<i128>().ok());
    let decimal = match exponent.and_then(|exponent| i32::try_from(exponent - fraction_len).ok()) {
        Some(exponent) if significant.len() <= 38 => Ok(Decimal {
            negative: text.starts_with('-'),
            coefficient: if significant.is_empty() {
                0
            } else {
                significant.parse().unwrap()
            },
            exponent,
        }),
        _ => Err(refused(ErrorKind::Range)),
    };
    let value: f64 = text.parse().unwrap();
    Some(Values {
        class,
        i64: integer_value(|value| i64::try_from(value).is_ok()).map(|value| i64::try_from(value).unwrap()),
        u64: integer_value(|value| u64::try_from(value).is_ok()).map(|value| u64::try_from(value).unwrap()),
        decimal,
        f64: if value.is_finite() {
            Ok(value.to_bits())
        } else {
            Err(refused(ErrorKind::Range))
        },
    })
}

/// Each string of up to 7 bytes over `0`, `1`, `9`, `-`, `+`, `.`, `e`, `E`, and `x`, a byte
/// outside the grammar, is a number exactly when the grammar of RFC 8259 section 6 accepts it.
/// Then each accessor gives the value of the oracle. The bound holds `-0.0e+0`, which uses each
/// rule of the grammar.
#[test]
fn each_short_string_is_a_number_exactly_when_the_grammar_accepts_it() {
    const ALPHABET: &[u8; 9] = b"019-+.eEx";
    let mut text = String::new();
    for len in 1..=7 {
        for index in 0..ALPHABET.len().pow(len) {
            text.clear();
            let mut rest = index;
            for _ in 0..len {
                text.push(char::from(ALPHABET[rest % ALPHABET.len()]));
                rest /= ALPHABET.len();
            }
            assert_eq!(values(&text), oracle(&text), "{text}");
        }
    }
}

/// Each line of Nigel Tao's data: a string that the grammar accepts reads as the binary64 value of
/// its line, or as the range error if that value is an infinity. Each other string refuses.
#[test]
fn each_number_of_nigel_taos_data_reads_its_binary64_value() {
    const FILES: [&str; 14] = [
        "exhaustive-float16",
        "freetype-2-7",
        "google-double-conversion",
        "google-wuffs",
        "ibm-fpgen",
        "lemire-fast-double-parser",
        "lemire-fast-float",
        "more-test-cases",
        "remyoudompheng-fptest-0",
        "remyoudompheng-fptest-1",
        "remyoudompheng-fptest-2",
        "remyoudompheng-fptest-3",
        "tencent-rapidjson",
        "ulfjack-ryu",
    ];
    for name in FILES {
        let path = format!("tests/data/fetched/numbers/{name}.txt");
        let data = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{path}: {error}; tests/data/README.md fetches it"));
        for line in data.lines() {
            // Each line has the bits of the f16, f32, and f64 values, and then the string.
            let fields: Vec<_> = line.splitn(4, ' ').collect();
            let [_, _, bits, text] = fields[..] else {
                panic!("{path}: the line {line:?} has 4 fields");
            };
            let bits = u64::from_str_radix(bits, 16).unwrap();
            let expected = grammar(text).map(|_| {
                if f64::from_bits(bits).is_infinite() {
                    Err(Error::new(ErrorKind::Range, 0))
                } else {
                    Ok(bits)
                }
            });
            assert_eq!(values(text).map(|values| values.f64), expected, "{path}: {text}");
        }
    }
}

/// A fixed-seed sample of finite `f64` values reads back from three forms of each: the shortest,
/// the exponent form, and 31 significant digits, which take the path of more than 19 digits.
#[test]
fn each_value_of_a_fixed_seed_sample_reads_back_from_each_form() {
    let mut state = 0x5eed_0004_u64;
    for _ in 0..1 << 18 {
        // xorshift64*: a fixed seed gives the same values on each run.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state.wrapping_mul(0x2545_f491_4f6c_dd1d));
        if !value.is_finite() {
            continue;
        }
        for text in [format!("{value:?}"), format!("{value:e}"), format!("{value:.30e}")] {
            let read = values(&text).map(|values| values.f64);
            assert_eq!(read, Some(Ok(value.to_bits())), "{text}");
        }
    }
}

/// The value of `accessor` on the number `text` in an array, at offset 1.
fn at_one<T>(text: &str, accessor: impl FnOnce(Number<'_>) -> Result<T, Error>) -> Result<T, Error> {
    let mut scratch = [Slot::EMPTY; 1];
    let document = format!("[{text}]");
    read(document.as_bytes(), &LIMITS, Profile::Rfc8259, &mut scratch, |reader| {
        let mut elements = reader.value().elements()?;
        let number = elements.next()?.expect("the array has an element").number()?;
        accessor(number)
    })
}

/// Each accessor reads the values at its limits. It refuses each value past them with its typed
/// error at the offset of the number, so `?` in the closure of `read` gives that error.
#[test]
fn each_accessor_reads_to_its_limits_and_refuses_past_them() {
    let range = Error::new(ErrorKind::Range, 1);
    let other_type = Error::new(ErrorKind::Type, 1);
    for (text, expected) in [
        ("9223372036854775807", Ok(i64::MAX)),
        ("-9223372036854775808", Ok(i64::MIN)),
        ("9223372036854775808", Err(range)),
        ("-9223372036854775809", Err(range)),
        ("-0", Ok(0)),
        ("100000000000000000000", Err(range)),
        ("1.0", Err(other_type)),
        ("1e2", Err(other_type)),
    ] {
        assert_eq!(at_one(text, |number| number.to_i64()), expected, "{text}");
    }
    for (text, expected) in [
        ("18446744073709551615", Ok(u64::MAX)),
        ("18446744073709551616", Err(range)),
        ("99999999999999999999", Err(range)),
        ("100000000000000000000", Err(range)),
        ("-1", Err(range)),
        ("-0", Ok(0)),
        ("0.5", Err(other_type)),
        ("1E2", Err(other_type)),
    ] {
        assert_eq!(at_one(text, |number| number.to_u64()), expected, "{text}");
    }
    let decimal = |negative, coefficient, exponent| {
        Ok(Decimal {
            negative,
            coefficient,
            exponent,
        })
    };
    let zeros_40 = "0".repeat(40);
    for (text, expected) in [
        (&*"9".repeat(38), decimal(false, 10_u128.pow(38) - 1, 0)),
        (&*format!("1{}", "0".repeat(38)), Err(range)),
        (&*format!("0.{zeros_40}1"), decimal(false, 1, -41)),
        (&*format!("1.{zeros_40}"), Err(range)),
        ("1.50", decimal(false, 150, -2)),
        ("-0.0", decimal(true, 0, -1)),
        ("-1.5e3", decimal(true, 15, 2)),
        ("1e2147483647", decimal(false, 1, i32::MAX)),
        ("1e2147483648", Err(range)),
        ("1.5e-2147483647", decimal(false, 15, i32::MIN)),
        ("1.5e-2147483648", Err(range)),
        ("1e999999999999999999999999", Err(range)),
    ] {
        assert_eq!(at_one(text, |number| number.to_decimal()), expected, "{text}");
    }
    let bits = |value: f64| Ok(value.to_bits());
    for (text, expected) in [
        ("1.7976931348623157e308", bits(f64::MAX)),
        // The halfway point to 2^1024 is 1.797693134862315807…e308. A value at it or above it
        // rounds to infinity, because the significand of `f64::MAX` is odd.
        ("1.7976931348623158e308", bits(f64::MAX)),
        ("1.7976931348623159e308", Err(range)),
        ("-1.7976931348623159e308", Err(range)),
        ("17976931348623159000000e286", Err(range)),
        ("1e309", Err(range)),
        ("1e-400", bits(0.0)),
        ("-1e-400", bits(-0.0)),
        ("-0", bits(-0.0)),
        ("0e99999999999999999999", bits(0.0)),
        // Half of the least subnormal value is 2.4703282292062327208…e-324.
        ("2.4703282292062327e-324", bits(0.0)),
        ("2.4703282292062328e-324", bits(5e-324)),
        ("2.2250738585072014e-308", bits(f64::MIN_POSITIVE)),
    ] {
        assert_eq!(
            at_one(text, |number| number.to_f64().map(f64::to_bits)),
            expected,
            "{text}"
        );
    }
}
