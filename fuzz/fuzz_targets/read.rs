//! The differential target of the JSON reader (C206). For each input, in each profile:
//!
//! 1. The read paths of rscodec agree: the tokens of `visit`, the value walk, a walk that drops
//!    each odd item, `Value::skip`, and `read` with a closure that reads nothing. Each gives the same
//!    result, error, and offset, and the tokens and the value walk give the same tree.
//! 2. `IJson` refuses what `Rfc8259` refuses, and also each noncharacter and each number that
//!    binary64 does not hold, at its offset.
//! 3. The accessors of each number give the values of the parsers of `core`.
//! 4. serde_json and simdjson give the tree of `Rfc8259`, except for the rules in README.md.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rscodec::json::{self, Error, ErrorKind, Kind, Limit, Limits, Number, NumberClass, Profile, Slot, Str, Token, Value};

/// serde_json refuses the 128th container. simdjson does not count an empty container, so it can
/// accept a document that is deeper.
const DEPTH_MAX: u16 = 127;

/// No limit other than the depth refuses an input of the fuzz.
const LIMITS: Limits = Limits {
    input_bytes_max: u32::MAX,
    depth_max: DEPTH_MAX,
    string_bytes_max: u32::MAX,
    members_max: u32::MAX,
    elements_max: u32::MAX,
    number_digits_max: u16::MAX,
};

fuzz_target!(|input: &[u8]| check(input));

fn check(input: &[u8]) {
    let rfc = read(input, Profile::Rfc8259);
    let ijson = read(input, Profile::IJson);
    check_profiles(input, &rfc, &ijson);
    let serde = serde_json::from_slice::<serde_json::Value>(input).map(Oracle::from);
    check_oracle("serde_json", Order::Sorted, &rfc, serde.map_err(drop));
    // simdjson skips a byte order mark, which rscodec refuses. RFC 8259 section 8.1 allows both.
    let simdjson = simdjson(input).map_err(drop);
    match input.strip_prefix(b"\xef\xbb\xbf") {
        Some(rest) => {
            let error = rfc.as_ref().err().map(|error| (error.kind(), error.offset()));
            assert_eq!(error, Some((ErrorKind::Syntax, 0)), "rscodec refuses a byte order mark");
            check_oracle("simdjson", Order::Kept, &read(rest, Profile::Rfc8259), simdjson);
        }
        None => check_oracle("simdjson", Order::Kept, &rfc, simdjson),
    }
}

/// A value that rscodec reads. A number keeps its text and its offset.
#[derive(Debug, PartialEq)]
enum Tree<'a> {
    Null,
    Bool(bool),
    Number(&'a str, usize),
    String(String),
    Array(Vec<Tree<'a>>),
    Object(Vec<(String, Tree<'a>)>),
}

/// The items that a value walk reads.
#[derive(Clone, Copy)]
enum Walk {
    /// Each item.
    Each,
    /// Each even item. The walk drops each odd item, so the reader skips it.
    Even,
}

/// The document in `profile`, after a check that each read path gives the same result.
fn read(input: &[u8], profile: Profile) -> Result<Tree<'_>, Error> {
    let mut scratch = vec![Slot::EMPTY; input.len() + 1];
    let tokens = token_tree(input, profile, &mut scratch);
    let values = json::read(input, &LIMITS, profile, &mut scratch, |reader| value_tree(reader.value(), input, Walk::Each));
    let even = json::read(input, &LIMITS, profile, &mut scratch, |reader| value_tree(reader.value(), input, Walk::Even));
    let skip = json::read(input, &LIMITS, profile, &mut scratch, |reader| reader.value().skip());
    let finish = json::read(input, &LIMITS, profile, &mut scratch, |_| Ok(()));
    let error = tokens.as_ref().err().copied();
    for (path, other) in [
        ("value walk", values.as_ref().err().copied()),
        ("even walk", even.err()),
        ("skip", skip.err()),
        ("finish", finish.err()),
    ] {
        assert_eq!(other, error, "{profile:?}: the {path} and the token walk");
    }
    if let (Ok(tokens), Ok(values)) = (&tokens, &values) {
        assert_eq!(tokens, values, "{profile:?}: the trees of the walks");
    }
    tokens
}

fn token_tree<'a>(input: &'a [u8], profile: Profile, scratch: &mut [Slot]) -> Result<Tree<'a>, Error> {
    // Each open container, with its key in its parent.
    let mut open: Vec<(Option<String>, Tree<'a>)> = Vec::new();
    let (mut key, mut root) = (None, None);
    json::visit(input, &LIMITS, profile, scratch, |token| {
        let value = match token {
            Token::ObjectStart => {
                open.push((key.take(), Tree::Object(Vec::new())));
                return Ok(());
            }
            Token::ArrayStart => {
                open.push((key.take(), Tree::Array(Vec::new())));
                return Ok(());
            }
            Token::Key(name) => {
                key = Some(text(name));
                return Ok(());
            }
            Token::ObjectEnd | Token::ArrayEnd => {
                let (name, tree) = open.pop().expect("an end has its start");
                assert_eq!(matches!(tree, Tree::Object(_)), token == Token::ObjectEnd, "the end of its start");
                key = name;
                tree
            }
            Token::Str(string) => Tree::String(text(string)),
            Token::Number(number) => checked(number, input),
            Token::Bool(value) => Tree::Bool(value),
            Token::Null => Tree::Null,
        };
        match open.last_mut() {
            None => assert!(root.replace(value).is_none(), "a document has one value"),
            Some((_, Tree::Array(items))) => items.push(value),
            Some((_, Tree::Object(members))) => members.push((key.take().expect("a member has a key"), value)),
            Some(_) => unreachable!("only containers are open"),
        }
        Ok(())
    })?;
    assert!(open.is_empty(), "the end of the document closes each container");
    Ok(root.expect("a document has a value"))
}

fn value_tree<'a>(mut value: Value<'_, 'a, '_>, input: &[u8], walk: Walk) -> Result<Tree<'a>, Error> {
    let read = |index: usize| matches!(walk, Walk::Each) || index % 2 == 0;
    Ok(match value.kind()? {
        Kind::Object => {
            let (mut members, mut tree) = (value.members()?, Vec::new());
            while let Some((key, value)) = members.next()? {
                if read(tree.len()) {
                    tree.push((text(key), value_tree(value, input, walk)?));
                } else {
                    tree.push((text(key), Tree::Null));
                }
            }
            Tree::Object(tree)
        }
        Kind::Array => {
            let (mut elements, mut tree) = (value.elements()?, Vec::new());
            while let Some(value) = elements.next()? {
                tree.push(if read(tree.len()) { value_tree(value, input, walk)? } else { Tree::Null });
            }
            Tree::Array(tree)
        }
        Kind::String => Tree::String(text(value.str()?)),
        Kind::Number => checked(value.number()?, input),
        Kind::Bool => Tree::Bool(value.bool()?),
        Kind::Null => value.null().map(|()| Tree::Null)?,
    })
}

/// The string after unescape.
fn text(string: Str<'_>) -> String {
    let mut buf = vec![0; string.len()];
    let decoded = string.decode(&mut buf).expect("a buffer of `len` bytes holds the string").to_owned();
    assert!(string == decoded.as_str(), "`==` compares after unescape: {decoded:?}");
    decoded
}

/// The number, after a check of each accessor with the parsers of `core`.
fn checked<'a>(number: Number<'a>, input: &[u8]) -> Tree<'a> {
    let text = number.as_str();
    let class = if text.contains(['e', 'E']) {
        NumberClass::Float
    } else if text.contains('.') {
        NumberClass::Decimal
    } else {
        NumberClass::Integer
    };
    assert_eq!(number.class(), class, "{text}");
    let float: f64 = text.parse().expect("a JSON number is a Rust float");
    match number.to_f64() {
        Ok(value) => assert!(float.is_finite() && value.to_bits() == float.to_bits(), "{text}: {value:?}"),
        Err(error) => assert!(float.is_infinite() && error.kind() == ErrorKind::Range, "{text}: {error:?}"),
    }
    if class == NumberClass::Integer {
        let unsigned = if text == "-0" { Ok(0) } else { text.parse::<u64>() };
        assert_eq!(kind(number.to_i64()), text.parse::<i64>().map_err(|_| ErrorKind::Range), "{text}");
        assert_eq!(kind(number.to_u64()), unsigned.map_err(|_| ErrorKind::Range), "{text}");
    } else {
        assert_eq!(kind(number.to_i64()), Err(ErrorKind::Type), "{text}");
        assert_eq!(kind(number.to_u64()), Err(ErrorKind::Type), "{text}");
    }
    let decimal = number.to_decimal().map(|decimal| (decimal.negative, decimal.coefficient, decimal.exponent));
    assert_eq!(kind(decimal), decimal_of(text).ok_or(ErrorKind::Range), "{text}");
    Tree::Number(text, text.as_ptr().addr() - input.as_ptr().addr())
}

fn kind<T>(result: Result<T, Error>) -> Result<T, ErrorKind> {
    result.map_err(|error| error.kind())
}

/// The sign, the coefficient, and the exponent of `text`, if the coefficient has at most 38
/// significant digits and the exponent fits `i32`.
fn decimal_of(text: &str) -> Option<(bool, u128, i32)> {
    let (negative, text) = text.strip_prefix('-').map_or((false, text), |rest| (true, rest));
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{integer}{fraction}");
    let significant = digits.trim_start_matches('0');
    if significant.len() > 38 {
        return None;
    }
    let coefficient = if significant.is_empty() { 0 } else { significant.parse().ok()? };
    let exponent = exponent.parse::<i64>().ok()?.checked_sub(i64::try_from(fraction.len()).ok()?)?;
    Some((negative, coefficient, i32::try_from(exponent).ok()?))
}

/// Whether binary64 holds the number `text`, in the rule of `IJson`.
fn binary64_holds(text: &str) -> bool {
    if !text.contains(['.', 'e', 'E']) {
        return text.parse::<i64>().is_ok_and(|value| value.unsigned_abs() < 1 << 53);
    }
    let mantissa = text.split(['e', 'E']).next().unwrap_or(text);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let significant = digits.trim_start_matches('0').trim_end_matches('0').len();
    let value: f64 = text.parse().expect("a JSON number is a Rust float");
    significant <= 17 && value.is_finite() && (value != 0.0 || significant == 0)
}

fn noncharacter(value: u32) -> bool {
    (0xfdd0..=0xfdef).contains(&value) || value & 0xfffe == 0xfffe
}

/// Whether the character or the escape at `at` is a noncharacter.
fn noncharacter_at(input: &[u8], at: usize) -> bool {
    let hex = |from: usize| {
        let digits = input.get(from..from + 4)?;
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u32::from_str_radix(core::str::from_utf8(digits).ok()?, 16).ok()
    };
    let value = if input.get(at) == Some(&b'\\') {
        match hex(at + 2) {
            Some(high @ 0xd800..0xdc00) => match hex(at + 8) {
                Some(low @ 0xdc00..0xe000) => 0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00),
                _ => return false,
            },
            Some(value) => value,
            None => return false,
        }
    } else {
        let tail = &input[at..input.len().min(at + 4)];
        let valid = core::str::from_utf8(tail).unwrap_or_else(|error| {
            core::str::from_utf8(&tail[..error.valid_up_to()]).expect("the prefix is UTF-8")
        });
        match valid.chars().next() {
            Some(character) => u32::from(character),
            None => return false,
        }
    };
    noncharacter(value)
}

/// The JSON number at `at`.
fn number_at(input: &[u8], at: usize) -> &str {
    let digits = |mut end: usize| {
        while input.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        end
    };
    let mut end = digits(at + usize::from(input.get(at) == Some(&b'-')));
    if input.get(end) == Some(&b'.') {
        end = digits(end + 1);
    }
    if matches!(input.get(end), Some(b'e' | b'E')) {
        end += 1 + usize::from(matches!(input.get(end + 1), Some(b'+' | b'-')));
        end = digits(end);
    }
    core::str::from_utf8(&input[at..end]).expect("a number is ASCII")
}

/// Whether the refusal of `IJson`, but not of `Rfc8259`, has its cause at its offset.
fn ijson_cause(input: &[u8], error: Error) -> bool {
    match error.kind() {
        ErrorKind::Number => !binary64_holds(number_at(input, error.offset())),
        ErrorKind::Noncharacter => noncharacter_at(input, error.offset()),
        _ => false,
    }
}

/// Whether `IJson` holds each value of `tree`.
fn ijson_holds(tree: &Tree<'_>) -> bool {
    let clean = |text: &str| !text.chars().any(|character| noncharacter(u32::from(character)));
    match tree {
        Tree::Null | Tree::Bool(_) => true,
        Tree::Number(text, _) => binary64_holds(text),
        Tree::String(text) => clean(text),
        Tree::Array(items) => items.iter().all(ijson_holds),
        Tree::Object(members) => members.iter().all(|(key, value)| clean(key) && ijson_holds(value)),
    }
}

fn check_profiles(input: &[u8], rfc: &Result<Tree<'_>, Error>, ijson: &Result<Tree<'_>, Error>) {
    match (rfc, ijson) {
        (Ok(tree), Ok(same)) => {
            assert_eq!(tree, same, "the trees of the profiles");
            assert!(ijson_holds(tree), "IJson accepts a value that it refuses");
        }
        (Ok(_), Err(error)) => assert!(ijson_cause(input, *error), "IJson refuses with no cause: {error:?}"),
        (Err(rfc), Err(ijson)) => assert!(
            ijson == rfc || ijson.offset() < rfc.offset() && ijson_cause(input, *ijson),
            "IJson {ijson:?}, Rfc8259 {rfc:?}"
        ),
        (Err(error), Ok(_)) => panic!("IJson accepts what Rfc8259 refuses: {error:?}"),
    }
}

/// A value that a trusted reader reads. A string is bytes, so a string that is not UTF-8 compares
/// too.
#[derive(Debug)]
enum Oracle {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    /// An integer that 64 bits do not hold, as its text.
    Integer(String),
    String(Vec<u8>),
    Array(Vec<Oracle>),
    Object(Vec<(Vec<u8>, Oracle)>),
}

impl From<serde_json::Value> for Oracle {
    fn from(value: serde_json::Value) -> Self {
        use serde_json::Value;
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(number) => number
                .as_u64()
                .map(Self::U64)
                .or_else(|| number.as_i64().map(Self::I64))
                .unwrap_or_else(|| Self::F64(number.as_f64().expect("serde_json holds an f64"))),
            Value::String(text) => Self::String(text.into_bytes()),
            Value::Array(items) => Self::Array(items.into_iter().map(Self::from).collect()),
            Value::Object(members) => {
                Self::Object(members.into_iter().map(|(key, value)| (key.into_bytes(), Self::from(value))).collect())
            }
        }
    }
}

/// The order of the members of an oracle.
#[derive(Clone, Copy)]
enum Order {
    /// The order of the input.
    Kept,
    /// The order of the keys: serde_json holds an object in a `BTreeMap`.
    Sorted,
}

/// Whether `tree` and `oracle` hold the same value. A number compares in the type of the oracle.
fn same(tree: &Tree<'_>, oracle: &Oracle, order: Order) -> bool {
    match (tree, oracle) {
        (Tree::Null, Oracle::Null) => true,
        (Tree::Bool(value), Oracle::Bool(other)) => value == other,
        (Tree::Number(text, _), Oracle::U64(value)) => text.parse::<u64>() == Ok(*value),
        (Tree::Number(text, _), Oracle::I64(value)) => text.parse::<i64>() == Ok(*value),
        (Tree::Number(text, _), Oracle::F64(value)) => text.parse::<f64>().is_ok_and(|own| own.to_bits() == value.to_bits()),
        (Tree::Number(text, _), Oracle::Integer(other)) => text == other,
        (Tree::String(text), Oracle::String(other)) => text.as_bytes() == other,
        (Tree::Array(items), Oracle::Array(others)) => {
            items.len() == others.len() && items.iter().zip(others).all(|(item, other)| same(item, other, order))
        }
        (Tree::Object(members), Oracle::Object(others)) => {
            members.len() == others.len()
                && match order {
                    Order::Kept => members
                        .iter()
                        .zip(others)
                        .all(|((key, value), (other_key, other))| key.as_bytes() == other_key && same(value, other, order)),
                    Order::Sorted => members.iter().all(|(key, value)| {
                        others
                            .iter()
                            .find(|(other_key, _)| key.as_bytes() == other_key)
                            .is_some_and(|(_, other)| same(value, other, order))
                    }),
                }
        }
        _ => false,
    }
}

/// Whether a number of `tree` has a magnitude that rounds to infinity. `Rfc8259` accepts it, and
/// each oracle refuses it.
fn overflows(tree: &Tree<'_>) -> bool {
    match tree {
        Tree::Number(text, _) => text.parse::<f64>().is_ok_and(f64::is_infinite),
        Tree::Array(items) => items.iter().any(overflows),
        Tree::Object(members) => members.iter().any(|(_, value)| overflows(value)),
        Tree::Null | Tree::Bool(_) | Tree::String(_) => false,
    }
}

fn check_oracle(name: &str, order: Order, rfc: &Result<Tree<'_>, Error>, oracle: Result<Oracle, ()>) {
    match (rfc, oracle) {
        (Ok(tree), Ok(oracle)) => assert!(same(tree, &oracle, order), "{name}: {oracle:?}, rscodec: {tree:?}"),
        (Ok(tree), Err(())) => assert!(overflows(tree), "{name} refuses what rscodec accepts: {tree:?}"),
        (Err(error), Ok(oracle)) => assert!(
            matches!(error.kind(), ErrorKind::Duplicate | ErrorKind::Limit(Limit::Depth)),
            "{name} accepts what rscodec refuses: {error:?}, {oracle:?}"
        ),
        (Err(_), Err(())) => {}
    }
}

unsafe extern "C" {
    /// `oracle.cpp`.
    fn simdjson_oracle(input: *const u8, len: usize, tree: *mut *const u8, tree_len: *mut usize) -> i32;
}

/// The tree of simdjson, or its error code.
fn simdjson(input: &[u8]) -> Result<Oracle, i32> {
    let (mut tree, mut len) = (core::ptr::null(), 0);
    // SAFETY: `input` is readable for its length, and the shim writes the two out pointers only.
    let code = unsafe { simdjson_oracle(input.as_ptr(), input.len(), &raw mut tree, &raw mut len) };
    if code != 0 {
        return Err(code);
    }
    // SAFETY: the shim holds `len` bytes at `tree` until its next call.
    let bytes = unsafe { core::slice::from_raw_parts(tree, len) };
    let mut at = 0;
    let oracle = decode(bytes, &mut at);
    assert_eq!(at, bytes.len(), "the shim writes one value");
    Ok(oracle)
}

/// The value at `at` in the form of `oracle.cpp`.
fn decode(bytes: &[u8], at: &mut usize) -> Oracle {
    fn take<'b>(bytes: &'b [u8], at: &mut usize, len: usize) -> &'b [u8] {
        *at += len;
        &bytes[*at - len..*at]
    }
    fn word(bytes: &[u8], at: &mut usize) -> u64 {
        u64::from_le_bytes(take(bytes, at, 8).try_into().expect("8 bytes"))
    }
    fn text(bytes: &[u8], at: &mut usize) -> Vec<u8> {
        let len = u32::from_le_bytes(take(bytes, at, 4).try_into().expect("4 bytes"));
        take(bytes, at, len as usize).to_vec()
    }
    match take(bytes, at, 1)[0] {
        b'n' => Oracle::Null,
        b't' => Oracle::Bool(true),
        b'f' => Oracle::Bool(false),
        b'i' => Oracle::I64(word(bytes, at).cast_signed()),
        b'u' => Oracle::U64(word(bytes, at)),
        b'd' => Oracle::F64(f64::from_bits(word(bytes, at))),
        b'Z' => Oracle::Integer(String::from_utf8(text(bytes, at)).expect("an integer is ASCII")),
        b's' => Oracle::String(text(bytes, at)),
        b'[' => {
            let mut items = Vec::new();
            while bytes[*at] != b']' {
                items.push(decode(bytes, at));
            }
            *at += 1;
            Oracle::Array(items)
        }
        b'{' => {
            let mut members = Vec::new();
            while bytes[*at] != b'}' {
                assert_eq!(take(bytes, at, 1), b"k", "a member starts with its key");
                let key = text(bytes, at);
                members.push((key, decode(bytes, at)));
            }
            *at += 1;
            Oracle::Object(members)
        }
        tag => unreachable!("the shim writes no tag {tag}"),
    }
}
