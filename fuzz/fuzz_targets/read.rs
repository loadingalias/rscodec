//! Compare reader paths, number conversions, profile errors, and two independent parsers.
//! The README specifies the parser differences and the evidence required for each exception.

#![cfg_attr(not(test), no_main)]

use std::collections::{BTreeMap, BTreeSet};

use rscodec::json::{
    self, Error, ErrorKind, Kind, Limit, Limits, Number, NumberClass, Profile, Slot, Str, Token, Value,
};

/// serde_json refuses the 128th container. simdjson does not count an empty container, so it can
/// accept a document that is deeper.
const DEPTH_MAX: u16 = 127;

const LIMITS: Limits = Limits {
    input_bytes_max: u32::MAX,
    depth_max: DEPTH_MAX,
    string_bytes_max: u32::MAX,
    members_max: u32::MAX,
    elements_max: u32::MAX,
    number_digits_max: u16::MAX,
};

#[cfg(not(test))]
libfuzzer_sys::fuzz_target!(|input: &[u8]| check(input));

fn check(input: &[u8]) {
    let (rfc, first_number) = read(input, Profile::Rfc8259);
    let (ijson, _) = read(input, Profile::IJson);
    check_profiles(input, &rfc, first_number, &ijson);
    check_oracle(input, Parser::SerdeJson, &rfc, Parser::SerdeJson.parse(input));
    // simdjson skips a byte order mark, which rscodec refuses. RFC 8259 section 8.1 allows both.
    let simdjson = Parser::Simdjson.parse(input);
    match input.strip_prefix(b"\xef\xbb\xbf") {
        Some(rest) => {
            let error = rfc.as_ref().err().map(|error| (error.kind(), error.offset()));
            assert_eq!(error, Some((ErrorKind::Syntax, 0)), "rscodec refuses a byte order mark");
            check_oracle(rest, Parser::Simdjson, &read(rest, Profile::Rfc8259).0, simdjson);
        }
        None => check_oracle(input, Parser::Simdjson, &rfc, simdjson),
    }
}

#[derive(Debug, PartialEq)]
enum Tree<'a> {
    Null,
    Bool(bool),
    Number(&'a str, usize),
    String(String),
    Array(Vec<Tree<'a>>),
    Object(Vec<(String, Tree<'a>)>),
}

#[derive(Clone, Copy)]
enum Walk {
    Each,
    Even,
}

fn read(input: &[u8], profile: Profile) -> (Result<Tree<'_>, Error>, Option<usize>) {
    let mut scratch = vec![Slot::EMPTY; input.len() + 1];
    let mut first_number = None;
    let tokens = token_tree(input, profile, &mut scratch, &mut first_number);
    let values = json::read(input, &LIMITS, profile, &mut scratch, |reader| {
        value_tree(reader.value(), input, Walk::Each)
    });
    let even = json::read(input, &LIMITS, profile, &mut scratch, |reader| {
        value_tree(reader.value(), input, Walk::Even)
    });
    let skip = json::read(input, &LIMITS, profile, &mut scratch, |reader| reader.value().skip());
    let finish = json::read(input, &LIMITS, profile, &mut scratch, |_| Ok(()));
    let error = tokens.as_ref().err().copied();
    for (path, other) in [
        ("value walk", values.as_ref().err().copied()),
        ("even walk", even.as_ref().err().copied()),
        ("skip", skip.err()),
        ("finish", finish.err()),
    ] {
        assert_eq!(other, error, "{profile:?}: the {path} and the token walk");
    }
    if let (Ok(tokens), Ok(values), Ok(even)) = (&tokens, values, even) {
        assert_eq!(tokens, &values, "{profile:?}: the trees of the walks");
        values.check_even(even);
    }
    (tokens, first_number)
}

impl Tree<'_> {
    fn check_even(mut self, even: Self) {
        self.skip_odd(0);
        assert_eq!(even, self, "the even items and all member keys in input order");
    }

    fn skip_odd(&mut self, index: usize) {
        if index % 2 == 1 {
            *self = Self::Null;
            return;
        }
        match self {
            Self::Array(items) => {
                for (index, item) in items.iter_mut().enumerate() {
                    item.skip_odd(index);
                }
            }
            Self::Object(items) => {
                for (index, (_, item)) in items.iter_mut().enumerate() {
                    item.skip_odd(index);
                }
            }
            _ => {}
        }
    }
}

fn token_tree<'a>(
    input: &'a [u8],
    profile: Profile,
    scratch: &mut [Slot],
    first_number: &mut Option<usize>,
) -> Result<Tree<'a>, Error> {
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
                assert_eq!(
                    matches!(tree, Tree::Object(_)),
                    token == Token::ObjectEnd,
                    "the end of its start"
                );
                key = name;
                tree
            }
            Token::Str(string) => Tree::String(text(string)),
            Token::Number(number) => {
                let text = number.as_str();
                if profile == Profile::Rfc8259 && first_number.is_none() && !binary64_holds(text) {
                    *first_number = Some(text.as_ptr().addr() - input.as_ptr().addr());
                }
                checked(number, input)
            }
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
                tree.push(if read(tree.len()) {
                    value_tree(value, input, walk)?
                } else {
                    Tree::Null
                });
            }
            Tree::Array(tree)
        }
        Kind::String => Tree::String(text(value.str()?)),
        Kind::Number => checked(value.number()?, input),
        Kind::Bool => Tree::Bool(value.bool()?),
        Kind::Null => value.null().map(|()| Tree::Null)?,
    })
}

fn text(string: Str<'_>) -> String {
    let mut buf = vec![0; string.len()];
    let decoded = string
        .decode(&mut buf)
        .expect("a buffer of `len` bytes holds the string")
        .to_owned();
    assert!(string == decoded.as_str(), "`==` compares after unescape: {decoded:?}");
    decoded
}

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
        Ok(value) => assert!(
            float.is_finite() && value.to_bits() == float.to_bits(),
            "{text}: {value:?}"
        ),
        Err(error) => assert!(
            float.is_infinite() && error.kind() == ErrorKind::Range,
            "{text}: {error:?}"
        ),
    }
    if class == NumberClass::Integer {
        let unsigned = if text == "-0" { Ok(0) } else { text.parse::<u64>() };
        assert_eq!(
            kind(number.to_i64()),
            text.parse::<i64>().map_err(|_| ErrorKind::Range),
            "{text}"
        );
        assert_eq!(kind(number.to_u64()), unsigned.map_err(|_| ErrorKind::Range), "{text}");
    } else {
        assert_eq!(kind(number.to_i64()), Err(ErrorKind::Type), "{text}");
        assert_eq!(kind(number.to_u64()), Err(ErrorKind::Type), "{text}");
    }
    let decimal = number
        .to_decimal()
        .map(|decimal| (decimal.negative, decimal.coefficient, decimal.exponent));
    assert_eq!(kind(decimal), decimal_of(text).ok_or(ErrorKind::Range), "{text}");
    Tree::Number(text, text.as_ptr().addr() - input.as_ptr().addr())
}

fn kind<T>(result: Result<T, Error>) -> Result<T, ErrorKind> {
    result.map_err(|error| error.kind())
}

fn decimal_of(text: &str) -> Option<(bool, u128, i32)> {
    let (negative, text) = text.strip_prefix('-').map_or((false, text), |rest| (true, rest));
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{integer}{fraction}");
    let significant = digits.trim_start_matches('0');
    if significant.len() > 38 {
        return None;
    }
    let coefficient = if significant.is_empty() {
        0
    } else {
        significant.parse().ok()?
    };
    let exponent = exponent
        .parse::<i64>()
        .ok()?
        .checked_sub(i64::try_from(fraction.len()).ok()?)?;
    Some((negative, coefficient, i32::try_from(exponent).ok()?))
}

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

fn noncharacter_at(input: &[u8], at: usize) -> bool {
    let hex = |from: usize| {
        let digits = input.get(from..from + 4)?;
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u32::from_str_radix(core::str::from_utf8(digits).ok()?, 16).ok()
    };
    let value = if input.get(at) == Some(&b'\\') {
        if input.get(at + 1) != Some(&b'u') {
            return false;
        }
        match hex(at + 2) {
            Some(high @ 0xd800..0xdc00) => match hex(at + 8) {
                Some(low @ 0xdc00..0xe000) if input.get(at + 6..at + 8) == Some(br"\u") => {
                    0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                }
                _ => return false,
            },
            Some(value) => value,
            None => return false,
        }
    } else {
        let tail = &input[at..input.len().min(at + 4)];
        let valid = core::str::from_utf8(tail)
            .unwrap_or_else(|error| core::str::from_utf8(&tail[..error.valid_up_to()]).expect("the prefix is UTF-8"));
        match valid.chars().next() {
            Some(character) => u32::from(character),
            None => return false,
        }
    };
    noncharacter(value)
}

fn lexemes(input: &[u8]) -> impl Iterator<Item = (usize, &[u8])> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        while input
            .get(pos)
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        {
            pos += 1;
        }
        let start = pos;
        let byte = *input.get(pos)?;
        pos += 1;
        match byte {
            b'"' => {
                while let Some(&next) = input.get(pos) {
                    pos += 1;
                    match next {
                        b'"' => break,
                        b'\\' if pos < input.len() => pos += 1,
                        _ => {}
                    }
                }
            }
            b'[' | b']' | b'{' | b'}' | b',' | b':' => {}
            _ => {
                while input.get(pos).is_some_and(|byte| {
                    !matches!(
                        byte,
                        b' ' | b'\t' | b'\n' | b'\r' | b'"' | b'[' | b']' | b'{' | b'}' | b',' | b':'
                    )
                }) {
                    pos += 1;
                }
            }
        }
        Some((start, &input[start..pos]))
    })
}

fn first_noncharacter(input: &[u8]) -> Option<usize> {
    for (start, raw) in lexemes(input).filter(|(_, raw)| raw[0] == b'"') {
        let mut at = 1;
        while let Some(&byte) = raw.get(at) {
            if (byte == b'\\' || byte >= 0xef) && noncharacter_at(raw, at) {
                return Some(start + at);
            }
            at += if byte == b'\\' { 2 } else { 1 };
        }
    }
    None
}

fn check_profiles(
    input: &[u8],
    rfc: &Result<Tree<'_>, Error>,
    first_number: Option<usize>,
    ijson: &Result<Tree<'_>, Error>,
) {
    let refusal = |e: &Error| (e.kind(), e.offset());
    let rfc_error = rfc.as_ref().err().map(refusal);
    let end = rfc_error.map_or(input.len(), |(_, at)| at);
    let number = first_number.filter(|&at| at < end).map(|at| (ErrorKind::Number, at));
    let character = first_noncharacter(&input[..end]).map(|at| (ErrorKind::Noncharacter, at));
    let expected = [number, character, rfc_error]
        .into_iter()
        .flatten()
        .min_by_key(|&(_, at)| at);
    assert_eq!(
        ijson.as_ref().err().map(refusal),
        expected,
        "the first IJson error and byte offset"
    );
    if expected.is_none() {
        assert_eq!(ijson, rfc, "the trees of the profiles");
    }
}

#[derive(Debug)]
enum Oracle {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
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
            Value::Object(members) => Self::Object(
                members
                    .into_iter()
                    .map(|(key, value)| (key.into_bytes(), Self::from(value)))
                    .collect(),
            ),
        }
    }
}

/// serde_json sorts object members. The simdjson shim keeps their order and duplicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parser {
    SerdeJson,
    Simdjson,
}

impl Parser {
    fn parse(self, input: &[u8]) -> Result<Oracle, ()> {
        match self {
            Self::SerdeJson => serde_json::from_slice::<serde_json::Value>(input)
                .map(Oracle::from)
                .map_err(drop),
            Self::Simdjson => simdjson(input).map_err(drop),
        }
    }
}

fn same(tree: &Tree<'_>, oracle: &Oracle, parser: Parser, replaced: &mut BTreeMap<usize, &str>) -> bool {
    if let Tree::Number(text, at) = tree {
        let text = match replaced.remove(at) {
            Some(raw) if raw == *text => "0",
            Some(_) => return false,
            None => *text,
        };
        return match oracle {
            Oracle::U64(value) => text.parse::<u64>() == Ok(*value),
            Oracle::I64(value) => text.parse::<i64>() == Ok(*value),
            Oracle::F64(value) => text.parse::<f64>().is_ok_and(|own| own.to_bits() == value.to_bits()),
            Oracle::Integer(other) => text == other,
            _ => false,
        };
    }
    match (tree, oracle) {
        (Tree::Null, Oracle::Null) => true,
        (Tree::Bool(value), Oracle::Bool(other)) => value == other,
        (Tree::String(text), Oracle::String(other)) => text.as_bytes() == other,
        (Tree::Array(items), Oracle::Array(others)) => {
            items.len() == others.len()
                && items
                    .iter()
                    .zip(others)
                    .all(|(item, other)| same(item, other, parser, replaced))
        }
        (Tree::Object(members), Oracle::Object(others)) => {
            members.len() == others.len()
                && match parser {
                    Parser::Simdjson => members.iter().zip(others).all(|((key, value), (other_key, other))| {
                        key.as_bytes() == other_key && same(value, other, parser, replaced)
                    }),
                    Parser::SerdeJson => others.iter().all(|(other_key, other)| {
                        members
                            .iter()
                            .find(|(key, _)| key.as_bytes() == other_key)
                            .is_some_and(|(_, value)| same(value, other, parser, replaced))
                    }),
                }
        }
        _ => false,
    }
}

fn oracle_exception(input: &[u8], parser: Parser) -> Option<(ErrorKind, usize)> {
    let mut keys = Vec::<BTreeSet<String>>::new();
    let mut tokens = lexemes(input).peekable();
    while let Some((at, raw)) = tokens.next() {
        match raw[0] {
            open @ (b'[' | b'{') => {
                if keys.len() == usize::from(DEPTH_MAX) {
                    let close = if open == b'[' { b']' } else { b'}' };
                    let empty = tokens.peek().is_some_and(|(_, next)| next == &[close]);
                    return (parser == Parser::Simdjson && empty).then_some((ErrorKind::Limit(Limit::Depth), at));
                }
                keys.push(BTreeSet::new());
            }
            b']' | b'}' => {
                keys.pop()?;
            }
            b'"' if tokens.peek().is_some_and(|(_, next)| *next == b":") => {
                let key = serde_json::from_slice::<String>(raw).ok()?;
                if !keys.last_mut()?.insert(key) {
                    return Some((ErrorKind::Duplicate, at));
                }
            }
            _ => {}
        }
    }
    None
}

/// Retry only with complete, overflowing JSON numbers replaced by zero. Spaces preserve byte offsets.
fn finite_oracle<'a>(input: &'a [u8], parser: Parser, replaced: &mut BTreeMap<usize, &'a str>) -> Result<Oracle, ()> {
    let mut finite = None;
    for (at, raw) in lexemes(input) {
        if matches!(raw[0], b'-' | b'0'..=b'9')
            && let Ok(text) = core::str::from_utf8(raw)
            && text.parse::<f64>().is_ok_and(f64::is_infinite)
            && serde_json::from_slice::<&serde_json::value::RawValue>(raw).is_ok()
        {
            let copy = finite.get_or_insert_with(|| input.to_vec());
            copy[at] = b'0';
            copy[at + 1..at + raw.len()].fill(b' ');
            replaced.insert(at, text);
        }
    }
    parser.parse(finite.as_deref().ok_or(())?)
}

fn check_oracle(input: &[u8], parser: Parser, rfc: &Result<Tree<'_>, Error>, oracle: Result<Oracle, ()>) {
    let mut replaced = BTreeMap::new();
    let oracle = oracle.or_else(|()| finite_oracle(input, parser, &mut replaced));
    match (rfc, oracle) {
        (Ok(tree), Ok(oracle)) => assert!(
            same(tree, &oracle, parser, &mut replaced) && replaced.is_empty(),
            "{parser:?}: {oracle:?}, rscodec: {tree:?}"
        ),
        (Ok(tree), Err(())) => panic!("{parser:?} refuses the document: {tree:?}"),
        (Err(error), Ok(_)) => assert_eq!(
            oracle_exception(input, parser),
            Some((error.kind(), error.offset())),
            "{parser:?} accepts the document"
        ),
        (Err(_), Err(())) => {}
    }
}

unsafe extern "C" {
    fn simdjson_oracle(input: *const u8, len: usize, tree: *mut *const u8, tree_len: *mut usize) -> i32;
}

fn simdjson(input: &[u8]) -> Result<Oracle, i32> {
    let (mut tree, mut len) = (core::ptr::null(), 0);
    // SAFETY: `input` is readable for its length, and the shim writes the two out pointers only.
    let code = unsafe { simdjson_oracle(input.as_ptr(), input.len(), &raw mut tree, &raw mut len) };
    if code != 0 {
        return Err(code);
    }
    // SAFETY: the shim holds `len` bytes at `tree` until the next call on this thread.
    let bytes = unsafe { core::slice::from_raw_parts(tree, len) };
    let mut at = 0;
    let oracle = decode(bytes, &mut at);
    assert_eq!(at, bytes.len(), "the shim writes one value");
    Ok(oracle)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;

    #[test]
    fn oracle_acceptance_requires_the_actual_duplicate_key_or_empty_container_at_the_depth_limit() {
        let duplicate = br#"{"a":0,"\u0061":1}"#;
        let rfc = read(duplicate, Profile::Rfc8259).0;
        let shallow = br#"{"a":0}"#;
        let deep = format!("{} {} {}", "[".repeat(127), "{}", "]".repeat(127));
        let depth = Err(json::visit(
            deep.as_bytes(),
            &LIMITS,
            Profile::Rfc8259,
            &mut [Slot::EMPTY; 128],
            |_| Ok(()),
        )
        .unwrap_err());
        for parser in [Parser::SerdeJson, Parser::Simdjson] {
            assert_eq!(oracle_exception(duplicate, parser), Some((ErrorKind::Duplicate, 7)));
            for kind in [ErrorKind::Duplicate, ErrorKind::Limit(Limit::Depth)] {
                let false_error = Err(Error::new(kind, 1));
                assert!(catch_unwind(|| check_oracle(shallow, parser, &false_error, parser.parse(shallow))).is_err());
            }
            assert_eq!(oracle_exception(br#"{"a":0,"nested":{"a":1}}"#, parser), None);
            check_oracle(duplicate, parser, &rfc, parser.parse(duplicate));
        }
        assert_eq!(oracle_exception(deep.as_bytes(), Parser::SerdeJson), None);
        assert_eq!(
            oracle_exception(deep.as_bytes(), Parser::Simdjson),
            Some((ErrorKind::Limit(Limit::Depth), 128))
        );
        assert_eq!(
            oracle_exception(deep.replace("{}", "[0]").as_bytes(), Parser::Simdjson),
            None
        );
        check_oracle(
            deep.as_bytes(),
            Parser::Simdjson,
            &depth,
            Parser::Simdjson.parse(deep.as_bytes()),
        );
    }

    #[test]
    fn an_overflow_preserves_the_grammar_and_each_other_value() {
        let valid = b"[1e400,true]";
        let rfc = read(valid, Profile::Rfc8259).0;
        for parser in [Parser::SerdeJson, Parser::Simdjson] {
            check_oracle(valid, parser, &rfc, parser.parse(valid));
            for input in [
                b"[1e400,]".as_slice(),
                b"[1e400,false]",
                b"[01e400,true]",
                br#"["1e400",true]"#,
            ] {
                assert!(catch_unwind(|| check_oracle(input, parser, &rfc, parser.parse(input))).is_err());
            }
        }
        check(br#"{"a":[1e400,-1e401],"b":"1e400"}"#);
    }

    #[test]
    fn ijson_reports_the_first_disallowed_token_and_its_byte_offset() {
        let prefix = format!("[{}", "0,".repeat(2048));
        let dense = format!("{prefix}1e400,1e400]");
        for (input, wrong) in [
            (b"null".as_slice(), 0),
            (b"[1e400,1e400]", 7),
            (dense.as_bytes(), prefix.len() + 6),
        ] {
            let (rfc, number) = read(input, Profile::Rfc8259);
            let error = Err(Error::new(ErrorKind::Number, wrong));
            assert!(catch_unwind(|| check_profiles(input, &rfc, number, &error)).is_err());
            check(input);
        }
        for input in [
            br#"["\\uFFFF",1e400,"\uFFFF"]"#.as_slice(),
            br#"{"\uDBFF\uDFFF":1e400}"#,
            br#"["\uFFFF\q",1e400]"#,
            "[\"\u{fdd0}\",1e400]".as_bytes(),
            br#"[1e400,]"#,
        ] {
            check(input);
        }
    }

    #[test]
    fn an_alternating_skip_preserves_keys_and_the_following_sibling_at_each_depth() {
        let input = br#"{"a":[true,{"drop":1},false],"b":[2,3],"c":{"x":null,"y":4,"z":true}}"#;
        check(input);
        let missing = br#"{"a":[true,null],"b":null,"c":{"x":null,"y":null,"z":true}}"#;
        let reordered = br#"{"a":[true,null,false],"b":null,"c":{"y":null,"x":null,"z":true}}"#;
        for wrong in [missing.as_slice(), reordered] {
            let full = read(input, Profile::Rfc8259).0.unwrap();
            let actual = read(wrong, Profile::Rfc8259).0.unwrap();
            assert!(catch_unwind(|| full.check_even(actual)).is_err());
        }
        let tree = json::read(input, &LIMITS, Profile::Rfc8259, &mut [Slot::EMPTY; 16], |reader| {
            value_tree(reader.value(), input, Walk::Even)
        })
        .unwrap();
        assert_eq!(
            tree,
            Tree::Object(vec![
                (
                    "a".into(),
                    Tree::Array(vec![Tree::Bool(true), Tree::Null, Tree::Bool(false)])
                ),
                ("b".into(), Tree::Null),
                (
                    "c".into(),
                    Tree::Object(vec![
                        ("x".into(), Tree::Null),
                        ("y".into(), Tree::Null),
                        ("z".into(), Tree::Bool(true))
                    ])
                ),
            ])
        );
    }
}
