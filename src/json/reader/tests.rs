//! The contracts of the reader. Each test reads with each kernel of this CPU, and each kernel must
//! give the same result. The expected results come from the standards, from `core`, or from
//! serde_json, not from the reader.

extern crate std;

use std::{
    borrow::ToOwned,
    format, fs,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use super::super::{
    Backend, Error, ErrorKind, Kind, Limit, Limits, Profile, Reader, Slot, Token, read_with, tests::kernels, visit_with,
};

/// Limits that no test document reaches, unless a test sets one.
const WIDE: Limits = Limits {
    input_bytes_max: u32::MAX,
    depth_max: 1024,
    string_bytes_max: u32::MAX,
    members_max: u32::MAX,
    elements_max: u32::MAX,
    number_digits_max: 1024,
};

const PROFILES: [Profile; 2] = [Profile::Rfc8259, Profile::IJson];

/// A token with its strings decoded.
#[derive(Clone, Debug, PartialEq)]
enum Owned {
    ObjectStart,
    ObjectEnd,
    ArrayStart,
    ArrayEnd,
    Key(String),
    Str(String),
    Number(String),
    Bool(bool),
    Null,
}

fn decoded(text: super::Str<'_>) -> String {
    let mut buf = vec![0; text.len()];
    let decoded = text
        .decode(&mut buf)
        .expect("a buffer of `len` bytes holds the string")
        .to_owned();
    assert!(text == decoded.as_str(), "a string equals its decoded text");
    decoded
}

fn owned(token: Token<'_>) -> Owned {
    match token {
        Token::ObjectStart => Owned::ObjectStart,
        Token::ObjectEnd => Owned::ObjectEnd,
        Token::ArrayStart => Owned::ArrayStart,
        Token::ArrayEnd => Owned::ArrayEnd,
        Token::Key(key) => Owned::Key(decoded(key)),
        Token::Str(text) => Owned::Str(decoded(text)),
        Token::Number(number) => Owned::Number(number.as_str().to_owned()),
        Token::Bool(value) => Owned::Bool(value),
        Token::Null => Owned::Null,
    }
}

/// The result of `run` with each kernel, on `input`, which each kernel must give.
fn each_kernel<T: PartialEq + core::fmt::Debug>(
    input: &[u8],
    profile: Profile,
    mut run: impl FnMut(Backend) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut results = kernels().into_iter().map(|kernel| (kernel, run(kernel)));
    let (_, first) = results.next().expect("each CPU runs the reference");
    for (kernel, result) in results {
        assert_eq!(
            result,
            first,
            "{kernel:?}, profile {profile:?}, input {:?}",
            String::from_utf8_lossy(input)
        );
    }
    first
}

/// The result of `decode` on `input`, which each kernel must give.
fn read_each<T: PartialEq + core::fmt::Debug>(
    input: &[u8],
    limits: &Limits,
    profile: Profile,
    slots: usize,
    mut decode: impl FnMut(&mut Reader<'_, '_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut scratch = vec![Slot::EMPTY; slots];
    each_kernel(input, profile, |kernel| {
        read_with(kernel, input, limits, profile, &mut scratch, &mut decode)
    })
}

/// Each token of `input`, which each kernel must give.
fn tokens(input: &[u8], limits: &Limits, profile: Profile, slots: usize) -> Result<Vec<Owned>, Error> {
    let mut scratch = vec![Slot::EMPTY; slots];
    each_kernel(input, profile, |kernel| {
        let mut tokens = Vec::new();
        visit_with(kernel, input, limits, profile, &mut scratch, |token| {
            tokens.push(owned(token));
            Ok(())
        })
        .map(|()| tokens)
    })
}

/// The result of reading `input` with its tokens, without them.
fn check(input: &[u8], limits: &Limits, profile: Profile, slots: usize) -> Result<(), Error> {
    tokens(input, limits, profile, slots).map(drop)
}

fn error(kind: ErrorKind, offset: usize) -> Result<(), Error> {
    Err(Error::new(kind, offset))
}

/// The serde_json tree of the tokens of `input`. A number is the serde_json value of its text.
pub(in crate::json) fn tree(input: &[u8], profile: Profile) -> Result<serde_json::Value, Error> {
    use serde_json::{Map, Value};
    // Each container and each key takes one slot and at least one byte, so a slot for each byte
    // is enough. A scratch of 1 MiB for each short string cost new pages on Windows at each call.
    let tokens = tokens(input, &WIDE, profile, input.len().min(1 << 16) + 1)?;
    let mut open: Vec<(Value, Option<String>)> = Vec::new();
    let mut key = None;
    let mut root = None;
    for token in tokens {
        let value = match token {
            Owned::ObjectStart => {
                open.push((Value::Object(Map::new()), key.take()));
                continue;
            }
            Owned::ArrayStart => {
                open.push((Value::Array(Vec::new()), key.take()));
                continue;
            }
            Owned::Key(name) => {
                key = Some(name);
                continue;
            }
            Owned::ObjectEnd | Owned::ArrayEnd => {
                let (value, name) = open.pop().expect("each end has its start");
                key = name;
                value
            }
            Owned::Str(text) => Value::String(text),
            Owned::Number(text) => serde_json::from_str(&text).expect("serde_json reads each number of the reader"),
            Owned::Bool(value) => Value::Bool(value),
            Owned::Null => Value::Null,
        };
        match open.last_mut() {
            Some((Value::Object(members), _)) => {
                members.insert(key.take().expect("each member has a key"), value);
            }
            Some((Value::Array(elements), _)) => elements.push(value),
            _ => root = Some(value),
        }
    }
    Ok(root.expect("a document has a value"))
}

fn jsontestsuite() -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<_> = fs::read_dir("tests/data/JSONTestSuite/test_parsing")
        .expect("tests/data holds JSONTestSuite")
        .map(|entry| {
            let path = entry.expect("a directory entry is readable").path();
            let name = path
                .file_name()
                .expect("a file has a name")
                .to_string_lossy()
                .into_owned();
            (name, fs::read(&path).expect("a test file is readable"))
        })
        .collect();
    files.sort();
    files
}

/// Each `y_` file accepts and each `n_` file refuses, in each profile, with three exceptions that
/// the design decides. Each profile refuses a duplicate key ([CODEC Decisions] of 2026-11-17,
/// item 2), so the two `y_` files with one refuse. `IJson` refuses the `y_` files with a
/// noncharacter. Each `i_` file gives the result in the table, which the user docs list.
#[test]
fn each_file_of_jsontestsuite_gives_its_decided_result() {
    use ErrorKind::{Duplicate, Noncharacter, Number, Surrogate, Syntax, Utf8};
    // The `y_` files that refuse, with the profiles that refuse them.
    let refused: [(&str, &[Profile], ErrorKind, usize); 10] = [
        ("y_object_duplicated_key.json", &PROFILES, Duplicate, 9),
        ("y_object_duplicated_key_and_value.json", &PROFILES, Duplicate, 9),
        ("y_string_escaped_noncharacter.json", &[Profile::IJson], Noncharacter, 2),
        (
            "y_string_last_surrogates_1_and_2.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_nonCharacterInUTF-8_U+10FFFF.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_nonCharacterInUTF-8_U+FFFF.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_unicode_U+10FFFE_nonchar.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_unicode_U+1FFFE_nonchar.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_unicode_U+FDD0_nonchar.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
        (
            "y_string_unicode_U+FFFE_nonchar.json",
            &[Profile::IJson],
            Noncharacter,
            2,
        ),
    ];
    // Each `i_` file: `None` accepts, and `Some` refuses at the offset, in `Rfc8259` and in `IJson`.
    type Refusal = Option<(ErrorKind, usize)>;
    let implementation_defined: [(&str, Refusal, Refusal); 35] = [
        ("i_number_double_huge_neg_exp.json", None, Some((Number, 1))),
        ("i_number_huge_exp.json", None, Some((Number, 1))),
        ("i_number_neg_int_huge_exp.json", None, Some((Number, 1))),
        ("i_number_pos_double_huge_exp.json", None, Some((Number, 1))),
        ("i_number_real_neg_overflow.json", None, Some((Number, 1))),
        ("i_number_real_pos_overflow.json", None, Some((Number, 1))),
        ("i_number_real_underflow.json", None, Some((Number, 1))),
        ("i_number_too_big_neg_int.json", None, Some((Number, 1))),
        ("i_number_too_big_pos_int.json", None, Some((Number, 1))),
        ("i_number_very_big_negative_int.json", None, Some((Number, 1))),
        (
            "i_object_key_lone_2nd_surrogate.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_1st_surrogate_but_2nd_missing.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_1st_valid_surrogate_2nd_invalid.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        ("i_string_UTF-16LE_with_BOM.json", Some((Utf8, 0)), Some((Utf8, 0))),
        ("i_string_UTF-8_invalid_sequence.json", Some((Utf8, 7)), Some((Utf8, 7))),
        ("i_string_UTF8_surrogate_U+D800.json", Some((Utf8, 2)), Some((Utf8, 2))),
        (
            "i_string_incomplete_surrogate_and_escape_valid.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_incomplete_surrogate_pair.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_incomplete_surrogates_escape_valid.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_invalid_lonely_surrogate.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_invalid_surrogate.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        ("i_string_invalid_utf-8.json", Some((Utf8, 2)), Some((Utf8, 2))),
        (
            "i_string_inverted_surrogates_U+1D11E.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        ("i_string_iso_latin_1.json", Some((Utf8, 2)), Some((Utf8, 2))),
        (
            "i_string_lone_second_surrogate.json",
            Some((Surrogate, 2)),
            Some((Surrogate, 2)),
        ),
        (
            "i_string_lone_utf8_continuation_byte.json",
            Some((Utf8, 2)),
            Some((Utf8, 2)),
        ),
        ("i_string_not_in_unicode_range.json", Some((Utf8, 2)), Some((Utf8, 2))),
        (
            "i_string_overlong_sequence_2_bytes.json",
            Some((Utf8, 2)),
            Some((Utf8, 2)),
        ),
        (
            "i_string_overlong_sequence_6_bytes.json",
            Some((Utf8, 2)),
            Some((Utf8, 2)),
        ),
        (
            "i_string_overlong_sequence_6_bytes_null.json",
            Some((Utf8, 2)),
            Some((Utf8, 2)),
        ),
        ("i_string_truncated-utf-8.json", Some((Utf8, 2)), Some((Utf8, 2))),
        ("i_string_utf16BE_no_BOM.json", Some((Syntax, 0)), Some((Syntax, 0))),
        ("i_string_utf16LE_no_BOM.json", Some((Syntax, 1)), Some((Syntax, 1))),
        // Accepted with a depth limit of 500 or more.
        ("i_structure_500_nested_arrays.json", None, None),
        // A byte order mark refuses (RFC 8259 section 8.1 lets a reader refuse or ignore it).
        (
            "i_structure_UTF-8_BOM_empty_object.json",
            Some((Syntax, 0)),
            Some((Syntax, 0)),
        ),
    ];
    let files = jsontestsuite();
    let count = |prefix: &str| files.iter().filter(|(name, _)| name.starts_with(prefix)).count();
    assert_eq!(
        (count("y_"), count("n_"), count("i_")),
        (95, 188, 35),
        "the files of the pinned commit"
    );
    for (name, text) in &files {
        for profile in PROFILES {
            let result = check(text, &WIDE, profile, 2048);
            let expected = if let Some(&(_, profiles, kind, offset)) = refused.iter().find(|file| file.0 == name) {
                profiles.contains(&profile).then_some((kind, offset))
            } else if let Some(&(_, rfc8259, ijson)) = implementation_defined.iter().find(|file| file.0 == name) {
                if profile == Profile::Rfc8259 { rfc8259 } else { ijson }
            } else if name.starts_with("y_") {
                None
            } else {
                assert!(name.starts_with("n_"), "{name} is in no table");
                assert!(result.is_err(), "{name} refuses in {profile:?}");
                continue;
            };
            let expected = expected.map_or(Ok(()), |(kind, offset)| error(kind, offset));
            assert_eq!(result, expected, "{name} in {profile:?}");
        }
    }
    let (_, nested) = files
        .iter()
        .find(|(name, _)| name == "i_structure_500_nested_arrays.json")
        .expect("the file of 500 arrays");
    let limits = |depth_max| Limits { depth_max, ..WIDE };
    assert_eq!(check(nested, &limits(500), Profile::Rfc8259, 500), Ok(()));
    assert_eq!(
        check(nested, &limits(499), Profile::Rfc8259, 500),
        error(ErrorKind::Limit(Limit::Depth), 499)
    );
}

/// Each line of Markus Kuhn's UTF-8 test file, in a string: `Rfc8259` accepts it exactly when
/// `core::str::from_utf8` does, with the same offset, and `IJson` also refuses each noncharacter.
/// The whole file in one string gives the first error of the file.
#[test]
fn each_line_of_kuhns_file_gives_the_result_of_its_profile() {
    let text = fs::read("tests/data/kuhn/UTF-8-test.txt").expect("tests/data holds Kuhn's file");
    // A quote, a `\`, and a byte under 0x20 become ASCII that a string holds, so each line is one
    // string with the same UTF-8.
    let plain: Vec<u8> = text
        .iter()
        .map(|&byte| match byte {
            b'"' => b'\'',
            b'\\' => b'/',
            0..0x20 if byte != b'\n' => b' ',
            _ => byte,
        })
        .collect();
    let whole: Vec<u8> = plain
        .iter()
        .map(|&byte| if byte == b'\n' { b' ' } else { byte })
        .collect();
    let mut lines: Vec<&[u8]> = plain.split(|&byte| byte == b'\n').collect();
    lines.push(&whole);
    let (mut invalid, mut noncharacters) = (0, 0);
    for line in lines {
        let document = [b"\"", line, b"\""].concat();
        for profile in PROFILES {
            let expected = match core::str::from_utf8(line) {
                Err(utf8) => error(ErrorKind::Utf8, 1 + utf8.valid_up_to()),
                Ok(text) => match text.char_indices().find(|&(_, char)| {
                    let value = u32::from(char);
                    matches!(value, 0xfdd0..=0xfdef) || value & 0xfffe == 0xfffe
                }) {
                    Some((at, _)) if profile == Profile::IJson => error(ErrorKind::Noncharacter, 1 + at),
                    _ => Ok(()),
                },
            };
            invalid += usize::from(core::str::from_utf8(line).is_err());
            noncharacters += usize::from(expected.is_err_and(|error| error.kind() == ErrorKind::Noncharacter));
            assert_eq!(
                check(&document, &WIDE, profile, 1),
                expected,
                "{profile:?}: {}",
                String::from_utf8_lossy(line)
            );
        }
    }
    // The file has lines that are not UTF-8, and lines with U+FFFE and U+FFFF.
    assert!(invalid > 50 && noncharacters >= 2, "{invalid} {noncharacters}");
}

/// Each pair of `\u` escapes of the surrogates U+D800 to U+DFFF: a high and then a low surrogate is
/// one character, which `char::decode_utf16` gives. Each other pair refuses at its first lone
/// surrogate, in each profile. `IJson` also refuses the noncharacters of the planes 1 to 16.
#[test]
fn each_surrogate_escape_pair_gives_its_character_or_refuses() {
    let kernels = kernels();
    let mut scratch = [Slot::EMPTY; 1];
    let mut buf = [0; 4];
    for first in 0xd800..=0xdfff_u16 {
        for second in 0xd800..=0xdfff_u16 {
            let document = format!("\"\\u{first:04x}\\u{second:04X}\"");
            let char = char::decode_utf16([first, second]).next().and_then(Result::ok);
            for profile in PROFILES {
                let expected = match char {
                    Some(char) if profile == Profile::IJson && u32::from(char) & 0xfffe == 0xfffe => {
                        Err(Error::new(ErrorKind::Noncharacter, 1))
                    }
                    Some(char) => Ok(Some(char)),
                    // A low surrogate first is lone, and so is a high surrogate without a low one.
                    None => Err(Error::new(ErrorKind::Surrogate, 1)),
                };
                for &kernel in &kernels {
                    let read = read_with(kernel, document.as_bytes(), &WIDE, profile, &mut scratch, |reader| {
                        let text = reader.value().str()?;
                        let decoded = text.decode(&mut buf)?;
                        assert!(text == decoded);
                        Ok(decoded.chars().next())
                    });
                    assert_eq!(read, expected, "{kernel:?} {profile:?} {document}");
                }
            }
        }
    }
}

/// The 66 noncharacters of Unicode, as UTF-8 and as escapes: `Rfc8259` accepts each, and `IJson`
/// refuses each at its offset. The characters next to them are no noncharacters.
#[test]
fn each_noncharacter_refuses_only_in_ijson() {
    let noncharacters: Vec<char> = (0xfdd0..=0xfdef)
        .chain((0..=16).flat_map(|plane| [plane << 16 | 0xfffe, plane << 16 | 0xffff]))
        .filter_map(char::from_u32)
        .collect();
    assert_eq!(noncharacters.len(), 66);
    let neighbours = [
        '\u{fdcf}',
        '\u{fdf0}',
        '\u{fffd}',
        '\u{1fffd}',
        '\u{10fffd}',
        '\u{e000}',
    ];
    let escaped = |char: char| {
        let mut units = [0; 2];
        char.encode_utf16(&mut units)
            .iter()
            .map(|unit| format!("\\u{unit:04x}"))
            .collect::<String>()
    };
    for (char, refuses) in noncharacters
        .iter()
        .map(|&char| (char, true))
        .chain(neighbours.map(|char| (char, false)))
    {
        for document in [format!("[\"ab{char}\"]"), format!("[\"ab{}\"]", escaped(char))] {
            for profile in PROFILES {
                let expected = if refuses && profile == Profile::IJson {
                    Err(Error::new(ErrorKind::Noncharacter, 4))
                } else {
                    Ok(vec![
                        Owned::ArrayStart,
                        Owned::Str(format!("ab{char}")),
                        Owned::ArrayEnd,
                    ])
                };
                assert_eq!(
                    tokens(document.as_bytes(), &WIDE, profile, 1),
                    expected,
                    "{profile:?} {document}"
                );
            }
        }
    }
}

/// The hostile set of `docs/research/json.md` (profile A6) gives its typed errors. A document of
/// depth 60,000 reads on a thread with a stack of 64 KiB, so the stack does not grow with depth.
#[test]
fn the_hostile_set_gives_its_typed_errors() {
    let tight = Limits {
        input_bytes_max: 1 << 22,
        depth_max: 32,
        string_bytes_max: 1 << 16,
        members_max: 1024,
        elements_max: 1024,
        number_digits_max: 64,
    };
    let deep = "[".repeat(10_000) + &"]".repeat(10_000);
    let mut members = String::from("{");
    for index in 0..100_000 {
        members += &format!("\"k{index}\":0,");
    }
    members.pop();
    members.push('}');
    let digits = "1".repeat(4301);
    let long = format!("\"{}\"", "a".repeat(1 << 20));
    let cases: [(&str, &[u8], Profile, ErrorKind, usize); 13] = [
        (
            "depth 10,000",
            deep.as_bytes(),
            Profile::Rfc8259,
            ErrorKind::Limit(Limit::Depth),
            32,
        ),
        (
            "a duplicate",
            br#"{"alg":"none","alg":"ES256"}"#,
            Profile::Rfc8259,
            ErrorKind::Duplicate,
            14,
        ),
        (
            "an escaped duplicate",
            br#"{"alg":"none","\u0061lg":"ES256"}"#,
            Profile::Rfc8259,
            ErrorKind::Duplicate,
            14,
        ),
        (
            "a lone surrogate",
            br#"{"kid":"\ud800"}"#,
            Profile::Rfc8259,
            ErrorKind::Surrogate,
            8,
        ),
        (
            "a noncharacter",
            "{\"kid\":\"\u{fffe}\"}".as_bytes(),
            Profile::IJson,
            ErrorKind::Noncharacter,
            8,
        ),
        ("1e400", b"[1e400]", Profile::IJson, ErrorKind::Number, 1),
        (
            "4,301 digits",
            digits.as_bytes(),
            Profile::Rfc8259,
            ErrorKind::Limit(Limit::NumberDigits),
            64,
        ),
        (
            "a string of 1 MiB",
            long.as_bytes(),
            Profile::Rfc8259,
            ErrorKind::Limit(Limit::StringBytes),
            1 + (1 << 16),
        ),
        (
            "100,000 members",
            members.as_bytes(),
            Profile::Rfc8259,
            ErrorKind::Limit(Limit::Members),
            members.match_indices('"').nth(2048).map_or(0, |(at, _)| at),
        ),
        ("trailing data", b"{} {}", Profile::Rfc8259, ErrorKind::Syntax, 3),
        (
            "a byte order mark",
            b"\xef\xbb\xbf{}",
            Profile::Rfc8259,
            ErrorKind::Syntax,
            0,
        ),
        ("NaN", b"[NaN]", Profile::Rfc8259, ErrorKind::Syntax, 1),
        (
            "a nonzero number that rounds to zero",
            b"[1e-400]",
            Profile::IJson,
            ErrorKind::Number,
            1,
        ),
    ];
    for (name, document, profile, kind, offset) in cases {
        assert_eq!(check(document, &tight, profile, 1 << 17), error(kind, offset), "{name}");
    }
    // Within wide limits, each document of the set that has no other defect reads.
    let wide = Limits {
        depth_max: u16::MAX,
        number_digits_max: 4301,
        ..WIDE
    };
    for document in [deep.as_bytes(), members.as_bytes(), digits.as_bytes(), long.as_bytes()] {
        assert_eq!(check(document, &wide, Profile::Rfc8259, 100_001), Ok(()));
    }
    assert_eq!(
        check(digits.as_bytes(), &wide, Profile::IJson, 1),
        error(ErrorKind::Number, 0)
    );
    let deepest = "[".repeat(60_000) + &"]".repeat(60_000);
    let read_deepest = move || {
        let mut scratch = vec![Slot::EMPTY; 60_000];
        let read = super::super::read(deepest.as_bytes(), &wide, Profile::Rfc8259, &mut scratch, |_| Ok(()));
        assert_eq!(read, Ok(()));
    };
    // WebAssembly with WASI has no threads, so the read runs on the stack of the test there.
    if cfg!(target_family = "wasm") {
        read_deepest();
    } else {
        std::thread::Builder::new()
            .stack_size(64 << 10)
            .spawn(read_deepest)
            .expect("a thread starts")
            .join()
            .expect("the read returns on a stack of 64 KiB");
    }
}

/// Each limit accepts a document at the limit and refuses the first item past it, at its offset.
/// The count of a container comes back after a nested container ends.
#[test]
fn each_limit_refuses_the_first_item_past_it() {
    use Limit::{Depth, Elements, InputBytes, Members, NumberDigits, StringBytes};
    let with = |set: fn(&mut Limits, u32), value: u32| {
        let mut limits = WIDE;
        set(&mut limits, value);
        limits
    };
    let input: fn(&mut Limits, u32) = |limits, value| limits.input_bytes_max = value;
    let depth: fn(&mut Limits, u32) = |limits, value| limits.depth_max = u16::try_from(value).unwrap();
    let string: fn(&mut Limits, u32) = |limits, value| limits.string_bytes_max = value;
    let members: fn(&mut Limits, u32) = |limits, value| limits.members_max = value;
    let elements: fn(&mut Limits, u32) = |limits, value| limits.elements_max = value;
    let digits: fn(&mut Limits, u32) = |limits, value| limits.number_digits_max = u16::try_from(value).unwrap();
    // The document, the limit that it meets, and the offset of the first item past one less.
    type Set = fn(&mut Limits, u32);
    let cases: [(&str, Set, Limit, u32, usize); 12] = [
        ("[1,2]", input, InputBytes, 5, 4),
        ("[[[]]]", depth, Depth, 3, 2),
        (r#"{"a":[{"b":[]}]}"#, depth, Depth, 4, 11),
        (r#"["ab\ncd"]"#, string, StringBytes, 6, 7),
        (r#"["ab\n"]"#, string, StringBytes, 4, 5),
        (r#"{"abc":1}"#, string, StringBytes, 3, 4),
        (r#"{"a":1,"b":2,"c":3}"#, members, Members, 3, 13),
        (r#"{"a":{"x":1,"y":2},"b":1,"c":1}"#, members, Members, 3, 25),
        ("[1,2,3]", elements, Elements, 3, 5),
        ("[[1],[2],[3]]", elements, Elements, 3, 9),
        ("[[1,2,3]]", elements, Elements, 3, 6),
        ("[-12.34e+56]", digits, NumberDigits, 6, 10),
    ];
    for (document, set, limit, value, offset) in cases {
        for profile in PROFILES {
            assert_eq!(
                check(document.as_bytes(), &with(set, value), profile, 16),
                Ok(()),
                "{document} at {value}"
            );
            assert_eq!(
                check(document.as_bytes(), &with(set, value - 1), profile, 16),
                error(ErrorKind::Limit(limit), offset),
                "{document} at {}",
                value - 1
            );
        }
    }
}

/// A scratch needs one slot for each open container and one for each key of each open object. A
/// scratch one slot too short refuses at the item that needs the slot.
#[test]
fn a_scratch_one_slot_too_short_refuses_at_the_item_that_needs_it() {
    for (document, slots, offset) in [
        (r#"{"a":1,"b":2}"#, 3, 7),
        ("[[[]]]", 3, 2),
        (r#"[{"a":{"b":0}}]"#, 5, 7),
    ] {
        assert_eq!(
            check(document.as_bytes(), &WIDE, Profile::Rfc8259, slots),
            Ok(()),
            "{document}"
        );
        assert_eq!(
            check(document.as_bytes(), &WIDE, Profile::Rfc8259, slots - 1),
            error(ErrorKind::Scratch, offset),
            "{document}"
        );
    }
}

/// A duplicate key in an object at depth 3 that the decoder skips refuses before the reader
/// reports the end of that object. An object of 16 members compares each key at once. An object
/// of 17 compares when it ends.
#[test]
fn a_duplicate_in_a_skipped_object_refuses_before_the_end_of_the_object() {
    for members in [16, 17] {
        let keys: Vec<String> = (1..members).map(|index| format!("\"k{index}\":0")).collect();
        let inner = format!("{{\"k0\":0,{},\"k0\":0}}", keys[..members - 2].join(","));
        let document = format!("{{\"a\":{{\"b\":{inner},\"c\":1}},\"d\":2}}");
        let duplicate = document.rfind("\"k0\"").unwrap();
        let expected = error(ErrorKind::Duplicate, duplicate);
        let walk = read_each(document.as_bytes(), &WIDE, Profile::Rfc8259, 64, |reader| {
            let mut outer = reader.value().members()?;
            while let Some((_, value)) = outer.next()? {
                let mut inner = value.members()?;
                while let Some((key, value)) = inner.next()? {
                    if key == "b" {
                        value.skip()?;
                    }
                }
            }
            Ok(())
        });
        assert_eq!(walk, expected, "{members} members");
        let mut scratch = vec![Slot::EMPTY; 64];
        let given = each_kernel(document.as_bytes(), Profile::Rfc8259, |kernel| {
            let mut ends = 0;
            let result = visit_with(
                kernel,
                document.as_bytes(),
                &WIDE,
                Profile::Rfc8259,
                &mut scratch,
                |token| {
                    ends += usize::from(token == Token::ObjectEnd);
                    Ok(())
                },
            );
            assert_eq!(
                ends, 0,
                "{members} members: the error comes before the end of each object"
            );
            result
        });
        assert_eq!(given, expected, "{members} members");
    }
}

/// After a nested container ends, the frame of its object comes back with the filter of its keys:
/// a key after the container that equals a key before it refuses.
#[test]
fn a_duplicate_after_a_nested_container_refuses() {
    for document in [r#"{"a":{"b":1},"a":2}"#, r#"{"a":[1,{"a":1}],"a":2}"#] {
        let at = document.rfind(r#""a""#).unwrap();
        for profile in PROFILES {
            assert_eq!(
                check(document.as_bytes(), &WIDE, profile, 16),
                error(ErrorKind::Duplicate, at),
                "{document}"
            );
        }
    }
}

/// Errors come in the order of their offsets, also when an object of more than 16 members checks
/// a later key only when it ends: an error after a duplicate key, in the object or in an object in
/// it, or the end of the input, gives the duplicate.
#[test]
fn a_duplicate_comes_before_a_later_error_in_a_large_object() {
    // The members of an object, `"k0":0` to `"k{count - 1}":0`, with the key of member `twin` equal
    // to the key of member 2, and the value of member `bad` replaced by `value`.
    let members = |count: usize, twin: usize, bad: usize, value: &str| {
        (0..count)
            .map(|index| {
                let key = if index == twin { 2 } else { index };
                let value = if index == bad { value } else { "0" };
                format!("\"k{key}\":{value}")
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    // The offset of the second `"k2"`.
    let twin = |document: &str| document.match_indices("\"k2\"").nth(1).map_or(0, |(at, _)| at);
    let cases = [
        // The duplicate at member 17, and an error of the grammar at member 18.
        format!("{{{}}}", members(20, 17, 18, "tru")),
        // The same error in an object in a member after the duplicate.
        format!("{{{}}}", members(20, 17, 18, r#"{"x":[1,tru]}"#)),
        // The same in an object in the large object, which is not the root.
        format!("[{{\"a\":{{{}}}}}]", members(20, 17, 18, "tru")),
        // The input ends after the duplicate.
        format!("{{{}", members(20, 17, 18, "0")),
        // A duplicate at member 3 refuses at once, before an error at member 15.
        format!("{{{}}}", members(17, 3, 15, "tru")),
    ];
    for document in &cases {
        for profile in PROFILES {
            assert_eq!(
                check(document.as_bytes(), &WIDE, profile, 64),
                error(ErrorKind::Duplicate, twin(document)),
                "{document}"
            );
        }
    }
    // An error before the duplicate comes first.
    let document = format!("{{{}}}", members(20, 17, 16, "tru"));
    let at = document.find("tru").unwrap() + 3;
    assert_eq!(
        check(document.as_bytes(), &WIDE, Profile::Rfc8259, 64),
        error(ErrorKind::Syntax, at)
    );
}

/// Keys of 20 bytes that differ only in bytes 4 to 7 and 12 to 15, so their fingerprints are equal:
/// the first, middle, and last 4 bytes, and the length.
fn twin_key(index: usize) -> String {
    format!("aaaa{index:04}aaaa{:04}aaaa", index * 7 % 10_000)
}

/// The object of `keys`, each with the value 0, and `tail` after the last member.
fn object_of(keys: &[String], tail: &str) -> String {
    let members: Vec<String> = keys.iter().map(|key| format!("\"{key}\":0")).collect();
    format!("{{{}{tail}}}", members.join(","))
}

/// In an object of more than 16 members, keys whose fingerprints are equal but whose bytes differ
/// read: the filter compares them, and it refuses no key that is not a duplicate.
#[test]
fn keys_of_a_large_object_with_one_fingerprint_that_differ_read() {
    // 16 keys that the linear check compares, then 6 keys of one fingerprint, so the filter
    // compares 5 of them and does not sort.
    let mut keys: Vec<String> = (0..16).map(|index| format!("k{index}")).collect();
    keys.extend((0..6).map(twin_key));
    for profile in PROFILES {
        assert_eq!(check(object_of(&keys, "").as_bytes(), &WIDE, profile, 32), Ok(()));
    }
}

/// Keys of one length with the same first, middle, and last 4 bytes, as a hostile object can have:
/// past 8 keys of one fingerprint, the check sorts. It still refuses each duplicate, before a later
/// error, and an object of such distinct keys reads.
#[test]
fn keys_that_share_their_start_middle_and_end_refuse_each_duplicate() {
    let mut keys: Vec<String> = (0..2000).map(twin_key).collect();
    assert_eq!(
        check(object_of(&keys, "").as_bytes(), &WIDE, Profile::Rfc8259, 2048),
        Ok(())
    );
    for twin in [0, 5, 1998] {
        keys.push(twin_key(twin));
        for tail in ["", r#","x":tru"#] {
            let document = object_of(&keys, tail);
            let at = document.rfind(&format!("\"{}\"", twin_key(twin))).unwrap();
            assert_eq!(
                check(document.as_bytes(), &WIDE, Profile::Rfc8259, 2048),
                error(ErrorKind::Duplicate, at),
                "{twin} {tail}"
            );
        }
        keys.pop();
    }
}

/// Keys compare after unescape, in an object of up to 16 members and in a larger one. Thus
/// `"\u0061lg"` equals `"alg"`.
#[test]
fn keys_compare_after_unescape() {
    let pairs = [
        (r#""alg""#, r#""\u0061lg""#),
        (r#""\u0061lg""#, r#""alg""#),
        (r#""a\u006cg""#, r#""al\u0067""#),
        (r#""\/""#, r#""/""#),
        ("\"\u{1f600}\"", r#""\ud83d\ude00""#),
        (r#""\n""#, r#""\u000A""#),
    ];
    for (first, second) in pairs {
        for filler in [0, 15] {
            let others: String = (0..filler).map(|index| format!("\"f{index}\":0,")).collect();
            let document = format!("{{{first}:1,{others}{second}:2}}");
            let at = document.len() - second.len() - 3;
            for profile in PROFILES {
                assert_eq!(
                    check(document.as_bytes(), &WIDE, profile, 32),
                    error(ErrorKind::Duplicate, at),
                    "{document}"
                );
            }
        }
    }
    // Each key of 1 to 24 bytes, with each of its bytes once as an escape: the fingerprint of the
    // key with the escape reads the same bytes as that of the key without it.
    let letters = b"abcdefghijklmnopqrstuvwx";
    let text = |bytes: &[u8]| core::str::from_utf8(bytes).unwrap().to_owned();
    for len in 1..=letters.len() {
        let plain = text(&letters[..len]);
        for at in 0..len {
            let escaped = format!(
                "{}\\u{:04x}{}",
                text(&letters[..at]),
                letters[at],
                text(&letters[at + 1..len])
            );
            for (first, second) in [(&*plain, &*escaped), (&*escaped, &*plain)] {
                for filler in [0, 15] {
                    let others: String = (0..filler).map(|index| format!("\"f{index}\":0,")).collect();
                    let document = format!("{{\"{first}\":1,{others}\"{second}\":2}}");
                    let at = document.len() - second.len() - 5;
                    assert_eq!(
                        check(document.as_bytes(), &WIDE, Profile::Rfc8259, 32),
                        error(ErrorKind::Duplicate, at),
                        "{document}"
                    );
                }
            }
        }
    }
    for distinct in [r#"{"alg":1,"alh":2}"#, r#"{"a":1,"aa":2}"#, r#"{"é":1,"e":2}"#] {
        assert_eq!(
            check(distinct.as_bytes(), &WIDE, Profile::Rfc8259, 4),
            Ok(()),
            "{distinct}"
        );
    }
    let found = read_each(br#"{"alg":"ES256"}"#, &WIDE, Profile::IJson, 4, |reader| {
        let mut found = false;
        let mut members = reader.value().members()?;
        while let Some((key, value)) = members.next()? {
            found = key == "alg" && value.str()? == "ES256";
        }
        Ok(found)
    });
    assert_eq!(found, Ok(true));
}

/// An invalid UTF-8 sequence at a token start in the last 3 bytes of a block, which ends in the
/// next block, gives the UTF-8 error at its offset, not an error of the grammar. A valid
/// character there gives the error of the grammar. This holds at each place where a token starts.
#[test]
fn an_invalid_sequence_at_a_token_in_the_last_bytes_of_a_block_gives_the_utf8_error() {
    let invalid: [&[u8]; 5] = [
        &[0xe2, 0x82, b' '],
        &[0xf0, 0x9f, 0x98, b' '],
        &[0xc3, b' '],
        &[0x80],
        &[0xff],
    ];
    // The bytes before the sequence: a value in an array, a key, the end of a value, and the root.
    let before: [(&[u8], &[u8]); 4] = [(b"[0,", b"]"), (b"{", b"}"), (b"[0", b"]"), (b"", b"")];
    for at in 60..64 {
        for (head, tail) in before {
            for sequence in invalid.iter().copied().chain([&b"\xc3\xa9"[..]]) {
                let mut document = head.to_vec();
                document.resize(at, b' ');
                document.extend_from_slice(sequence);
                document.resize(at + 70, b' ');
                document.extend_from_slice(tail);
                let kind = if core::str::from_utf8(&document).is_err() {
                    ErrorKind::Utf8
                } else {
                    ErrorKind::Syntax
                };
                assert_eq!(
                    check(&document, &WIDE, Profile::Rfc8259, 4),
                    error(kind, at),
                    "{sequence:x?} at {at} after {:?}",
                    String::from_utf8_lossy(head)
                );
            }
        }
    }
}

/// Each document of profile A with one byte changed or one character added: an error comes no
/// later than the UTF-8 error of `core::str::from_utf8`, and at that offset, it is the UTF-8
/// error. A document that is UTF-8 has no UTF-8 error.
#[test]
fn each_changed_document_gives_no_error_after_its_utf8_error() {
    for document in generated_documents()
        .into_iter()
        .filter(|(class, _)| class.starts_with('a'))
    {
        let text = document.1;
        // Each place in a short document, and each 11th place in a longer one, so the test takes seconds.
        let step = if text.len() > 1024 { 11 } else { 1 };
        for at in (0..text.len()).step_by(step) {
            for change in [&[0x80][..], &[0xc3], &[0xff], &[0xe9], "é".as_bytes()] {
                let mut changed = text.clone();
                if change.len() == 1 {
                    changed[at] = change[0];
                } else {
                    changed.splice(at..at, change.iter().copied());
                }
                let result = check(&changed, &WIDE, Profile::Rfc8259, 64);
                match core::str::from_utf8(&changed) {
                    Ok(_) => {
                        assert!(
                            result.is_ok_and(|()| true) || result.is_err_and(|error| error.kind() != ErrorKind::Utf8)
                        )
                    }
                    Err(utf8) => {
                        let error = result.expect_err("a document that is not UTF-8 refuses");
                        let offset = utf8.valid_up_to();
                        assert!(error.offset() <= offset, "{error:?} after {offset}");
                        assert_eq!(
                            error.offset() == offset,
                            error.kind() == ErrorKind::Utf8,
                            "{error:?} {offset}"
                        );
                    }
                }
            }
        }
    }
}

/// A value that the walk skips gets each check of its profile: the walk gives the error that the
/// tokens give.
#[test]
fn a_skipped_value_gets_each_check_of_its_profile() {
    let defects: [(&[u8], Profile); 11] = [
        (br#""\ud800""#, Profile::Rfc8259),
        (br#""\x""#, Profile::Rfc8259),
        (b"\"a\x01\"", Profile::Rfc8259),
        (br#"{"a":1,"a":2}"#, Profile::Rfc8259),
        (b"01", Profile::Rfc8259),
        (b"tru", Profile::Rfc8259),
        (b"[1,]", Profile::Rfc8259),
        (b"\"\xff\"", Profile::Rfc8259),
        ("\"\u{fffe}\"".as_bytes(), Profile::IJson),
        (b"1e400", Profile::IJson),
        (b"9007199254740992", Profile::IJson),
    ];
    for (defect, profile) in defects {
        let document = [br#"{"skip":[0,{"deep":"#, defect, br#"}],"alg":"x"}"#].concat();
        let tokens = check(&document, &WIDE, profile, 16);
        assert!(tokens.is_err(), "{}", String::from_utf8_lossy(&document));
        let left = read_each(&document, &WIDE, profile, 16, |reader| {
            let mut members = reader.value().members()?;
            while let Some((key, value)) = members.next()? {
                if key == "alg" {
                    value.str()?;
                }
            }
            Ok(())
        });
        let skipped = read_each(&document, &WIDE, profile, 16, |reader| reader.value().skip());
        let nothing = read_each(&document, &WIDE, profile, 16, |_| Ok(()));
        for walk in [left, skipped, nothing] {
            assert_eq!(walk, tokens, "{}", String::from_utf8_lossy(&document));
        }
    }
}

/// The walk reads a value with a type, and `kind` gives the type to choose the read. A read of
/// another type refuses at the value and takes nothing.
#[test]
fn the_walk_reads_each_type_and_refuses_another() {
    for (document, expected) in [(r#"{"aud":"a"}"#, vec!["a"]), (r#"{"aud":["a","b"]}"#, vec!["a", "b"])] {
        let audience = read_each(document.as_bytes(), &WIDE, Profile::IJson, 4, |reader| {
            let mut audience = Vec::new();
            let mut members = reader.value().members()?;
            while let Some((_, mut value)) = members.next()? {
                if value.kind()? == Kind::String {
                    audience.push(decoded(value.str()?));
                } else {
                    let mut elements = value.elements()?;
                    while let Some(value) = elements.next()? {
                        audience.push(decoded(value.str()?));
                    }
                }
            }
            Ok(audience)
        });
        assert_eq!(audience, Ok(expected.into_iter().map(String::from).collect()));
    }
    let values = read_each(br#"[1.5,true,null,"x",[],{}]"#, &WIDE, Profile::IJson, 4, |reader| {
        let mut seen = Vec::new();
        let mut index = 0;
        let mut elements = reader.value().elements()?;
        while let Some(value) = elements.next()? {
            index += 1;
            match index {
                1 => seen.push(value.number()?.as_str().to_owned()),
                2 => seen.push(value.bool()?.to_string()),
                3 => value.null()?,
                4 => seen.push(decoded(value.str()?)),
                5 => _ = value.elements()?,
                _ => _ = value.members()?,
            }
        }
        Ok(seen)
    });
    assert_eq!(values, Ok(vec!["1.5".to_owned(), "true".to_owned(), "x".to_owned()]));
    for (document, offset) in [("[1]", 1), (r#"{"a":true}"#, 5), ("[]", 2)] {
        let read = read_each(document.as_bytes(), &WIDE, Profile::IJson, 4, |reader| {
            if document.starts_with('{') {
                let mut members = reader.value().members()?;
                while let Some((_, value)) = members.next()? {
                    value.str()?;
                }
            } else {
                let mut elements = reader.value().elements()?;
                while let Some(value) = elements.next()? {
                    value.str()?;
                }
            }
            reader.value().str().map(drop)
        });
        assert_eq!(read, error(ErrorKind::Type, offset), "{document}");
    }
}

/// A loop can leave a value, or a container that it opened, part way. The next call of the walk
/// skips the rest, with each check, and gives the next item of its own container. After the end
/// of its container, a walk gives `None` again and takes nothing after the end.
#[test]
fn the_walk_skips_the_rest_of_what_the_loop_left() {
    let walk = |document: &[u8]| {
        read_each(document, &WIDE, Profile::Rfc8259, 16, |reader| {
            let mut seen = Vec::new();
            let mut root = reader.value().members()?;
            while let Some((key, mut value)) = root.next()? {
                seen.push(decoded(key));
                match value.kind()? {
                    Kind::Object => {
                        let mut inner = value.members()?;
                        inner.next()?;
                    }
                    Kind::Array => {
                        let mut elements = value.elements()?;
                        while elements.next()?.is_some() {}
                        if elements.next()?.is_some() {
                            seen.push("after the end".to_owned());
                        }
                    }
                    _ => seen.push(decoded(value.str()?)),
                }
            }
            Ok(seen)
        })
    };
    let document = br#"{"a":{"x":{"alg":1},"y":[2,{"z":3}]},"b":[4,[5]],"alg":"outer"}"#;
    assert_eq!(
        walk(document),
        Ok(["a", "b", "alg", "outer"].map(String::from).to_vec())
    );
    let duplicate = br#"{"a":{"x":1,"y":[2,{"z":3,"z":4}]},"alg":"outer"}"#;
    let offset = duplicate.windows(4).rposition(|window| window == br#""z":"#).unwrap();
    assert_eq!(walk(duplicate), Err(Error::new(ErrorKind::Duplicate, offset)));
}

/// The classes of `benches/data/json`, one document for each line of a `.jsonl` file.
fn generated_documents() -> Vec<(String, Vec<u8>)> {
    let mut documents = Vec::new();
    let mut classes: Vec<_> = fs::read_dir("benches/data/json")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    classes.sort();
    for class in classes {
        let name = class.file_name().unwrap().to_string_lossy().into_owned();
        let mut files: Vec<_> = fs::read_dir(&class)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        files.sort();
        for file in files {
            let text = fs::read(&file).unwrap();
            if file.extension().is_some_and(|extension| extension == "jsonl") {
                documents.extend(
                    text.split(|&byte| byte == b'\n')
                        .filter(|line| !line.is_empty())
                        .map(|line| (name.clone(), line.to_vec())),
                );
            } else {
                documents.push((name.clone(), text));
            }
        }
    }
    assert!(!documents.is_empty());
    documents
}

/// The tokens of each corpus give the tree of serde_json: the reader gives each string, key, and
/// number of the document, in order.
#[test]
fn the_tokens_of_each_corpus_give_the_tree_of_serde_json() {
    let fetched = ["twitter", "citm_catalog", "canada"].map(|name| {
        let path = format!("benches/data/fetched/{name}.json");
        let text = fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}; benches/data/README.md fetches it"));
        (name.to_owned(), text)
    });
    let generated = generated_documents()
        .into_iter()
        .map(|document| (document, &PROFILES[..]));
    // twitter.json has integers past 2^53, which `IJson` refuses.
    let fetched = fetched.into_iter().map(|document| (document, &PROFILES[..1]));
    for ((class, document), profiles) in generated.chain(fetched) {
        let expected: serde_json::Value = serde_json::from_slice(&document).unwrap();
        for &profile in profiles {
            assert_eq!(tree(&document, profile).as_ref(), Ok(&expected), "{class} {profile:?}");
        }
    }
}

/// Each string of up to 7 bytes over `\`, `u`, `0`, `a`, `"`, `n`, and 0x1f, in quotes: the reader
/// accepts it exactly when serde_json does, and it decodes it to the string of serde_json.
#[test]
fn each_short_string_decodes_as_serde_json_does() {
    const BYTES: [u8; 7] = [b'\\', b'u', b'0', b'a', b'"', b'n', 0x1f];
    let mut document = Vec::new();
    for len in 0..=7_u32 {
        for index in 0..BYTES.len().pow(len) {
            document.clear();
            document.push(b'"');
            let mut rest = index;
            for _ in 0..len {
                document.push(BYTES[rest % BYTES.len()]);
                rest /= BYTES.len();
            }
            document.push(b'"');
            let expected = serde_json::from_slice::<String>(&document).ok();
            let read = tokens(&document, &WIDE, Profile::Rfc8259, 1)
                .ok()
                .map(|tokens| match tokens.as_slice() {
                    [Owned::Str(text)] => text.clone(),
                    other => panic!("{other:?}"),
                });
            assert_eq!(read, expected, "{:?}", String::from_utf8_lossy(&document));
        }
    }
}

/// Each byte at each of the 4 digits of a `\u` escape: the reader accepts the escape exactly when
/// the byte is a hex digit, and gives the character of its value. Otherwise it refuses at that
/// byte: a byte of 0x80 or more is not UTF-8, and each other byte is a syntax error.
#[test]
fn each_byte_at_each_digit_of_a_unicode_escape_is_a_hex_digit_or_refuses() {
    for at in 3..7 {
        for byte in 0..=u8::MAX {
            let mut document = b"\"\\u0000\"".to_vec();
            document[at] = byte;
            let expected = if byte.is_ascii_hexdigit() {
                let digits = core::str::from_utf8(&document[3..7]).unwrap();
                let char = char::from_u32(u32::from_str_radix(digits, 16).unwrap()).unwrap();
                Ok(vec![Owned::Str(char.to_string())])
            } else {
                let kind = if byte >= 0x80 {
                    ErrorKind::Utf8
                } else {
                    ErrorKind::Syntax
                };
                Err(Error::new(kind, at))
            };
            assert_eq!(tokens(&document, &WIDE, Profile::Rfc8259, 1), expected, "{document:x?}");
        }
    }
}

/// The number rule of `IJson`: the reader accepts a number exactly when the oracle does. The
/// oracle counts the significant digits of the text, and it reads the value with the correctly
/// rounded parser of `core`.
#[test]
fn ijson_accepts_a_number_exactly_when_binary64_holds_it() {
    fn holds(text: &str) -> bool {
        let unsigned = text.trim_start_matches('-');
        let (mantissa, exponent) = unsigned
            .split_once(['e', 'E'])
            .map_or((unsigned, None), |(m, e)| (m, Some(e)));
        let (integer, fraction) = mantissa.split_once('.').map_or((mantissa, None), |(i, f)| (i, Some(f)));
        if fraction.is_none() && exponent.is_none() {
            return integer.parse::<u128>().is_ok_and(|value| value < 1 << 53);
        }
        let digits = format!("{integer}{}", fraction.unwrap_or(""));
        let significant = digits.trim_start_matches('0').trim_end_matches('0');
        if significant.is_empty() {
            return true;
        }
        let value: f64 = text.parse().unwrap();
        significant.len() <= 17 && value.is_finite() && value != 0.0
    }
    let mut numbers: Vec<String> = [
        "1.7976931348623157e308",
        "1.7976931348623158e308",
        "1.7976931348623159e308",
        "17976931348623158e292",
        "17976931348623159e292",
        "0.17976931348623158e309",
        "-1.7976931348623159e308",
        "2.4703282292062327e-324",
        "2.4703282292062328e-324",
        "4.9406564584124654e-324",
        "5e-324",
        "3e-324",
        "2e-324",
        "-2.4703282292062327e-324",
        "0e99999999999999999999",
        "0.000e-99999",
        "-0.0",
        "-0",
        "0",
        "9007199254740991",
        "-9007199254740991",
        "9007199254740992",
        "-9007199254740992",
        "12345678901234567.0",
        "123456789012345678.0",
        "1.00000000000000000000",
        "100000000000000000000e-5",
        "1e308",
        "1e309",
        "1e-323",
        "1e-324",
        "1E+2",
        "0.0000000000000000000000000000001e-293",
        "10000000000000000000000000000000e276",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    // A pseudo-random number under `bound` (xorshift64).
    let mut below = |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        usize::try_from(state % u64::try_from(bound).unwrap()).unwrap()
    };
    for _ in 0..100_000 {
        let digits = 1 + below(20);
        let mantissa: String = (0..digits)
            .map(|_| char::from_digit(u32::try_from(below(10)).unwrap(), 10).unwrap())
            .collect();
        let mantissa = mantissa.trim_start_matches('0');
        let mantissa = if mantissa.is_empty() { "0" } else { mantissa };
        let point = below(digits + 1);
        let text = match (below(3), mantissa.split_at_checked(point)) {
            (0, _) => mantissa.to_owned(),
            (1, Some((integer, fraction))) if !integer.is_empty() && !fraction.is_empty() => {
                format!("{integer}.{fraction}")
            }
            _ => format!("{mantissa}e{}", i64::try_from(below(800)).unwrap() - 400),
        };
        numbers.push(text);
    }
    for text in numbers {
        let document = format!("[{text}]");
        let expected = if holds(&text) {
            Ok(())
        } else {
            error(ErrorKind::Number, 1)
        };
        assert_eq!(check(document.as_bytes(), &WIDE, Profile::IJson, 1), expected, "{text}");
        assert_eq!(check(document.as_bytes(), &WIDE, Profile::Rfc8259, 1), Ok(()), "{text}");
    }
}
