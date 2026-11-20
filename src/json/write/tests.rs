//! The writer against references: the escapes of RFC 8259 and RFC 8785 byte by byte, `core::fmt`
//! for numbers, and the reader of this crate for each written text.

extern crate std;

use std::{format, fs, string::String, vec, vec::Vec};

use serde_json::Value as Tree;

use super::{Place, write};
use crate::json::{Error, ErrorKind, Limits, Profile, Slot, read, reader::tests::tree, string_len};

/// The JSON string of `text` with the escapes of RFC 8785, one character at a time.
fn reference(text: &str) -> String {
    let mut out = String::from("\"");
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if u32::from(control) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(control))),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Writes `tree` at `place`. A number writes as `u64`, as `i64`, or as `f64`, in this order.
fn put(place: Place<'_>, tree: &Tree) -> Result<(), Error> {
    match tree {
        Tree::Null => place.null(),
        Tree::Bool(value) => place.bool(*value),
        Tree::Number(number) => match (number.as_u64(), number.as_i64()) {
            (Some(value), _) => place.u64(value),
            (None, Some(value)) => place.i64(value),
            (None, None) => place.f64(
                number
                    .as_f64()
                    .expect("serde_json holds each number in u64, i64, or f64"),
            ),
        },
        Tree::String(text) => place.str(text),
        Tree::Array(items) => place.array(|array| items.iter().try_for_each(|item| put(array.item(), item))),
        Tree::Object(members) => place.object(|object| {
            members
                .iter()
                .try_for_each(|(key, value)| put(object.member(key), value))
        }),
    }
}

/// The text that `fill` writes into a slice of `len` bytes.
fn written(len: usize, fill: impl FnOnce(Place<'_>) -> Result<(), Error>) -> Result<Vec<u8>, Error> {
    let mut out = vec![0; len];
    let len = write(&mut out, fill)?;
    out.truncate(len);
    Ok(out)
}

/// Each document of `benches/data/json` and of the fetched corpora. A missing fetched file fails.
fn corpora() -> Vec<Tree> {
    let mut files = Vec::new();
    for class in fs::read_dir("benches/data/json").unwrap() {
        files.extend(
            fs::read_dir(class.unwrap().path())
                .unwrap()
                .map(|file| file.unwrap().path()),
        );
    }
    files.extend(["twitter", "citm_catalog", "canada"].map(|name| format!("benches/data/fetched/{name}.json").into()));
    files.sort();
    let mut documents = Vec::new();
    for file in files {
        let text = fs::read(&file).unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        let lines: Vec<&[u8]> = if file.extension().is_some_and(|extension| extension == "jsonl") {
            text.split(|&byte| byte == b'\n')
                .filter(|line| !line.is_empty())
                .collect()
        } else {
            vec![&text]
        };
        documents.extend(
            lines
                .into_iter()
                .map(|line| serde_json::from_slice::<Tree>(line).unwrap()),
        );
    }
    documents
}

/// Each string of `tree`, keys included.
fn strings<'t>(tree: &'t Tree, out: &mut Vec<&'t str>) {
    match tree {
        Tree::String(text) => out.push(text),
        Tree::Array(items) => items.iter().for_each(|item| strings(item, out)),
        Tree::Object(members) => members.iter().for_each(|(key, value)| {
            out.push(key);
            strings(value, out);
        }),
        Tree::Null | Tree::Bool(_) | Tree::Number(_) => {}
    }
}

#[test]
fn each_corpus_document_writes_text_that_the_reader_reads_as_its_tree() {
    for document in corpora() {
        let text = written(1 << 24, |place| put(place, &document)).unwrap();
        assert_eq!(tree(&text, Profile::Rfc8259), Ok(document));
    }
}

/// Each corpus string after 0 to 15 bytes, so that each string ends at each place of the last block
/// of the scan, and each escape falls at each place of a block.
#[test]
fn each_corpus_string_at_each_tail_length_writes_its_reference_escape_and_its_exact_length() {
    let documents = corpora();
    let mut texts = Vec::new();
    documents.iter().for_each(|document| strings(document, &mut texts));
    texts.sort_unstable();
    texts.dedup();
    let controls: String = (0..0x20_u8).map(char::from).chain("\"\\é😀".chars()).collect();
    let mut padded = String::new();
    for text in texts.iter().copied().chain([controls.as_str()]) {
        for pad in 0..16 {
            padded.clear();
            padded.extend(core::iter::repeat_n('x', pad));
            padded.push_str(text);
            let out = written(padded.len() * 6 + 2, |place| place.str(&padded)).unwrap();
            assert_eq!(out, reference(&padded).as_bytes(), "the string {padded:?}");
            assert_eq!(string_len(&padded), out.len(), "the length of {padded:?}");
            assert_eq!(tree(&out, Profile::Rfc8259), Ok(Tree::String(padded.clone())));
        }
    }
}

/// Each slice shorter than the document: each member that does not fit refuses at the written
/// length and writes no byte, and the members after it still write when they fit.
#[test]
fn a_write_past_the_end_of_the_slice_refuses_and_keeps_the_written_length() {
    let members: [(&str, Tree); 6] = [
        ("challenge", Tree::from("dGhlIGNoYWxsZW5nZQ")),
        ("timeout", Tree::from(60_000)),
        (
            "rp",
            serde_json::json!({ "id": "example.com", "name": "Example \"Co\"" }),
        ),
        (
            "excludeCredentials",
            serde_json::json!([{ "type": "public-key", "id": "a2V5LTE" }]),
        ),
        ("delta", Tree::from(-1.5e-7)),
        ("ok", Tree::from(true)),
    ];
    let full = written(1 << 12, |place| {
        place.object(|object| {
            members
                .iter()
                .try_for_each(|(key, value)| put(object.member(key), value))
        })
    })
    .unwrap();
    for len in 0..=full.len() {
        let mut refusals = Vec::new();
        let result = written(len, |place| {
            place.object(|object| {
                for (key, value) in &members {
                    if let Err(error) = put(object.member(key), value) {
                        refusals.push(error);
                    }
                }
                Ok(())
            })
        });
        let Ok(text) = result else {
            assert!(
                len < 2,
                "an object of {len} bytes refuses only when `{{}}` does not fit"
            );
            assert_eq!(
                result,
                Err(Error {
                    kind: ErrorKind::Full,
                    offset: 0
                })
            );
            continue;
        };
        let Ok(Tree::Object(read)) = tree(&text, Profile::Rfc8259) else {
            panic!("the text of a slice of {len} bytes is not an object");
        };
        // Each member that fits, whole.
        assert!(
            read.iter()
                .all(|(key, value)| members.iter().any(|(name, member)| name == key && member == value))
        );
        assert_eq!(
            refusals.len() + read.len(),
            members.len(),
            "each member writes or refuses"
        );
        for refusal in refusals {
            assert_eq!(refusal.kind, ErrorKind::Full);
            // The written length at a refusal ends a member, or is the length after `{`.
            assert!(refusal.offset == 1 || matches!(text.get(refusal.offset), Some(b',' | b'}')));
        }
        assert_eq!(len == full.len(), text == full);
    }
}

#[test]
fn each_integer_writes_the_digits_of_core_fmt() {
    let powers = (0..20).map(|power| 10_u64.pow(power));
    let near = powers.flat_map(|power| [power - 1, power, power + 1]);
    let mut state = 0x5eed_0004_u64;
    let random = core::iter::repeat_with(|| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state >> (state % 64)
    });
    for value in near
        .chain([u64::MAX, u64::MAX - 1, 1 << 63])
        .chain(random.take(1 << 16))
    {
        let out = written(20, |place| place.u64(value)).unwrap();
        assert_eq!(out, format!("{value}").as_bytes());
        for signed in [value.cast_signed(), value.cast_signed().wrapping_neg()] {
            let out = written(20, |place| place.i64(signed)).unwrap();
            assert_eq!(out, format!("{signed}").as_bytes());
        }
    }
}

/// The printer against `core::fmt` is in the tests of `float`. Here each layout is a number of
/// RFC 8259 that the reader reads, and its text reads back to the same bits, −0 included.
#[test]
fn each_float_of_a_fixed_seed_sample_writes_a_number_that_reads_back() {
    let mut state = 0x5eed_0005_u64;
    let bits = core::iter::repeat_with(|| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    });
    let values: Vec<f64> = bits
        .take(1 << 16)
        .map(f64::from_bits)
        .chain([
            0.0,
            -0.0,
            1.0,
            1e-4,
            9.999999999999999e-5,
            1e16,
            9_999_999_999_999_998.0,
            5e-324,
            f64::MAX,
        ])
        .filter(|value| value.is_finite())
        .collect();
    let text = written(values.len() * 26, |place| {
        place.array(|array| values.iter().try_for_each(|&value| array.item().f64(value)))
    })
    .unwrap();
    let numbers = text
        .strip_prefix(b"[")
        .and_then(|text| text.strip_suffix(b"]"))
        .unwrap();
    for (number, value) in numbers.split(|&byte| byte == b',').zip(&values) {
        let number = core::str::from_utf8(number).unwrap();
        assert_eq!(number.parse::<f64>().map(f64::to_bits), Ok(value.to_bits()), "{number}");
    }
    let Ok(Tree::Array(read)) = tree(&text, Profile::Rfc8259) else {
        panic!("the reader reads the written floats as an array");
    };
    assert_eq!(read.len(), values.len());
}

#[test]
fn nan_and_infinity_refuse_and_a_closure_that_writes_no_value_refuses() {
    let refused = |kind, offset| Err(Error { kind, offset });
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // A refused member writes nothing and keeps the written length, here after `{"a":1`.
        let mut refusal = None;
        let text = written(64, |place| {
            place.object(|object| {
                object.member("a").u64(1)?;
                refusal = object.member("b").f64(value).err();
                Ok(())
            })
        });
        assert_eq!(text.as_deref(), Ok(br#"{"a":1}"#.as_slice()));
        assert_eq!(
            refusal,
            Some(Error {
                kind: ErrorKind::NonFinite,
                offset: 6
            })
        );
        // An array that its closure fails writes nothing, so the offset is its start.
        let result = written(64, |place| place.array(|array| array.item().f64(value)));
        assert_eq!(result, refused(ErrorKind::NonFinite, 0));
    }
    assert_eq!(written(64, |_| Ok(())), refused(ErrorKind::Type, 0));
}

/// The named float sweep of [CODEC Decisions], second session of 2026-11-19: each finite f32
/// value as an f64, and 10^9 random finite f64 values, print the digits of `core::fmt`, and
/// `Number::to_f64` reads each text back to its value. It runs once for each change to the printer
/// or to the reader of numbers, with `--run-ignored only`, and its result goes to the evidence of
/// the task.
#[test]
#[ignore = "the named float sweep runs for minutes, once for each change to the printer or to the reader of numbers"]
fn float_sweep() {
    const F64_VALUES: u64 = 1_000_000_000;
    const LIMITS: Limits = Limits {
        input_bytes_max: 32,
        depth_max: 0,
        string_bytes_max: 0,
        members_max: 0,
        elements_max: 0,
        number_digits_max: 32,
    };
    let read_back = |text: &str| {
        let mut scratch = [Slot::EMPTY; 1];
        let value = read(text.as_bytes(), &LIMITS, Profile::Rfc8259, &mut scratch, |reader| {
            reader.value().number()?.to_f64()
        });
        value.map(f64::to_bits)
    };
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let threads = u64::try_from(threads).unwrap();
    std::thread::scope(|scope| {
        for thread in 0..threads {
            scope.spawn(move || {
                let mut out = [0; 32];
                // Each finite f32, as the f64 of the same value.
                let f32_bits = (0..=u32::MAX).filter(|bits| u64::from(*bits) % threads == thread);
                for value in f32_bits
                    .map(f32::from_bits)
                    .filter(|value| value.is_finite())
                    .map(f64::from)
                {
                    let len = write(&mut out, |place| place.f64(value)).unwrap();
                    let text = core::str::from_utf8(&out[..len]).unwrap();
                    assert_eq!(text, format!("{value:?}"), "the form of {:#x}", value.to_bits());
                    assert_eq!(read_back(text), Ok(value.to_bits()), "{text} reads back");
                }
                let mut state = 0x5eed_5eed_u64 ^ thread.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let mut count = 0;
                while count < F64_VALUES / threads + u64::from(thread < F64_VALUES % threads) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let value = f64::from_bits(state.wrapping_mul(0x2545_f491_4f6c_dd1d));
                    if !value.is_finite() {
                        continue;
                    }
                    count += 1;
                    let len = write(&mut out, |place| place.f64(value)).unwrap();
                    let text = core::str::from_utf8(&out[..len]).unwrap();
                    assert_eq!(text, format!("{value:?}"), "the form of {:#x}", value.to_bits());
                    assert_eq!(read_back(text), Ok(value.to_bits()), "{text} reads back");
                }
            });
        }
    });
}
