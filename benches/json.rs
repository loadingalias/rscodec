//! Benchmark token visits, tree reads, typed decoding, writing, and float conversion.
//! Each row processes all documents in its input class. Check semantic results against serde_json.
//! simd-json mutates its input, so its timed read includes a copy into a reusable aligned buffer.

#[path = "common/aligned.rs"]
mod aligned;
mod common;
mod typed;

use core::{fmt::Write as _, hint::black_box};
use std::{fs, path::Path, sync::OnceLock};

use aligned::Aligned;
use common::Group;
use criterion::{Criterion, Throughput};
use rscodec::json::{self, Limits, Place, Profile, Slot, Token};
use serde_json::Value;
use simd_json::prelude::Writable as _;
use typed::Typed;

const CLASSES: [(&str, &str); 10] = [
    ("a1-client-data", "json/a1-client-data"),
    ("a2-jose-header", "json/a2-jose-header"),
    ("a3-claims", "json/a3-claims"),
    ("a4-oidc-discovery", "json/a4-oidc-discovery"),
    ("a5-jwks", "json/a5-jwks"),
    ("b1-sparse-index", "json/b1-sparse-index"),
    ("b2-rustc-diagnostics", "json/b2-rustc-diagnostics"),
    ("twitter", "fetched/twitter.json"),
    ("citm_catalog", "fetched/citm_catalog.json"),
    ("canada", "fetched/canada.json"),
];

const LIMITS: Limits = Limits {
    input_bytes_max: 1 << 24,
    depth_max: 64,
    string_bytes_max: 1 << 20,
    members_max: 1 << 16,
    elements_max: 1 << 20,
    number_digits_max: 64,
};

const SLOTS: usize = 1 << 12;

const RANDOM_F64_COUNT: usize = 10_000;

struct Class {
    name: &'static str,
    docs: Vec<Aligned>,
    trees: Vec<Value>,
    bytes: usize,
}

impl Class {
    fn inputs(&self) -> Vec<&[u8]> {
        self.docs.iter().map(|doc| &doc[..]).collect()
    }
}

fn classes() -> &'static [Class] {
    static CELLS: OnceLock<Vec<Class>> = OnceLock::new();
    CELLS.get_or_init(|| CLASSES.iter().map(|&(name, path)| load(name, path)).collect())
}

fn load(name: &'static str, path: &str) -> Class {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/data").join(path);
    assert!(
        path.exists(),
        "{} is missing: benches/data/README.md fetches it",
        path.display()
    );
    let mut files = if path.is_dir() {
        fs::read_dir(&path)
            .expect("the directory of a class is readable")
            .map(|entry| entry.expect("a directory entry is readable").path())
            .filter(|file| {
                file.extension()
                    .is_some_and(|extension| extension == "json" || extension == "jsonl")
            })
            .collect()
    } else {
        vec![path]
    };
    files.sort();
    let mut docs = Vec::new();
    let mut keep = |doc: &[u8]| docs.push(Aligned::new(doc));
    for file in files {
        let text = fs::read(&file).expect("a document file is readable");
        if file.extension().is_some_and(|extension| extension == "jsonl") {
            text.split(|&byte| byte == b'\n')
                .filter(|line| !line.is_empty())
                .for_each(&mut keep);
        } else {
            keep(&text);
        }
    }
    assert!(!docs.is_empty(), "class {name} has no document");
    let trees = docs
        .iter()
        .map(|doc| serde_json::from_slice(doc).expect("serde_json reads each document"))
        .collect();
    let bytes = docs.iter().map(|doc| doc.len()).sum();
    Class {
        name,
        docs,
        trees,
        bytes,
    }
}

fn copy_buffer(class: &Class) -> Aligned {
    Aligned::new(&vec![0; class.docs.iter().map(|doc| doc.len()).max().unwrap_or(0)])
}

fn read_row<S>(
    group: &mut Group<'_>,
    class: &'static Class,
    codec: &str,
    state: S,
    mut read: impl FnMut(&mut S, &'static [u8]) -> bool,
    mut tree: impl FnMut(&mut S, &'static [u8]) -> Value,
) {
    let docs = class.inputs();
    group.row(
        codec,
        class.name,
        state,
        |state| docs.iter().filter(|doc| read(state, doc)).count(),
        |read_count, state| {
            assert_eq!(read_count, docs.len(), "{codec} reads each document of {}", class.name);
            for (index, (doc, expected)) in docs.iter().zip(&class.trees).enumerate() {
                assert!(
                    tree(state, doc) == *expected,
                    "{codec} reads document {index} of {} into the reference tree",
                    class.name
                );
            }
        },
    );
}

fn read(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "json-read");
    for class in classes() {
        group.throughput(Throughput::Bytes(class.bytes.try_into().expect("a length fits u64")));
        let g = &mut group;
        read_row(
            g,
            class,
            "serde_json",
            (),
            |(), doc| black_box(serde_json::from_slice::<Value>(doc)).is_ok(),
            |(), doc| serde_json::from_slice(doc).expect("valid"),
        );
        read_row(
            g,
            class,
            "sonic-rs",
            (),
            |(), doc| black_box(sonic_rs::from_slice::<sonic_rs::Value>(doc)).is_ok(),
            |(), doc| {
                serde_json::to_value(sonic_rs::from_slice::<sonic_rs::Value>(doc).expect("valid")).expect("a tree")
            },
        );
        read_row(
            g,
            class,
            "simd-json",
            (copy_buffer(class), simd_json::Buffers::default()),
            |(copy, buffers), doc| {
                let copy = &mut copy[..doc.len()];
                copy.copy_from_slice(doc);
                black_box(simd_json::to_borrowed_value_with_buffers(copy, buffers)).is_ok()
            },
            |(copy, buffers), doc| {
                let copy = &mut copy[..doc.len()];
                copy.copy_from_slice(doc);
                serde_json::to_value(simd_json::to_borrowed_value_with_buffers(copy, buffers).expect("valid"))
                    .expect("a tree")
            },
        );
        read_row(
            g,
            class,
            "jiter",
            jiter::JsonValueScratch::new(),
            |scratch, doc| black_box(scratch.parse(doc, false, jiter::PartialMode::Off)).is_ok(),
            |scratch, doc| jiter_tree(&scratch.parse(doc, false, jiter::PartialMode::Off).expect("valid")),
        );
        read_row(
            g,
            class,
            "flexon",
            (),
            |(), doc| black_box(flexon::parse::<_, flexon::Value<'_>>(doc)).is_ok(),
            |(), doc| serde_json::to_value(flexon::parse::<_, flexon::Value<'_>>(doc).expect("valid")).expect("a tree"),
        );
        read_row(
            g,
            class,
            "rscodec",
            (vec![Slot::EMPTY; SLOTS], vec![0; LIMITS.string_bytes_max as usize]),
            |(scratch, buf), doc| {
                json::visit(
                    doc,
                    &LIMITS,
                    Profile::Rfc8259,
                    scratch,
                    #[inline(always)]
                    |token| take(token, buf),
                )
                .is_ok()
            },
            |(scratch, buf), doc| rscodec_tree(doc, scratch, buf),
        );
    }
}

#[inline(always)]
fn take(token: Token<'_>, buf: &mut [u8]) -> Result<(), json::Error> {
    match token {
        Token::Key(text) | Token::Str(text) => {
            black_box(text.decode(buf)?);
        }
        Token::Number(value) => {
            black_box(number(value)?);
        }
        token => {
            black_box(token);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Scalar {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
}

#[inline(always)]
fn number(number: json::Number<'_>) -> Result<Scalar, json::Error> {
    let integer = if number.as_str().starts_with('-') {
        number.to_i64().ok().filter(|&value| value != 0).map(Scalar::Signed)
    } else {
        number.to_u64().ok().map(Scalar::Unsigned)
    };
    match integer {
        Some(integer) => Ok(integer),
        None => number.to_f64().map(Scalar::Float),
    }
}

fn rscodec_tree(doc: &[u8], scratch: &mut [Slot], buf: &mut [u8]) -> Value {
    let mut open: Vec<(Value, Option<String>)> = Vec::new();
    let (mut key, mut root) = (None, None);
    json::visit(doc, &LIMITS, Profile::Rfc8259, scratch, |token| {
        let value = match token {
            Token::ObjectStart => {
                open.push((Value::Object(serde_json::Map::new()), key.take()));
                return Ok(());
            }
            Token::ArrayStart => {
                open.push((Value::Array(Vec::new()), key.take()));
                return Ok(());
            }
            Token::Key(text) => {
                key = Some(text.decode(buf)?.to_owned());
                return Ok(());
            }
            Token::ObjectEnd | Token::ArrayEnd => {
                let (value, name) = open.pop().expect("each end has its start");
                key = name;
                value
            }
            Token::Str(text) => Value::String(text.decode(buf)?.to_owned()),
            Token::Number(value) => match number(value)? {
                Scalar::Unsigned(value) => Value::from(value),
                Scalar::Signed(value) => Value::from(value),
                Scalar::Float(value) => Value::from(value),
            },
            Token::Bool(value) => Value::Bool(value),
            Token::Null => Value::Null,
        };
        match open.last_mut() {
            Some((Value::Object(members), _)) => {
                members.insert(key.take().expect("each member has a key"), value);
            }
            Some((Value::Array(elements), _)) => elements.push(value),
            _ => root = Some(value),
        }
        Ok(())
    })
    .expect("rscodec reads each document");
    root.expect("a document has a value")
}

fn matches_reference<'a, C: Typed>(doc: &'a [u8], value: Option<C::Of<'a>>) -> bool {
    let expected = serde_json::from_slice::<C::Of<'a>>(doc).expect("serde_json decodes each document");
    value == Some(expected)
}

fn typed_rows<C: Typed>(group: &mut Group<'_>, class: &'static Class) {
    let docs = class.inputs();
    let check = |codec: &str, decode: &mut dyn FnMut(&'static [u8]) -> bool| {
        let read_count = docs.iter().filter(|doc| decode(doc)).count();
        assert_eq!(
            read_count,
            docs.len(),
            "{codec} decodes each document of {}",
            class.name
        );
    };
    group.row(
        "serde_json",
        class.name,
        (),
        |()| {
            docs.iter()
                .filter(|doc| black_box(serde_json::from_slice::<C::Of<'_>>(doc)).is_ok())
                .count()
        },
        |count, ()| assert_eq!(count, docs.len()),
    );
    group.row(
        "sonic-rs",
        class.name,
        (),
        |()| {
            docs.iter()
                .filter(|doc| black_box(sonic_rs::from_slice::<C::Of<'_>>(doc)).is_ok())
                .count()
        },
        |_, ()| {
            check("sonic-rs", &mut |doc| {
                matches_reference::<C>(doc, sonic_rs::from_slice::<C::Of<'_>>(doc).ok())
            })
        },
    );
    group.row(
        "simd-json",
        class.name,
        copy_buffer(class),
        |copy| {
            docs.iter()
                .filter(|doc| {
                    let copy = &mut copy[..doc.len()];
                    copy.copy_from_slice(doc);
                    black_box(simd_json::serde::from_slice::<C::Of<'_>>(copy)).is_ok()
                })
                .count()
        },
        |_, copy| {
            check("simd-json", &mut |doc| {
                let copy = &mut copy[..doc.len()];
                copy.copy_from_slice(doc);
                matches_reference::<C>(doc, simd_json::serde::from_slice::<C::Of<'_>>(copy).ok())
            });
        },
    );
    group.row(
        "flexon",
        class.name,
        (),
        |()| {
            docs.iter()
                .filter(|doc| black_box(flexon::from_slice::<C::Of<'_>>(doc)).is_ok())
                .count()
        },
        |_, ()| {
            check("flexon", &mut |doc| {
                matches_reference::<C>(doc, flexon::from_slice::<C::Of<'_>>(doc).ok())
            })
        },
    );
    group.row(
        "rscodec",
        class.name,
        vec![Slot::EMPTY; 64],
        |scratch| {
            docs.iter()
                .filter(|doc| black_box(C::rscodec(doc, scratch)).is_ok())
                .count()
        },
        |_, scratch| {
            check("rscodec", &mut |doc| {
                matches_reference::<C>(doc, C::rscodec(doc, scratch).ok())
            })
        },
    );
}

fn typed(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "json-typed");
    type Rows = fn(&mut Group<'_>, &'static Class);
    let rows: [(&str, Rows); 5] = [
        ("a1-client-data", typed_rows::<typed::A1>),
        ("a2-jose-header", typed_rows::<typed::A2>),
        ("a3-claims", typed_rows::<typed::A3>),
        ("a4-oidc-discovery", typed_rows::<typed::A4>),
        ("a5-jwks", typed_rows::<typed::A5>),
    ];
    for (name, rows) in rows {
        let class = classes()
            .iter()
            .find(|class| class.name == name)
            .expect("each class of profile A");
        group.throughput(Throughput::Bytes(class.bytes.try_into().expect("a length fits u64")));
        rows(&mut group, class);
    }
}

/// The serde_json tree of a jiter value. jiter does not implement `Serialize`.
fn jiter_tree(value: &jiter::JsonValue<'_>) -> Value {
    match value {
        jiter::JsonValue::Null => Value::Null,
        jiter::JsonValue::Bool(value) => Value::Bool(*value),
        jiter::JsonValue::Int(value) => Value::from(*value),
        jiter::JsonValue::Float(value) => Value::from(*value),
        jiter::JsonValue::Str(value) => Value::from(value.as_ref()),
        jiter::JsonValue::Array(values) => values.iter().map(jiter_tree).collect(),
        jiter::JsonValue::Object(members) => Value::Object(
            members
                .iter()
                .map(|(key, value)| (key.to_string(), jiter_tree(value)))
                .collect(),
        ),
    }
}

fn write_row<V>(
    group: &mut Group<'_>,
    class: &'static Class,
    codec: &str,
    trees: Vec<V>,
    write: impl FnMut(&V, &mut Vec<u8>) -> bool,
) {
    group.row(
        codec,
        class.name,
        (trees, Vec::new(), write),
        |(trees, text, write)| {
            trees
                .iter()
                .filter(|tree| {
                    text.clear();
                    write(tree, text)
                })
                .count()
        },
        |written, (trees, text, write)| {
            assert_eq!(
                written,
                class.docs.len(),
                "{codec} writes each document of {}",
                class.name
            );
            for (index, (tree, expected)) in trees.iter().zip(&class.trees).enumerate() {
                text.clear();
                assert!(write(tree, text), "{codec} writes document {index} of {}", class.name);
                let back: Value = serde_json::from_slice(text).expect("the text is JSON");
                assert!(
                    back == *expected,
                    "{codec} writes document {index} of {} as the text of the reference tree",
                    class.name
                );
            }
        },
    );
}

enum Node {
    Null,
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Text(String),
    Array(Vec<Node>),
    Object(Vec<(String, Node)>),
}

impl Node {
    fn of(tree: &Value) -> Self {
        match tree {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(*value),
            Value::Number(number) => match (number.as_u64(), number.as_i64()) {
                (Some(value), _) => Self::Unsigned(value),
                (None, Some(value)) => Self::Signed(value),
                (None, None) => Self::Float(number.as_f64().expect("serde_json holds each number")),
            },
            Value::String(text) => Self::Text(text.clone()),
            Value::Array(items) => Self::Array(items.iter().map(Self::of).collect()),
            Value::Object(members) => Self::Object(
                members
                    .iter()
                    .map(|(key, value)| (key.clone(), Self::of(value)))
                    .collect(),
            ),
        }
    }

    fn put(&self, place: Place<'_>) -> Result<(), json::Error> {
        match self {
            Self::Null => place.null(),
            Self::Bool(value) => place.bool(*value),
            Self::Unsigned(value) => place.u64(*value),
            Self::Signed(value) => place.i64(*value),
            Self::Float(value) => place.f64(*value),
            Self::Text(text) => place.str(text),
            Self::Array(items) => place.array(|array| items.iter().try_for_each(|item| item.put(array.item()))),
            Self::Object(members) => place.object(|object| {
                members
                    .iter()
                    .try_for_each(|(key, value)| value.put(object.member(key)))
            }),
        }
    }
}

fn write_row_rscodec(group: &mut Group<'_>, class: &'static Class) {
    let trees: Vec<Node> = class.trees.iter().map(Node::of).collect();
    let out = Aligned::new(&vec![
        0;
        2 * class.docs.iter().map(|doc| doc.len()).max().unwrap_or(0)
            + 1024
    ]);
    group.row(
        "rscodec",
        class.name,
        (trees, out),
        |(trees, out)| {
            trees
                .iter()
                .filter(|tree| json::write(out, |place| tree.put(place)).is_ok())
                .count()
        },
        |written, (trees, out)| {
            assert_eq!(
                written,
                class.docs.len(),
                "rscodec writes each document of {}",
                class.name
            );
            for (index, (tree, expected)) in trees.iter().zip(&class.trees).enumerate() {
                let len = json::write(out, |place| tree.put(place)).expect("the slice holds the document");
                let back: Value = serde_json::from_slice(&out[..len]).expect("the text is JSON");
                assert!(
                    back == *expected,
                    "rscodec writes document {index} of {} as the text of the reference tree",
                    class.name
                );
            }
        },
    );
}

fn write(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "json-write");
    for class in classes() {
        group.throughput(Throughput::Bytes(class.bytes.try_into().expect("a length fits u64")));
        let g = &mut group;
        write_row(g, class, "serde_json", class.trees.clone(), |tree, text| {
            serde_json::to_writer(text, tree).is_ok()
        });
        let trees = class
            .docs
            .iter()
            .map(|doc| sonic_rs::from_slice::<sonic_rs::Value>(doc).expect("valid"));
        write_row(g, class, "sonic-rs", trees.collect(), |tree, text| {
            sonic_rs::to_writer(text, tree).is_ok()
        });
        let mut copies: Vec<_> = class.docs.iter().map(|doc| doc.to_vec()).collect();
        let trees = copies
            .iter_mut()
            .map(|copy| simd_json::to_borrowed_value(copy).expect("valid"));
        write_row(g, class, "simd-json", trees.collect(), |tree, text| {
            tree.write(text).is_ok()
        });
        let trees = class
            .docs
            .iter()
            .map(|doc| flexon::parse::<_, flexon::Value<'_>>(&doc[..]).expect("valid"));
        write_row(g, class, "flexon", trees.collect(), |tree, text| {
            flexon::to_writer(text, tree).is_ok()
        });
        write_row_rscodec(g, class);
    }
}

fn print_row(
    group: &mut Group<'_>,
    input: &str,
    values: &[f64],
    codec: &str,
    mut print: impl FnMut(&[f64], &mut String),
) {
    group.row(
        codec,
        input,
        String::new(),
        |text| {
            text.clear();
            print(values, text);
            text.len()
        },
        |_, text| {
            let mut back = text
                .split_terminator(',')
                .map(|value| value.parse::<f64>().map(f64::to_bits));
            for (index, value) in values.iter().enumerate() {
                assert_eq!(
                    back.next(),
                    Some(Ok(value.to_bits())),
                    "{codec} prints value {index} of {input} so that it parses back"
                );
            }
            assert_eq!(back.next(), None, "{codec} prints only the values of {input}");
        },
    );
}

fn f64_inputs() -> [(&'static str, Vec<f64>); 2] {
    fn floats(tree: &Value, out: &mut Vec<f64>) {
        match tree {
            Value::Number(number) if number.is_f64() => out.extend(number.as_f64()),
            Value::Array(values) => values.iter().for_each(|value| floats(value, out)),
            Value::Object(members) => members.values().for_each(|value| floats(value, out)),
            _ => {}
        }
    }
    let mut canada = Vec::new();
    for class in classes().iter().filter(|class| class.name == "canada") {
        class.trees.iter().for_each(|tree| floats(tree, &mut canada));
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let random = core::iter::repeat_with(|| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f64::from_bits(state.wrapping_mul(0x2545_f491_4f6c_dd1d))
    })
    .filter(|value| value.is_finite())
    .take(RANDOM_F64_COUNT)
    .collect();
    [("canada", canada), ("random", random)]
}

fn print_f64(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "f64-print");
    for (input, values) in &f64_inputs() {
        group.throughput(Throughput::Elements(values.len().try_into().expect("a count fits u64")));
        let g = &mut group;
        let out = Aligned::new(&vec![0; values.len() * json::F64_BYTES_MAX + values.len() + 2]);
        g.row(
            "rscodec",
            input,
            out,
            |out| {
                json::write(out, |place| {
                    place.array(|array| values.iter().try_for_each(|&value| array.item().f64(value)))
                })
            },
            |len, out| {
                let len = len.expect("the slice holds the array");
                let text = core::str::from_utf8(&out[..len]).expect("the writer writes ASCII");
                let inner = text
                    .strip_prefix('[')
                    .and_then(|text| text.strip_suffix(']'))
                    .expect("an array");
                let back: Vec<u64> = inner
                    .split(',')
                    .map(|value| value.parse::<f64>().map(f64::to_bits).expect("a number"))
                    .collect();
                assert!(
                    back == values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "rscodec prints each value of {input} so that it parses back"
                );
            },
        );
        print_row(g, input, values, "zmij", |values, text| {
            let mut buffer = zmij::Buffer::new();
            for &value in values {
                text.push_str(buffer.format_finite(value));
                text.push(',');
            }
        });
        print_row(g, input, values, "ryu", |values, text| {
            let mut buffer = ryu::Buffer::new();
            for &value in values {
                text.push_str(buffer.format_finite(value));
                text.push(',');
            }
        });
        // `{:?}` is the shortest form that parses back; `{}` writes every digit of a large exponent.
        print_row(g, input, values, "core-fmt", |values, text| {
            for value in values {
                write!(text, "{value:?},").expect("a String takes each text");
            }
        });
    }
}

fn numbers(doc: &[u8]) -> Vec<json::Number<'_>> {
    let mut scratch = vec![Slot::EMPTY; SLOTS];
    let mut numbers = Vec::new();
    json::visit(doc, &LIMITS, Profile::Rfc8259, &mut scratch, |token| {
        if let Token::Number(number) = token {
            numbers.push(number);
        }
        Ok(())
    })
    .expect("rscodec reads the document");
    numbers
}

fn read_f64(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "f64-read");
    let canada = &classes()
        .iter()
        .find(|class| class.name == "canada")
        .expect("canada is a class")
        .docs[0];
    let [_, (_, random)] = f64_inputs();
    let mut text = String::from("[");
    for value in &random {
        write!(text, "{value:?},").expect("a String takes each text");
    }
    text.pop();
    text.push(']');
    for (input, doc) in [("canada", &canada[..]), ("random", text.as_bytes())] {
        let numbers = numbers(doc);
        let texts: Vec<_> = numbers.iter().map(json::Number::as_str).collect();
        let expected: Vec<u64> = texts
            .iter()
            .map(|text| text.parse::<f64>().expect("core reads each number").to_bits())
            .collect();
        let check = move |codec: &str, values: &[f64]| {
            assert!(
                values.iter().map(|value| value.to_bits()).eq(expected.iter().copied()),
                "{codec} reads each number of {input} to the value of core"
            );
        };
        let check_core = check.clone();
        group.throughput(Throughput::Elements(
            numbers.len().try_into().expect("a count fits u64"),
        ));
        let g = &mut group;
        g.row(
            "rscodec",
            input,
            vec![0.0; numbers.len()],
            |values| {
                for (value, number) in values.iter_mut().zip(&numbers) {
                    *value = number.to_f64().unwrap_or(f64::NAN);
                }
            },
            |(), values| check("rscodec", values),
        );
        g.row(
            "core",
            input,
            vec![0.0; texts.len()],
            |values| {
                for (value, text) in values.iter_mut().zip(&texts) {
                    *value = text.parse().unwrap_or(f64::NAN);
                }
            },
            |(), values| check_core("core", values),
        );
    }
}

fn main() {
    common::run(&[read, typed, write, print_f64, read_f64]);
}
