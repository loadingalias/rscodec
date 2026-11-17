//! JSON: read each document into the tree of each crate, write that tree back as compact text, and
//! print f64 values as text.
//!
//! A cell is one class of documents, and one call reads, or writes, each document of the class.
//! serde_json reads each document into the reference tree once. Before timing, each read must give
//! that tree, and serde_json must read the text of each write back into it. Each printed f64 must
//! parse back to its bits.

mod common;

use core::{fmt::Write as _, hint::black_box};
use std::{fs, path::Path, sync::OnceLock};

use common::Group;
use criterion::{Criterion, Throughput};
use serde_json::Value;
use simd_json::prelude::Writable as _;

/// The classes of documents: a name, and a file or a directory in `benches/data`. Each line of a
/// `.jsonl` file is one document. benches/data/README.md fetches the files in `fetched`.
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

/// The number of random f64 values that the printers write.
const RANDOM_F64_COUNT: usize = 10_000;

struct Class {
    name: &'static str,
    docs: Vec<&'static [u8]>,
    trees: Vec<Value>,
    bytes: usize,
}

/// The classes, read once for each process. The documents live until the process ends, because the
/// trees of most crates borrow from them.
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
    for file in files {
        let text: &'static [u8] = fs::read(&file).expect("a document file is readable").leak();
        if file.extension().is_some_and(|extension| extension == "jsonl") {
            docs.extend(text.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()));
        } else {
            docs.push(text);
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

/// A row that reads each document of `class` with `read`, which returns whether the read succeeded.
/// `tree` reads one document in the same way and gives its serde_json tree, for the check.
fn read_row<S>(
    group: &mut Group<'_>,
    class: &'static Class,
    codec: &str,
    state: S,
    mut read: impl FnMut(&mut S, &'static [u8]) -> bool,
    mut tree: impl FnMut(&mut S, &'static [u8]) -> Value,
) {
    group.row(
        codec,
        class.name,
        state,
        |state| class.docs.iter().filter(|doc| read(state, doc)).count(),
        |read_count, state| {
            assert_eq!(
                read_count,
                class.docs.len(),
                "{codec} reads each document of {}",
                class.name
            );
            for (index, (doc, expected)) in class.docs.iter().zip(&class.trees).enumerate() {
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
        // simd-json reads in place, so each read first copies the document into a buffer of the row.
        read_row(
            g,
            class,
            "simd-json",
            (Vec::new(), simd_json::Buffers::default()),
            |(copy, buffers), doc| {
                copy.clear();
                copy.extend_from_slice(doc);
                black_box(simd_json::to_borrowed_value_with_buffers(copy, buffers)).is_ok()
            },
            |(copy, buffers), doc| {
                copy.clear();
                copy.extend_from_slice(doc);
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

/// A row that writes each tree in `trees` with `write`, which returns whether the write succeeded.
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
        let trees = class
            .docs
            .iter()
            .map(|doc| simd_json::to_borrowed_value(doc.to_vec().leak()).expect("valid"));
        write_row(g, class, "simd-json", trees.collect(), |tree, text| {
            tree.write(text).is_ok()
        });
        let trees = class
            .docs
            .iter()
            .map(|doc| flexon::parse::<_, flexon::Value<'_>>(*doc).expect("valid"));
        write_row(g, class, "flexon", trees.collect(), |tree, text| {
            flexon::to_writer(text, tree).is_ok()
        });
    }
}

/// A row that prints `values` with `print`, which appends each value and a comma to its text.
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

/// The floating-point numbers of canada.json, and finite f64 values from random bits.
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

fn main() {
    common::run(&[read, write, print_f64]);
}
