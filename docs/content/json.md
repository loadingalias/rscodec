# Read JSON

rscodec reads one JSON document from a slice in memory. The reader is a pull reader: your code asks
for each token or value in order, and the reader checks each byte that it passes. It allocates no
memory, and no code recurses, so a deep document does not grow the stack.

The `json` feature is in development. The first release with it is not on crates.io yet. This page
gives the reader. Exact number accessors and the writer come later. Now a number gives its text.

## Select the feature

```toml
[dependencies]
rscodec = { version = "0.1.0-alpha.2", features = ["json"] }
```

The `std` feature is on by default. Without it, the reader needs only `core`, and it selects its
kernel at compile time from `-C target-feature`.

## Read a document

`json::read` takes the document, the limits, the profile, a scratch, and a closure. The closure
gets a `Reader`. When the closure returns, `read` reads the rest of the document with each check.
Thus `read` gives `Ok` only for a valid document, also when the closure reads only part of it.

This decoder reads a JOSE header with the strict walk. `Value::members` opens the object, and each
call of `Members::next` gives the next member, in order. The loop reads the values that it needs,
and the walk skips the others with each check.

```rust
use rscodec::json::{self, Limits, Profile, Slot, Str};

const HEADER: Limits = Limits {
    input_bytes_max: 8192,
    depth_max: 8,
    string_bytes_max: 4096,
    members_max: 32,
    elements_max: 32,
    number_digits_max: 32,
};

/// The `alg` and `kid` members of a JOSE header.
fn alg_and_kid(header: &[u8]) -> Result<(Option<Str<'_>>, Option<Str<'_>>), json::Error> {
    let mut scratch = [Slot::EMPTY; 16];
    json::read(header, &HEADER, Profile::IJson, &mut scratch, #[inline(always)] |reader| {
        let (mut alg, mut kid) = (None, None);
        let mut members = reader.value().members()?;
        while let Some((key, value)) = members.next()? {
            if key == "alg" {
                alg = Some(value.str()?);
            } else if key == "kid" {
                kid = Some(value.str()?);
            }
        }
        Ok((alg, kid))
    })
}

let header = br#"{"alg":"ES256","kid":"key-1","crit":["b64"],"b64":false}"#;
let (alg, kid) = alg_and_kid(header)?;
assert!(alg.is_some_and(|alg| alg == "ES256"));
let mut buf = [0; 256];
if let Some(kid) = kid {
    assert_eq!(kid.decode(&mut buf)?, "key-1");
}
# Ok::<(), json::Error>(())
```

A `Str` borrows the document. `==` compares it with a `&str` after unescape, so the key
`"\u0061lg"` matches `"alg"` too. `Str::decode` gives the string: it borrows the document if the
string has no escape, and it writes a string with escapes into the buffer that you lend.
`Str::len` gives the length that the buffer needs.

A `Value` reads one value, once. Each of its methods takes the value:

| Method | Reads |
| --- | --- |
| `Value::members` | The `{` of an object. `Members::next` gives the key and the value of each member. |
| `Value::elements` | The `[` of an array. `Elements::next` gives each element. |
| `Value::str` | A string, as a `Str`. |
| `Value::number` | A number, as a `Number`. `Number::as_str` gives its text. |
| `Value::bool` | `true` or `false`. |
| `Value::null` | `null`. |
| `Value::skip` | A value of any type, with each check. |
| `Value::kind` | Nothing. It gives the `Kind` of the value, to choose the method. |

A `Value` from `next` is for one pass of the loop. The next call of `next` skips what the loop did
not read of it: a value that the loop left, or the rest of a container that the loop opened. Each
skip gets each check of the profile. After the end of its container, `next` gives `None`.

Write a decoder as a loop in the closure of `read`, and as helper functions that take a `Value`.
Mark the closure and each helper `#[inline(always)]`, so the reader keeps its state in registers.
The compiler does not inline a large closure or helper without it, also when `read` calls it at one
place. A decoder that the compiler does not inline still works, but slower, because the reader is
then in memory.

`read` selects the kernel of the CPU once. When the build fixes the kernel, from
`-C target-feature`, `-C target-cpu`, or a target without `std`, the kernel inlines into the reader.
When the CPU selects it at run time, the reader calls it once for each block of 64 bytes.

A method for another type gives `ErrorKind::Type` at the value and reads nothing. The JWT `aud`
claim can be a string or an array, so read it with `kind` first:

```rust
use rscodec::json::{self, Kind, Str, Value};

/// Each audience of the `aud` claim, which has one or more.
#[inline(always)]
fn audience<'a>(mut value: Value<'_, 'a, '_>, out: &mut Vec<Str<'a>>) -> Result<(), json::Error> {
    if value.kind()? == Kind::String {
        out.push(value.str()?);
        return Ok(());
    }
    let mut elements = value.elements()?;
    while let Some(value) = elements.next()? {
        out.push(value.str()?);
    }
    Ok(())
}
```

## Read the tokens

`Reader::token` gives each `Token` in order, and `None` after the end of the document. A `,`, a
`:`, and whitespace are no tokens. This loop reads the version and the checksum of each line of a
Cargo sparse-index file:

```rust
use rscodec::json::{self, Limits, Profile, Slot, Token};

const INDEX_LINE: Limits = Limits {
    input_bytes_max: 1 << 20,
    depth_max: 8,
    string_bytes_max: 1 << 16,
    members_max: 64,
    elements_max: 4096,
    number_digits_max: 32,
};

let line = br#"{"name":"rscodec","vers":"0.1.0","deps":[],"cksum":"0f1e","features":{},"yanked":false}"#;
let mut scratch = [Slot::EMPTY; 32];
let (vers, cksum) = json::read(line, &INDEX_LINE, Profile::Rfc8259, &mut scratch, #[inline(always)] |reader| {
    let (mut vers, mut cksum, mut last_key) = (None, None, None);
    while let Some(token) = reader.token()? {
        match token {
            Token::Key(key) => last_key = Some(key),
            Token::Str(text) if last_key.is_some_and(|key| key == "vers") => vers = Some(text),
            Token::Str(text) if last_key.is_some_and(|key| key == "cksum") => cksum = Some(text),
            _ => {}
        }
    }
    Ok((vers, cksum))
})?;
assert!(vers.is_some_and(|vers| vers == "0.1.0") && cksum.is_some_and(|cksum| cksum == "0f1e"));
# Ok::<(), json::Error>(())
```

The keys of a nested object are also `Token::Key`. Keep a depth from the start and end tokens if
your format can nest a key of the same name.

## Select the profile

| Profile | Rules |
| --- | --- |
| `Profile::Rfc8259` | RFC 8259. A key that equals an earlier key of its object refuses. A surrogate escape without its pair refuses, because a Rust `str` cannot hold it. |
| `Profile::IJson` | `Rfc8259` and I-JSON (RFC 7493). A noncharacter refuses, as UTF-8 and as an escape. A number refuses if binary64 does not hold it. |

The number rule of `IJson` refuses four numbers:

- An integer outside ±(2^53 − 1). An integer is a number with no fraction and no exponent.
- A number with more than 17 significant digits. The significant digits go from the first nonzero
  digit to the last. Thus `1.50000000000000000000` has two.
- A number whose magnitude rounds to infinity, for example `1e400`.
- A nonzero number that rounds to zero, for example `1e-400`. Zero, also `-0`, reads.

## Set the limits

`Limits` has no default. A limit that fits a header of 150 bytes refuses a file of 2 MB, and a
limit that fits the file lets a hostile header use memory and time. State each limit for the
documents that you read.

| Field | Bounds | Example for a JOSE header |
| --- | --- | --- |
| `input_bytes_max` | The bytes of the document. | 8192 |
| `depth_max` | The containers that one value can be in. `[]` has depth 1. | 8 |
| `string_bytes_max` | The bytes of one string or key between its quotes, as written. An escape counts with each of its bytes. | 4096 |
| `members_max` | The members of one object. | 32 |
| `elements_max` | The elements of one array. | 32 |
| `number_digits_max` | The digits of one number: its integer, its fraction, and its exponent. | 32 |

Each limit refuses the first item past it with `ErrorKind::Limit` and the offset of that item. For
example, with `members_max: 2`, the object `{"a":1,"b":2,"c":3}` refuses at the key `"c"`. A number
past `number_digits_max` gives the limit error and not the number error of `IJson`, because the
reader does not read the rest of the number.

## Size the scratch

The scratch is a slice of `Slot` that you lend. The reader uses one slot for each open container and
one for each key of each open object. It uses the keys for the duplicate check, with no hash.

- A JOSE header of one object with 8 members needs 9 slots.
- An array of depth 32 needs 32 slots.
- A document with the object `{"a":{"b":{}}}`, open to its depth, needs 5 slots: 3 containers and
  the keys `a` and `b`.

A scratch that is too short refuses with `ErrorKind::Scratch` at the container or key that needs a
slot. For a document of depth `d` with at most `m` members in each object, `d × (m + 1)` slots are
enough.

## Errors

Each refusal is a `json::Error` with a `kind` and an `offset` in the document. The errors come in
the order of their offsets: `read` gives the error with the lowest offset.

| `ErrorKind` | Cause | Recovery |
| --- | --- | --- |
| `Syntax` | RFC 8259 does not allow the byte at the offset there, for example a byte order mark, `NaN`, or a byte after the value. | Correct the document. |
| `End` | The document ends inside its value. The offset is its length. | Read the complete document. |
| `Utf8` | The bytes at the offset are not UTF-8. The offset is the `valid_up_to` of `core::str::from_utf8`. At one offset, this error comes before `Syntax`. | Send UTF-8. |
| `Surrogate` | The escape at the offset is a surrogate without its pair. | Escape the character with its pair. |
| `Noncharacter` | `IJson`: a noncharacter at the offset. | Read with `Rfc8259`, or remove the character. |
| `Number` | `IJson`: binary64 does not hold the number at the offset. | Read with `Rfc8259`, or send the number as a string. |
| `Duplicate` | The key at the offset equals an earlier key of its object, after unescape. | Remove one of the members. |
| `Limit(limit)` | The item at the offset passes the limit `limit` of `Limits`. | Raise the limit, if the document is trusted. |
| `Scratch` | The scratch has no slot for the container or key at the offset. | Lend a longer scratch. |
| `Output` | The buffer of `Str::decode` is too short. The offset is the decoded length, which `Str::len` also gives. | Lend a buffer of that length. |
| `Type` | The value at the offset does not have the type that the call reads. | Read with `Value::kind` first, or refuse the document. |

## JSONTestSuite

The tests read each file of JSONTestSuite (commit `1ef36fa`). Each `n_` file refuses in each
profile. Each `y_` file reads in each profile, with these exceptions:

| File | `Rfc8259` | `IJson` |
| --- | --- | --- |
| `y_object_duplicated_key.json`, `y_object_duplicated_key_and_value.json` | `Duplicate` | `Duplicate` |
| `y_string_escaped_noncharacter.json`, `y_string_last_surrogates_1_and_2.json`, `y_string_nonCharacterInUTF-8_U+10FFFF.json`, `y_string_nonCharacterInUTF-8_U+FFFF.json`, `y_string_unicode_U+10FFFE_nonchar.json`, `y_string_unicode_U+1FFFE_nonchar.json`, `y_string_unicode_U+FDD0_nonchar.json`, `y_string_unicode_U+FFFE_nonchar.json` | reads | `Noncharacter` |

The `i_` files are for each reader to decide. rscodec gives these results:

| Files | `Rfc8259` | `IJson` |
| --- | --- | --- |
| The 10 `i_number_` files: a huge exponent, an overflow, an underflow, and integers past ±(2^53 − 1) | reads, if `number_digits_max` holds the digits | `Number` |
| The 10 `i_string_` and `i_object_` files with a surrogate escape without its pair | `Surrogate` | `Surrogate` |
| The 11 `i_string_` files that are not UTF-8, `i_string_UTF-16LE_with_BOM.json` among them | `Utf8` | `Utf8` |
| `i_string_utf16BE_no_BOM.json`, `i_string_utf16LE_no_BOM.json` | `Syntax` (a NUL byte) | `Syntax` |
| `i_structure_500_nested_arrays.json` | reads, if `depth_max` is 500 or more | as `Rfc8259` |
| `i_structure_UTF-8_BOM_empty_object.json` | `Syntax` at offset 0 (a byte order mark) | `Syntax` |
