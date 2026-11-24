# Read and write JSON

rscodec reads one JSON document from a slice in memory. The reader is a pull reader: your code asks
for each token or value in order, and the reader checks each byte that it passes. It allocates no
memory, and no code recurses, so a deep document does not grow the stack.

The writer writes compact JSON into a slice that you lend. It allocates no memory either.

The `json` feature is in development: a differential fuzz campaign of the reader has not run yet.
Each change can change the API. The crate needs Rust 1.101, which is a nightly toolchain today. The
repository tests `nightly-2026-11-02`.

## Select the feature

```toml
[dependencies]
rscodec = { git = "https://github.com/loadingalias/rscodec", features = ["json"] }
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
| `Value::number` | A number, as a `Number`. Its accessors give its value (see [Read numbers](#read-numbers)). |
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

`read` selects the kernel of the CPU once. The reader calls the kernel once for each block of 64
bytes, so the code of the kernel stays out of the loop of the decoder. When the build fixes the
kernel, from `-C target-feature`, `-C target-cpu`, or a target without `std`, the call is direct.
When the CPU selects the kernel at run time, the call goes through a function pointer.

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

## Read numbers

`Value::number` and `Token::Number` give a `Number`. It borrows the text of the number, whose
grammar the reader checked. Each accessor gives the exact value, or it refuses:

| Method | Gives | Refuses |
| --- | --- | --- |
| `Number::to_i64`, `Number::to_u64` | The value of an integer: a number with no fraction and no exponent. `-0` gives 0. | `Type` for a fraction or an exponent, also `1.0` and `1e2`. `Range` for a value outside the type. |
| `Number::to_decimal` | A `Decimal` with the scale of the text: `negative`, a `u128` `coefficient`, and an `i32` `exponent`. `1.50` gives 150 × 10^−2, and `-1.5e3` gives −15 × 10^2. | `Range` for more than 38 significant digits, or for an exponent outside `i32`. |
| `Number::to_f64` | The nearest `f64`, with a tie to the even significand. `-0` gives `-0.0`. A nonzero value that rounds to zero gives zero with its sign. | `Range` if the magnitude rounds to infinity, for example `1e400`. |
| `Number::class` | `NumberClass::Integer`, `Decimal` (a fraction and no exponent), or `Float` (an exponent), to choose the accessor. | Nothing. |
| `Number::as_str` | The text, for example `-1.5e3`. Keep it for a number that no accessor holds. | Nothing. |

Each refusal is at the offset of the number in the document, so `?` in the closure of `read` gives
it. This decoder reads an integer, an amount with its scale, and a coordinate:

```rust
use rscodec::json::{self, Decimal, ErrorKind, Limits, Profile, Slot};

const ORDER: Limits = Limits {
    input_bytes_max: 8192,
    depth_max: 8,
    string_bytes_max: 4096,
    members_max: 32,
    elements_max: 32,
    number_digits_max: 32,
};

let order = br#"{"expires":1767225600,"amount":12.50,"lat":52.5163}"#;
let mut scratch = [Slot::EMPTY; 8];
let (expires, amount, lat) = json::read(order, &ORDER, Profile::Rfc8259, &mut scratch, #[inline(always)] |reader| {
    let (mut expires, mut amount, mut lat) = (None, None, None);
    let mut members = reader.value().members()?;
    while let Some((key, value)) = members.next()? {
        if key == "expires" {
            expires = Some(value.number()?.to_u64()?);
        } else if key == "amount" {
            amount = Some(value.number()?.to_decimal()?);
        } else if key == "lat" {
            lat = Some(value.number()?.to_f64()?);
        }
    }
    Ok((expires, amount, lat))
})?;
assert_eq!(expires, Some(1_767_225_600));
assert_eq!(amount, Some(Decimal { negative: false, coefficient: 1250, exponent: -2 }));
assert_eq!(lat, Some(52.5163));

// An integer accessor refuses a fraction, at the offset of the number.
let late = br#"{"expires":1.5}"#;
let error = json::read(late, &ORDER, Profile::Rfc8259, &mut scratch, |reader| {
    let mut members = reader.value().members()?;
    members.next()?.map(|(_, value)| value.number()?.to_u64()).transpose()
})
.unwrap_err();
assert_eq!((error.kind(), error.offset()), (ErrorKind::Type, 11));
# Ok::<(), json::Error>(())
```

`to_f64` reads a number of up to 19 significant digits with unrounded scaling (Russ Cox, 2026),
which is proved exact. It reads a number of more digits with the parser of `core`.
`number_digits_max` bounds that work.

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

## Write a document

`json::write` takes a slice and a closure, and returns the length of the text. The closure gets a
`Place`: the place of one value. Each method of `Place` writes one value and takes the place, so
each place gets at most one value:

- `str`, `i64`, `u64`, `f64`, `bool`, and `null` write a scalar.
- `base64url` writes bytes as a base64url string with no padding. It needs the `base64` feature
  too.
- `object` and `array` take a closure. `Object::member(key)` gives the place of the value of a
  member, and `Array::item()` gives the place of the next item.

The writer writes each `,`, `:`, quote, and escape. A place that your code drops writes nothing, so
the text is always valid JSON. This closure writes the options of a WebAuthn registration ceremony:

```rust
use rscodec::json::{self, ErrorKind};

let (user_id, challenge) = (b"user-4711", [0x9f; 16]);
let options = |out: &mut [u8]| {
    json::write(out, |options| {
        options.object(|options| {
            options.member("rp").object(|rp| {
                rp.member("id").str("example.com")?;
                rp.member("name").str("Example")
            })?;
            options.member("user").object(|user| {
                user.member("id").base64url(user_id)?;
                user.member("name").str("alice@example.com")?;
                user.member("displayName").str("Alice \"Al\" Smith")
            })?;
            options.member("challenge").base64url(&challenge)?;
            options.member("pubKeyCredParams").array(|params| {
                // ES256 and RS256.
                for alg in [-7, -257] {
                    params.item().object(|param| {
                        param.member("type").str("public-key")?;
                        param.member("alg").i64(alg)
                    })?;
                }
                Ok(())
            })?;
            options.member("timeout").u64(60_000)?;
            options.member("attestation").str("none")
        })
    })
};
let mut out = [0; 512];
let len = options(&mut out)?;
assert_eq!(
    &out[..len],
    concat!(
        r#"{"rp":{"id":"example.com","name":"Example"},"#,
        r#""user":{"id":"dXNlci00NzEx","name":"alice@example.com","displayName":"Alice \"Al\" Smith"},"#,
        r#""challenge":"n5-fn5-fn5-fn5-fn5-fnw","#,
        r#""pubKeyCredParams":[{"type":"public-key","alg":-7},{"type":"public-key","alg":-257}],"#,
        r#""timeout":60000,"attestation":"none"}"#,
    )
    .as_bytes(),
);

// A slice that is too short refuses, and the text that fits stays valid. Here the object does
// not fit, so nothing is written.
let error = options(&mut out[..len - 1]).unwrap_err();
assert_eq!((error.kind(), error.offset()), (ErrorKind::Full, 0));
# Ok::<(), json::Error>(())
```

The forms of the values:

- A string gets the escapes of RFC 8785, which are also valid RFC 8259: `\"`, `\\`, `\b`, `\f`,
  `\n`, `\r`, `\t`, and `\u00xx` with lowercase hex digits for each other byte under 0x20. Other
  characters stay as they are.
- `f64` writes the shortest digits that read back as the same value, in the layout of `{:?}` of
  `core::fmt`: `1.0`, `-0.0`, `0.0001`, `1e16`, and `1.5e-7`. JSON has no NaN or infinity, so they
  refuse with `ErrorKind::NonFinite`. For an `f32`, write `f64::from(value)`.
- The writer does not check the keys of an object. The reader refuses a duplicate key, so write each
  key once.

Each call writes all of its bytes or none. After an error, the written length is the length before
the call, also when an `object` or `array` closure fails. To write JSON Lines, write each record with
its own `json::write` call, and put a `\n` after it.

## Size the slice

The writer needs no more than the length of the text. Add the parts:

| Part | Bytes |
| --- | --- |
| A string or a key | `json::string_len(text)`, quotes and escapes included |
| A key | 1 more, for the `:` |
| `base64url(bytes)` | `base64::Profile::Url.encoded_len(bytes.len())` and 2 quotes |
| `i64`, `u64` | at most `json::I64_BYTES_MAX` or `json::U64_BYTES_MAX` (20) |
| `f64` | at most `json::F64_BYTES_MAX` (24) |
| `true`, `false`, `null` | 4, 5, 4 |
| An object or an array | 2, and 1 for each `,` between its items |

When the slice ends, the call that does not fit refuses with `ErrorKind::Full`. The offset is the
written length. Lend a longer slice, or take the written bytes and continue in a new slice.

## Errors

Each refusal is a `json::Error`: `kind()` gives its `ErrorKind`, and `offset()` gives its offset.
For the reader, the offset is in the document, and the errors come in the order of their offsets:
`read` gives the error with the lowest offset. For the writer, the offset is the written length. A
decoder that refuses a valid document, for example one that lacks a member that its type needs,
makes its refusal with `json::Error::new(kind, offset)`.

| `ErrorKind` | Cause | Recovery |
| --- | --- | --- |
| `Syntax` | RFC 8259 does not allow the byte at the offset there, for example a byte order mark, `NaN`, or a byte after the value. | Correct the document. |
| `End` | The document ends inside its value. The offset is its length. | Read the complete document. |
| `Utf8` | The bytes at the offset are not UTF-8. The offset is the `valid_up_to` of `core::str::from_utf8`. At one offset, this error comes before `Syntax`. | Send UTF-8. |
| `Surrogate` | The escape at the offset is a surrogate without its pair. | Escape the character with its pair. |
| `Noncharacter` | `IJson`: a noncharacter at the offset. | Read with `Rfc8259`, or remove the character. |
| `Number` | `IJson`: binary64 does not hold the number at the offset. | Read with `Rfc8259`, or send the number as a string. |
| `Range` | The value of the number at the offset does not fit the type of the accessor. | Read with `to_decimal` or `to_f64`, or keep the text of `as_str`. |
| `Duplicate` | The key at the offset equals an earlier key of its object, after unescape. | Remove one of the members. |
| `Limit(limit)` | The item at the offset passes the limit `limit` of `Limits`. | Raise the limit, if the document is trusted. |
| `Scratch` | The scratch has no slot for the container or key at the offset. | Lend a longer scratch. |
| `Output` | The buffer of `Str::decode` is too short. The offset is the decoded length, which `Str::len` also gives. | Lend a buffer of that length. |
| `Type` | The value at the offset does not have the type that the call reads. `to_i64` and `to_u64` give it for a number with a fraction or an exponent. The writer gives it at offset 0 when its closure writes no value. | Read with `Value::kind` or `Number::class` first, or refuse the document. Write a value at the place. |
| `Full` | The writer: the slice ends before the end of the value that the call writes. | Lend a longer slice, or continue in a new slice. |
| `NonFinite` | The writer: the call writes NaN or an infinity, which JSON has no number for. | Write a finite value, or `null`. |

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
