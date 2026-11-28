# Read and write JSON

```toml
[dependencies]
rscodec = { version = "0.1.0", features = ["json"] }
```

## Read a document

The reader branches on input and uses public-data timing. Pass one document, its limits, a profile, scratch slots, and a closure to `json::read`.
Use `reader.value().members()?` for an object or `.elements()?` for an array. Call `next()?` to read each item.
Read a value with `str`, `number`, `bool`, or `null`. Use `kind` to inspect its type.

The reader validates skipped values and the remaining input when the closure succeeds.
Returning early with `Ok` still rejects a duplicate key or trailing garbage. A caller error stops the read.

The [JOSE header example](../../examples/jose_header.rs) reads `alg` and checks the rest of the header.
Key comparison decodes escapes, so `"\u0061lg"` matches `"alg"`.

```sh
cargo run --locked --example jose_header --features json
```

## Read the tokens

`json::visit` emits keys, scalar values, and both ends of each container in input order.
Strings and numbers borrow the input. Track container depth when a key must belong to a specific object.
Use token results only after the visit succeeds. Late errors can invalidate an earlier token prefix.

The [Cargo sparse-index example](../../examples/sparse_index.rs) keeps two top-level strings.
A nested feature named `cksum` cannot replace the package checksum.

```sh
cargo run --locked --example sparse_index --features json
```

`Str::borrowed()` borrows strings that contain no escapes.
`Str::decode(&mut buf)` also handles escapes. Use `Str::len()` for the decoded byte count.

## Read numbers

| Method | Result |
| --- | --- |
| `Number::as_str` | Original number text |
| `Number::class` | Integer, decimal fraction, or exponent form |
| `Number::to_i64`, `to_u64` | Integer value. A fraction or exponent gives `Type`. A value outside the type gives `Range`. |
| `Number::to_decimal` | Sign, `u128` coefficient, and `i32` exponent. `1.50` gives 150 × 10⁻². More than 38 coefficient digits after leading zeros or a resulting exponent outside `i32` gives `Range`. |
| `Number::to_f64` | Nearest binary64 value, with ties to even. Underflow preserves the sign of zero. Overflow gives `Range`. |

## Select the profile

Both profiles reject duplicate decoded keys within an object and unpaired surrogate escapes.
`Profile::Rfc8259` accepts any JSON number within the caller's digit limit. Its accessors check the requested representation.

`Profile::IJson` also rejects Unicode noncharacters, regardless of how they are encoded.
Its number policy rejects integers outside ±(2⁵³ − 1), numbers with more than 17 significant digits,
overflow to infinity, and nonzero values that round to zero.
An integer here has no fraction or exponent. Leading and trailing zero digits of the coefficient do not count as significant.

## Set the limits

Set all six fields of `Limits` for the documents you accept.

| Field | What it counts |
| --- | --- |
| `input_bytes_max` | Bytes in the whole document, checked before parsing |
| `depth_max` | Open containers. A scalar has depth 0 and `[]` has depth 1. |
| `string_bytes_max` | Input bytes between the quotes of each string or key, including escapes |
| `members_max`, `elements_max` | Members per object and elements per array |
| `number_digits_max` | Digits across the integer, fraction, and exponent |

## Size the scratch

Lend one `Slot` per open container and one per key of each open object. An object with eight scalar members needs nine slots.
Initialize them with `Slot::EMPTY`. For depth `d` and at most `m` members per object, `d × (m + 1)` slots suffice.
A shorter slice sets a storage limit. `Scratch` identifies the container or key that exceeded it.

## Write a document

`json::write(&mut out, |root| ...)` returns the length of one compact JSON value.
Read only `&out[..len]` after success. Use `object` and `member` for objects, or `array` and `item` for arrays.
The writer escapes strings and keys. The caller selects unique keys.

Enable `base64` with `json` to write base64url strings directly.
Use `base64url` for public bytes and `base64url_secret` inside `with_dit` for a secret.
The [ceremony options example](../../examples/ceremony_options.rs) writes a WebAuthn challenge into the output buffer.

```sh
cargo run --locked --example ceremony_options --features json,base64
```

A write error restores the logical length of the failed value. The buffer can still contain bytes from that value.
Retry the entire document in a larger buffer after `Full`. Clear caller-owned secret buffers after use, including after failure.

## Size the slice

| Value | Output bytes |
| --- | --- |
| String or key | `json::string_len(text)`, including quotes and escapes |
| Object key separator | 1 for `:` |
| Base64url string | Encoded length plus 2 quotes |
| Integer or finite float | At most `I64_BYTES_MAX`, `U64_BYTES_MAX` (20), or `F64_BYTES_MAX` (24) |
| `true`, `false`, `null` | 4, 5, 4 |
| Object or array | 2 delimiters and 1 comma between adjacent items |

## Errors

Inspect `error.kind()` and `error.offset()`. Input offsets count bytes from zero.

| Kind | Meaning |
| --- | --- |
| `Syntax`, `End` | Invalid syntax at the offset, or an incomplete value at end of input |
| `Utf8`, `Surrogate`, `Noncharacter` | Invalid UTF-8, an unpaired surrogate escape, or an I-JSON noncharacter |
| `Duplicate` | Opening quote of the first repeated decoded key in the same object |
| `Number`, `Range` | The I-JSON profile rejects the number, or the requested type cannot represent it. |
| `Limit(limit)`, `Scratch` | The configured limit or slot capacity was exceeded. |
| `Type` | The value has the wrong type, or the writer closure supplied no root value. |
| `Output` | An escaped string needs a larger decode buffer. The offset is the required size. |
| `Full`, `NonFinite` | The writer ran out of space or received NaN or infinity. The offset is the logical length before the failed value or enclosing container. |

## JSONTestSuite

Both profiles reject every `n_` file in JSONTestSuite at `1ef36fa`. They accept the `y_` files with these exceptions.

| File | `Rfc8259` | `IJson` |
| --- | --- | --- |
| `y_object_duplicated_key.json`, `y_object_duplicated_key_and_value.json` | `Duplicate` | `Duplicate` |
| `y_string_escaped_noncharacter.json`, `y_string_last_surrogates_1_and_2.json`, `y_string_nonCharacterInUTF-8_U+10FFFF.json`, `y_string_nonCharacterInUTF-8_U+FFFF.json`, `y_string_unicode_U+10FFFE_nonchar.json`, `y_string_unicode_U+1FFFE_nonchar.json`, `y_string_unicode_U+FDD0_nonchar.json`, `y_string_unicode_U+FFFE_nonchar.json` | reads | `Noncharacter` |

The suite leaves the `i_` cases to the reader.

| Files | `Rfc8259` | `IJson` |
| --- | --- | --- |
| The 10 `i_number_` files: a huge exponent, an overflow, an underflow, and integers past ±(2^53 − 1) | reads, if `number_digits_max` holds the digits | `Number` |
| The 10 `i_string_` and `i_object_` files with a surrogate escape without its pair | `Surrogate` | `Surrogate` |
| The 11 `i_string_` files that are not UTF-8, `i_string_UTF-16LE_with_BOM.json` among them | `Utf8` | `Utf8` |
| `i_string_utf16BE_no_BOM.json`, `i_string_utf16LE_no_BOM.json` | `Syntax` (a NUL byte) | `Syntax` |
| `i_structure_500_nested_arrays.json` | reads, if `depth_max` is 500 or more | as `Rfc8259` |
| `i_structure_UTF-8_BOM_empty_object.json` | `Syntax` at offset 0 (a byte order mark) | `Syntax` |
