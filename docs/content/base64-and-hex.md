# Encode and decode base64, base64url, and hex

rscodec encodes and decodes three text forms of bytes:

| Form | Module item | Standard | Characters |
| --- | --- | --- | --- |
| base64url without padding | `base64::Profile::Url` | RFC 4648, section 5 | `A`–`Z`, `a`–`z`, `0`–`9`, `-`, `_` |
| base64 with padding | `base64::Profile::Standard` | RFC 4648, section 4 | `A`–`Z`, `a`–`z`, `0`–`9`, `+`, `/`, and `=` at the end |
| lowercase hex | `hex` | RFC 4648, section 8 | `0`–`9`, `a`–`f` |

Each function writes into a slice that you lend, and it allocates no memory.

## Select the codecs

Each codec is a feature. Select only the codecs that you use:

```toml
[dependencies]
rscodec = { version = "0.1.0-alpha.2", features = ["base64", "hex"] }
```

The first release with these codecs, `0.1.0-alpha.2`, is not on crates.io yet.

The `std` feature is on by default. Without it (`default-features = false`), the crate needs only
`core`, and it selects the SIMD backend at compile time from `-C target-feature`.

## Decode a base64url token into a stack buffer

```rust
use rscodec::base64::{Error, ErrorKind, Profile};

fn main() -> Result<(), Error> {
    // A WebAuthn challenge: 32 bytes in base64url without padding.
    let token = b"NI2kalLUY_EBIuNcW6yDtzD4K6bGemOZJL9NWqY_heQ";
    let mut buffer = [0; 32];
    let challenge = Profile::Url.decode(token, &mut buffer)?;
    assert_eq!(challenge[..4], [0x34, 0x8d, 0xa4, 0x6a]);

    // `/` is a character of base64, not of base64url, so the text refuses at the `/`.
    let standard = b"NI2kalLUY/EBIuNcW6yDtzD4K6bGemOZJL9NWqY/heQ";
    let refusal = Error { kind: ErrorKind::Symbol, offset: 9 };
    assert_eq!(Profile::Url.decode(standard, &mut buffer), Err(refusal));
    Ok(())
}
```

`decode` gives the decoded bytes, which are the start of the buffer. The buffer can be longer than
the result.

## Encode

```rust
use rscodec::{base64::Profile, hex};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let key = [0xc0, 0xde, 0x01];
    let mut text = [0; Profile::Standard.encoded_len(3).unwrap()];
    assert_eq!(Profile::Standard.encode(&key, &mut text)?, b"wN4B");
    let mut digits = [0; 6];
    assert_eq!(hex::encode(&key, &mut digits)?, b"c0de01");
    Ok(())
}
```

The text is ASCII, so `core::str::from_utf8` on it cannot fail.

## Size the output

| Function | Result |
| --- | --- |
| `Profile::encoded_len(len)` | The characters of the text of `len` bytes, or `None` if the number overflows `usize` |
| `Profile::decoded_len(text)` | The bytes of `text`, from its length and its padding, or a `Length` error |
| `hex::encoded_len(len)` | `2 * len`, or `None` if it overflows `usize` |
| `hex::decoded_len(text)` | `text.len() / 2`, or a `Length` error for an odd length |

`encoded_len` is a `const fn`, so it can give the length of an array, as the example of
[Encode](#encode) shows.

## Strict decoding

Each byte string has one text, and the decoders take only that text:

- Each profile has one alphabet. base64url refuses `+` and `/`, and base64 refuses `-` and `_`.
- base64 has a length that is a multiple of 4, and `=` only in the last two places. base64url has
  no `=`.
- The bits of the last character after the last byte are zero.
- No whitespace and no line breaks.
- hex is lowercase. `A`–`F` refuse.

## Errors

Each error has a kind and an offset, and it holds no byte of the input.

| Kind | Cause | Offset | Recovery |
| --- | --- | --- | --- |
| `Length` | No text of the form has this length | The length of the text | Refuse the input |
| `Output` | The output slice is too short | The length of the result | Give an output of that length |
| `Symbol` | A byte is not a character of the alphabet | The first such byte | Refuse the input |
| `TrailingBits` | base64 only: the last character has nonzero bits after the last byte | The last character before the padding | Refuse the input |

The decoder checks the length first, then the output, then the characters. It reads all of the
text before it refuses it for a character or for trailing bits.

After an error, the bytes of the output slice are unspecified.

## Backends

Each function without a `Backend` argument uses `Backend::detect()`, the fastest backend that the
CPU runs. `encode_with` and `decode_with` take a backend. `Backend::new(isa)` gives the backend of
an instruction set, or `None` if the CPU does not run it. Each backend gives the same result and the
same error for each input.

```rust
use rscodec::{Backend, Isa, base64::Profile};

fn main() -> Result<(), rscodec::base64::Error> {
    let portable = Backend::new(Isa::Portable).expect("each CPU runs the portable backend");
    let mut text = [0; 4];
    assert_eq!(Profile::Url.encode_with(portable, b"\xfb\xff\xbf", &mut text)?, b"-_-_");
    Ok(())
}
```

| Target | Instruction sets (`Isa`), fastest first |
| --- | --- |
| x86_64 | `Avx512Vbmi`, `Avx512Bw`, `Avx2`, `Ssse3`, `Portable` |
| aarch64 | `Neon`, `Portable` |
| wasm32 with `-C target-feature=+simd128` | `Simd128`, `Portable` |
| other targets | `Portable` |

- With `std`, the crate detects the CPU at run time. Without `std`, a backend is available only if
  `-C target-feature` turns on its features.
- WebAssembly has no run-time detection, so `Simd128` exists only with `+simd128`.
- `Avx512Vbmi` and `Avx512Bw` run the same hex kernel.
- On aarch64, each call sets `PSTATE.DIT` (data independent timing) if the CPU has it, and then
  restores it. If DIT is set already, for example because the caller set it for its thread, the
  call does not write it.

The source of each backend has no branch and no table load that depends on the bytes or the
characters. The time is to depend only on the lengths, and for an error, on its offset. A check of
the compiled code and a timing test of this rule are not complete yet, so this release makes no
timing claim. `Avx512Vbmi` uses byte permutes that Intel does not list with data-independent timing.
`Avx512Bw` uses only listed instructions.
