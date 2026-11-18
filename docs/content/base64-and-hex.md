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
`core`, and it selects each kernel at compile time from `-C target-feature`.

## Select the timing class

Each operation has a function for public data and a function for secret data:

| Data | Examples | base64 | hex |
| --- | --- | --- | --- |
| Public | A digest, a signature, a public key, a challenge | `Profile::encode`, `Profile::decode` | `hex::encode`, `hex::decode` |
| Secret | A private key, a bearer token, a session secret | `Profile::encode_secret`, `Profile::decode_secret` | `hex::encode_secret`, `hex::decode_secret` |

Both classes give the same result and the same error for each input. They differ in four ways:

- **Kernel.** The public functions run the fastest kernel of the CPU. The secret functions run only
  the kernels whose instructions Intel lists with data operand independent timing (DOIT) and Arm
  lists with data independent timing (DIT). On x86_64 with AVX-512 VBMI, public base64 runs the
  VBMI kernel, and secret base64 runs the AVX-512BW kernel. hex runs the same kernel in each class.
- **`PSTATE.DIT`.** On aarch64, a secret function sets `PSTATE.DIT` for the call, if the CPU has
  DIT, and then restores it. A public function does not write it.
- **Output after a refusal.** After `decode_secret` refuses a text for a character or for trailing
  bits, each byte that it wrote is zero. After `decode` refuses, the output is unspecified.
- **Temporary copies.** Before a secret function returns, on a result or a refusal, it writes zero
  to each temporary array that held input or result bytes, for example the last block of a SIMD
  kernel. The public functions do not.

The secret functions clear with volatile stores and a compiler fence. The clear does not cover the
input that you own, the result in your output, or the copies in registers and in the stack slots
where the compiler spills registers.

The secret functions can be slower. On aarch64, the two writes of `PSTATE.DIT` cost most on short
inputs. On x86_64 with VBMI, the AVX-512BW kernel of base64 is slower than the VBMI kernel. If you
are not sure that the data is public, use the secret functions.

## Decode a base64url token into a stack buffer

A WebAuthn challenge is public, so `decode` reads it:

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

## Decode a secret key

A private key is secret, so `decode_secret` reads it:

```rust
use rscodec::{base64::Profile, hex};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let mut key = [0; 32];
    let text = b"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    assert_eq!(hex::decode_secret(text, &mut key)?[31], 0x1f);

    // A refusal leaves zero in each byte that the decoder wrote.
    assert!(Profile::Url.decode_secret(b"AAAA*AAA", &mut key).is_err());
    assert_eq!(key[..6], [0; 6]);
    Ok(())
}
```

## Hold DIT for a loop of secrets

A secret function writes `PSTATE.DIT` twice when DIT is clear. If DIT is set already, it writes
nothing. To decode many secrets, set DIT one time for the loop with
`rscrypto::traits::ct::with_data_independent_timing`. This needs a dependency on `rscrypto`:

```rust
use rscodec::base64::{Error, Profile};
use rscrypto::traits::ct::with_data_independent_timing;

fn decode_keys(tokens: &[&[u8]], keys: &mut [[u8; 32]]) -> Result<(), Error> {
    with_data_independent_timing(|| {
        for (token, key) in tokens.iter().zip(keys.iter_mut()) {
            Profile::Url.decode_secret(token, key)?;
        }
        Ok(())
    })
}
```

On x86_64 and WebAssembly, and on an aarch64 CPU without DIT, the wrapper and the secret functions
write no timing mode.

## Encode

```rust
use rscodec::{base64::Profile, hex};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let digest = [0xc0, 0xde, 0x01];
    let mut text = [0; Profile::Standard.encoded_len(3).unwrap()];
    assert_eq!(Profile::Standard.encode(&digest, &mut text)?, b"wN4B");
    let mut digits = [0; 6];
    assert_eq!(hex::encode(&digest, &mut digits)?, b"c0de01");
    Ok(())
}
```

The text is ASCII, so `core::str::from_utf8` on it cannot fail. `encode_secret` takes the same
arguments.

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

A refusal for the length or the output writes no byte. After a refusal for a character or for
trailing bits, `decode` leaves unspecified bytes in the output, and `decode_secret` leaves zero in
the bytes of the result.

## Kernels

Each codec has a `Kernel` enum, `base64::Kernel` and `hex::Kernel`. `Backend::new(kernel)` of the
same module gives the backend of a kernel, or `None` if the CPU does not have its features.
`encode_with` and `decode_with` run the kernel of a backend, for example to compare the kernels.
They are functions of the public class: they do not write `PSTATE.DIT`, and `decode_with` does not
wipe its output.

```rust
use rscodec::base64::{Backend, Kernel, Profile};

fn main() -> Result<(), rscodec::base64::Error> {
    let portable = Backend::new(Kernel::Portable).expect("each CPU runs the portable kernel");
    let mut text = [0; 4];
    assert_eq!(Profile::Url.encode_with(portable, b"\xfb\xff\xbf", &mut text)?, b"-_-_");
    Ok(())
}
```

`Kernel::ALL` lists the kernels of the target, the fastest first:

| Target | `base64::Kernel` | `hex::Kernel` |
| --- | --- | --- |
| x86_64 | `Avx512Vbmi` (public class only), `Avx512Bw`, `Avx2`, `Ssse3`, `Portable` | `Avx512Bw`, `Avx2`, `Ssse3`, `Portable` |
| aarch64 | `Neon`, `Portable` | `Neon`, `Portable` |
| wasm32 with `-C target-feature=+simd128` | `Simd128`, `Portable` | `Simd128`, `Portable` |
| other targets | `Portable` | `Portable` |

- With `std`, the crate detects the CPU one time for each process. When `-C target-feature` turns
  on the features of the fastest kernel of a class, the choice is made at compile time.
- Without `std`, a kernel is available only if `-C target-feature` turns on its features.
- WebAssembly has no run-time detection, so `Simd128` exists only with `+simd128`.
- On an Intel hybrid CPU (Alder Lake to Arrow Lake), the AVX-512 kernels do not run, also if CPUID
  reports AVX-512. Its efficient cores do not have AVX-512.

The source of each kernel has no branch and no table load that depends on the bytes or the
characters. The time is to depend only on the lengths, and for an error, on its offset. A check of
the compiled code and a timing test of this rule are not complete yet, so this release makes no
timing claim. WebAssembly has no list of instructions with data independent timing, so its kernels
keep only this rule of the source.
