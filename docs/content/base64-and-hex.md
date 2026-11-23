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
  the kernels whose instructions on secret data Intel lists with data operand independent timing
  (DOIT) and Arm lists with data independent timing (DIT). On x86_64 with AVX-512 VBMI, public
  base64 runs the VBMI kernel, and secret base64 runs the AVX-512BW kernel. hex runs the same kernel
  in each class.
- **`PSTATE.DIT`.** A secret function takes `&Dit`, a proof that only `rscodec::with_dit` gives.
  On aarch64, `with_dit` sets `PSTATE.DIT` for its scope, if the CPU has DIT, then runs a
  speculation barrier (`sb`, or `dsb nsh` and `isb` on a CPU without SB), and at the end restores
  `PSTATE.DIT`. The secret functions and the public functions do not read or write it.
- **No proof without DIT.** On aarch64, a build that cannot tell whether the CPU has DIT gets
  `DitUnknown` from `with_dit`, and the scope does not run. `std` reports DIT on Linux, Android,
  and the Apple systems. Elsewhere, for example on Windows or without `std`, build with
  `-C target-feature=+dit` for CPUs that have DIT.
- **Output after a refusal.** After `decode_secret` refuses a text for a character or for trailing
  bits, each byte that it wrote is zero. After `decode` refuses, the output is unspecified.
- **Temporary copies.** Before a secret function returns, on a result or a refusal, it writes zero
  to each temporary array that held input or result bytes, for example the last block of a SIMD
  kernel. The public functions do not.

The secret functions clear with volatile stores and a compiler fence. The clear does not cover the
input that you own, the result in your output, or the copies in registers and in the stack slots
where the compiler spills registers.

The secret functions can be slower. On x86_64 with VBMI, the AVX-512BW kernel of base64 is slower
than the VBMI kernel. On aarch64, a scope of `with_dit` writes `PSTATE.DIT` two times and runs one
speculation barrier, which costs most for one short input. If you are not sure that the data is public, use the secret functions.

## Timing evidence

The harness `rscodec-ct` of the repository checks the compiled code of both classes. It checks
the shipped build (the default target features) on Linux on x86_64 (Xeon 6975P-C) and aarch64
(Graviton4), and the build with `-C target-cpu=native` on aarch64.

| Check | Result |
| --- | --- |
| Path count: the counters of the instructions and the branches of each call, on each kernel of the CPU, at each length up to 200 bytes and at 13 longer lengths, with random and fixed inputs | One count for each length and outcome |
| Scan of the disassembly | No load or store at an address from the data, no write of `PSTATE.DIT` in the secret code, and each instruction that reads secret data on the DOIT or DIT list |
| Painted stack: each secret call on a private stack | No run of 8 secret bytes or 16 secret characters stays in the stack |
| DudeCT: a statistical test of a fixed input against random inputs of the same length | No difference up to 64 bytes. At 4096 bytes, see the limits |

The result that the caller sees is public, so the time can show it: the lengths, the validity of a
text, the kind and the offset of a refusal, and the padding of base64.

Limits:

- A build whose target features add instructions off the DOIT list has no timing claim. With
  `-C target-cpu=native` on a CPU with AVX-512 VBMI or AVX512-FP16, LLVM puts VBMI and FP16
  instructions in the secret code.
- WebAssembly has no timing claim. It has no list of instructions with data independent timing,
  and the engine compiles the module again. Its kernels keep the rule of the source and the tests.
- At 4096 bytes, DudeCT finds a difference between a fixed input and random inputs: up to 5% when
  the input comes from outside the cache, on both CPUs, and about 0.05% for hex encode and base64
  decode on Graviton4 when the input is in the cache. The path count and the scan find no path or
  address that depends on the data. This result is open.
- Intel guarantees the DOIT list only in DOIT mode (DOITM), which the operating system sets on a CPU
  that has it. The tests did not set it.

To scan a build with your target features, run the scan from the repository. It needs `objdump`,
and it exits with 0 when it finds nothing and with 1 when it finds something:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release -p rscodec-ct -- scan
```

The last line of its output gives the number of findings, for example `scan: 18 entries, 0
findings`. The path count, `cargo run --release -p rscodec-ct -- paths`, needs the counters of
Linux.

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
    let refusal = Error::new(ErrorKind::Symbol, 9);
    assert_eq!(Profile::Url.decode(standard, &mut buffer), Err(refusal));
    Ok(())
}
```

`decode` gives the decoded bytes, which are the start of the buffer. The buffer can be longer than
the result.

## Decode a secret key

A private key is secret, so `decode_secret` reads it in a scope of `with_dit`:

```rust
use rscodec::{base64::Profile, hex, with_dit};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let mut key = [0; 32];
    let text = b"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    // The outer `?` is `DitUnknown`: this build cannot hold `PSTATE.DIT`.
    with_dit(|dit| -> Result<(), Box<dyn core::error::Error>> {
        assert_eq!(hex::decode_secret(dit, text, &mut key)?[31], 0x1f);

        // A refusal leaves zero in each byte that the decoder wrote.
        assert!(Profile::Url.decode_secret(dit, b"AAAA*AAA", &mut key).is_err());
        Ok(())
    })??;
    assert_eq!(key[..6], [0; 6]);
    Ok(())
}
```

The closure of `with_dit` gets `&Dit`. A `Dit` is neither `Send` nor `Sync`, and the reference is
valid only in the closure, so a secret function runs only on the thread and in the scope that hold
`PSTATE.DIT`.

## Hold DIT for a loop of secrets

Hold one scope for the loop. The scope writes `PSTATE.DIT` at most two times, and the secret
functions in it write nothing:

```rust
use rscodec::base64::{Error, Profile};
use rscodec::with_dit;

fn decode_keys(tokens: &[&[u8]], keys: &mut [[u8; 32]]) -> Result<(), Box<dyn core::error::Error>> {
    with_dit(|dit| {
        for (token, key) in tokens.iter().zip(keys.iter_mut()) {
            Profile::Url.decode_secret(dit, token, key)?;
        }
        Ok::<(), Error>(())
    })??;
    Ok(())
}
```

If `PSTATE.DIT` is set already, for example in an outer scope, `with_dit` writes nothing. On x86_64
and WebAssembly, and on an aarch64 CPU without DIT, `with_dit` only calls the closure. Code in the
scope must not clear `PSTATE.DIT`.

The writes of DIT read and write memory for the compiler, so the loads and stores of each secret
function stay in the scope. The proof covers the secret functions of rscodec. It does not cover
other code in the scope that works only on values in registers.

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
arguments after the `&Dit` of a scope of `with_dit`.

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

Each error has a kind, which `kind()` gives, and an offset, which `offset()` gives. It holds no byte
of the input. `Error::new(kind, offset)` makes an error.

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
- A short input runs inline in the caller, with no choice of kernel: on aarch64, up to 64 bytes on
  the Neon kernel, and on x86_64, for hex, an encode of up to 32 bytes and a decode of up to 16
  bytes of result on SSE2. When `-C target-feature` fixes a SIMD kernel of a class, that kernel
  runs inline at each length. `encode_with` and `decode_with` always run the kernel of their
  backend.
- Each kernel runs whole blocks, then one last block that ends at the end of the input and writes
  some bytes of the block before it again. Thus a tail needs no copy. Only an input shorter than
  one vector runs on arrays on the stack, or on AVX-512 with masked loads and stores.
- The AVX-512 base64 kernels have an entry with no loop for up to 48 bytes. The AVX-512BW hex
  kernel runs one 64-byte vector for an input of up to 32 bytes or a text of up to 64 characters,
  and an encode of 64 KiB or more in 32-byte vectors, which keep the clock of the CPU higher.
- WebAssembly has no run-time detection, so `Simd128` exists only with `+simd128`.
- On an Intel hybrid CPU (Alder Lake to Arrow Lake), the AVX-512 kernels do not run, also if CPUID
  reports AVX-512. Its efficient cores do not have AVX-512.

The source of each kernel has no branch and no table load that depends on the bytes or the
characters. The time is to depend only on the lengths, and for an error, on its offset.
[Timing evidence](#timing-evidence) gives the checks of the compiled code and their limits.
