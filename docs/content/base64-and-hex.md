# Encode/Decode Base64 and Hex

Enable the codecs you need.

```toml
[dependencies]
rscodec = { version = "0.1.0", features = ["base64", "hex"] }
```

| Format    | API                         | Accepted text                                           |
| --------- | --------------------------- | ------------------------------------------------------- |
| Base64    | `base64::Profile::Standard` | Standard alphabet (`+`, `/`), with required `=` padding |
| Base64url | `base64::Profile::Url`      | URL alphabet (`-`, `_`), without padding                |
| Hex       | `hex`                       | Two lowercase hexadecimal digits per byte               |

Decoders reject whitespace, other alphabets, misplaced padding, and nonzero trailing bits.
Both timing classes return the same bytes, error kind, and offset.

## Decode a Base64url Token into a Stack Buffer

Call `Profile::Url.decode(text, &mut out)` for a public identifier. The result borrows the bytes written to `out`.
Run the [public identifier example](../../examples/base64url.rs) from the checkout.

```sh
cargo run --locked --example base64url --features base64
```

## Select the Timing Class

Use `encode` and `decode` for public data, such as a signature or digest.
Use `encode_secret` and `decode_secret` for private keys, bearer tokens, and other secrets.

```rust
use rscodec::{hex, with_dit};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let mut key = [0; 2];
    with_dit(|dit| hex::decode_secret(dit, b"c0de", &mut key))??;
    assert_eq!(key, [0xc0, 0xde]);
    Ok(())
}
```

One `with_dit` scope can serve several secret calls on the same thread.
On Arm CPUs with data independent timing (DIT), the scope sets `PSTATE.DIT` and restores its previous state.
On a CPU known to lack DIT, the scope runs without that hardware guarantee.
On AArch64, an unknown DIT capability gives `DitUnknown` before the closure runs.
Enable `std` for detection on Linux, Android, and Apple systems. A build for known DIT hardware can use `-C target-feature=+dit`.

Secret functions clear explicit temporary arrays that held input or output. The caller owns the input and output.
Registers and stack slots that the compiler uses to spill registers fall outside this guarantee.

## Size the Output

`Profile::encoded_len(n)` and `hex::encoded_len(n)` return `None` if the required size exceeds `usize`.
`Profile::decoded_len(text)` and `hex::decoded_len(text)` check the length. Call `decode` to validate the characters.

## Errors

Read `error.kind()` and `error.offset()`. Decoders check length, output capacity, then characters and trailing bits.

| Error          | Offset                        | Action                               |
| -------------- | ----------------------------- | ------------------------------------ |
| `Length`       | Input length                  | Correct the text length or padding.  |
| `Output`       | Required output length        | Retry with a buffer of that size.    |
| `Symbol`       | First invalid byte            | Reject the text.                     |
| `TrailingBits` | Last character before padding | Reject the noncanonical base64 text. |

Length and capacity errors leave the buffer untouched. After a character or trailing-bit error, public output is unspecified.
Secret decode clears the result region. Bytes beyond that region stay untouched.

## Kernels

`Backend::new(kernel)` returns a backend when the CPU supports it.
Use `encode_with` or `decode_with` to select a public backend explicitly.
With `std`, dispatch can use CPU detection. Without it, dispatch uses build-time target features.

| Target  | Base64                                                  | Hex                              |
| ------- | ------------------------------------------------------- | -------------------------------- |
| x86_64  | AVX-512 VBMI (public), AVX-512BW, AVX2, SSSE3, portable | AVX-512BW, AVX2, SSSE3, portable |
| AArch64 | Neon, portable                                          | Neon, portable                   |
| wasm32  | SIMD128 when enabled, portable                          | SIMD128 when enabled, portable   |

Other architectures use the portable code. [Platform test coverage](support.md)

## Timing Evidence

Secret kernels avoid branches and memory addresses derived from secret bytes. Length, padding, validity, error kind, and error offset are public.
An instruction scan checks the compiled binary. Hardware counters, stack inspection, and timing samples test different properties.

| Check                                   | Recorded result                                                                                                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arm instruction transfer                | At `7074899`, the corrected scanner found six transfers through general `FMOV`. The decoders now use the DIT-listed byte transfer `UMOV`. The timing effect of the old instruction is unmeasured.                               |
| Corrected instruction scan              | Zero findings in 19 entries per Linux build. Arm GNU and musl each used default features and Neoverse V2. x86_64 musl used default features.                                                                                    |
| Corrected Arm stack check               | A Linux ARM64 VM found no residue in 2,256 secret calls. The public control found residue in 40 of 6,171 calls. The check searches for runs of eight decoded bytes or 16 encoded characters, excluding single-byte repetitions. |
| Path counts at `e03f296`                | For each tested length and outcome, instructions and branches had one count. Xeon tested 15,036 cases and Graviton4 tested 9,050. These runs preceded the transfer correction.                                                  |
| Fixed versus random inputs at `28bdb86` | All six secret operations passed at 16, 32, 64, and 4,096 decoded bytes on default Xeon, default Graviton4, and native Graviton4. At 4,096 bytes the largest absolute t statistic was 3.15 against a threshold of 8.            |

Graviton4 native standard base64 encode at 4,096 bytes exceeded the timing threshold in two of 35 runs on four instances.
The largest absolute t statistic was 17.04. All 12 default-build runs passed. The cause is unknown.
Earlier tests prepared fixed and random inputs differently. Tests with matched preparation did not reproduce the earlier default-build result.

Intel's hardware timing guarantee requires the operating system to enable data operand independent timing mode (DOITM). The tests did not set it.
Native x86_64 builds that emit unlisted VBMI or FP16 instructions and WebAssembly engines have no timing claim.

From the checkout, scan your Linux build with GNU `objdump` available.

```sh
cargo run --locked --release -p rscodec-ct -- scan
```
