# rscodec

Codecs for Rust: each codec is a leaf feature with portable and SIMD backends and no C.

`0.1.0-alpha.2` is an alpha release. An alpha release can change the API, so pin the exact version:

```toml
[dependencies]
rscodec = { version = "=0.1.0-alpha.2", features = ["base64", "hex"] }
```

The release needs a nightly toolchain. See [Toolchain](#toolchain).

## Codecs

- `base64`: base64 with padding and base64url without padding. See
  [Encode and decode base64, base64url, and hex](docs/content/base64-and-hex.md).
- `hex`: lowercase hex, on the same page.
- `json`: a JSON reader with two profiles, limits, and a strict walk for typed decoders, and a
  writer into a slice. See [Read and write JSON](docs/content/json.md). This codec is in
  development, and it is not in the support scope of this release.

Each codec writes into a slice that you lend, and it allocates no memory. The crate has no
dependency. The `std` feature is on by default. Without it, the crate needs only `core`.

```rust
use rscodec::{base64::Profile, hex};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let digest = [0xfb, 0xff, 0xbf];

    // base64url without padding. The text is the start of the slice that you lend.
    let mut text = [0; 4];
    assert_eq!(Profile::Url.encode(&digest, &mut text)?, b"-_-_");

    let mut bytes = [0; 3];
    assert_eq!(Profile::Url.decode(b"-_-_", &mut bytes)?, digest);

    // `+` and `/` are characters of base64, not of base64url, so this text refuses.
    assert!(Profile::Url.decode(b"+/+/", &mut bytes).is_err());

    let mut digits = [0; 6];
    assert_eq!(hex::encode(&digest, &mut digits)?, b"fbffbf");
    Ok(())
}
```

base64 and hex have functions for public data and functions for secret data, for example a key.
Their page tells you which functions to use.

## Support

The support scope of this release is the `base64` and `hex` features with `std`, on the targets of
this table. On each target, the tests ran each kernel that the CPU has and compared its results with
the portable kernel and with RFC 4648.

| Target | CPU or engine | Kernels that the tests ran |
| --- | --- | --- |
| x86_64 Linux | Xeon 6975P-C | AVX-512 VBMI (base64 only), AVX-512BW, AVX2, SSSE3, portable |
| aarch64 Linux | Graviton4 | Neon, portable |
| aarch64 macOS | Apple M1 Pro | Neon, portable |
| wasm32-wasip1 | Wasmtime 50.0.1 | Portable, and simd128 in a build with `-C target-feature=+simd128` |

These items are experimental:

- Each other target, for example Windows. No test ran on it.
- A build without `std`. The library compiles for `wasm32v1-none`, a target with no `std`, and no
  test runs that build.
- The `json` feature. A differential fuzz campaign of the reader has not run yet.

The timing evidence of the secret functions has its own builds and limits. See
[Timing evidence](docs/content/base64-and-hex.md#timing-evidence).

## Toolchain

The release needs a nightly toolchain. The tested toolchain is `nightly-2026-11-02` (rustc
1.101.0-nightly).

Stable Rust 1.100.0 does not compile the crate. It refuses `#[inline(always)]` on a function with
`#[target_feature]`, which each SIMD kernel uses. The crate turns on no unstable feature: it has no
`#![feature]` attribute.

## Speed

This release does not claim that rscodec is the fastest codec. Three of its 192 speed cells are
open.

One round on 2026-11-24 ran rscodec and its rivals in one process, on the same buffers, on one CPU
core. A cell is one operation at one size, in one timing class, in one build, on one host:

- The operations are base64 encode, base64 decode, hex encode, and hex decode.
- The sizes are 16, 32, 64, 256, 4096, and 65536 bytes.
- The builds are the default build and a build with `-C target-cpu=native`.
- The hosts are a Xeon 6975P-C and a Graviton4, with Linux 7.0 and rustc 1.101.0-nightly.

A cell passes when the median time of rscodec is at most 1.03 times the median time of the fastest
rival. The `[dev-dependencies]` of `Cargo.toml` pin the version of each rival.

| Functions | Rivals | Cells that passed |
| --- | --- | ---: |
| Public | base64-simd, base64-turbo, base64, base64-ng, encodify, data-encoding, simdutf, const-hex, faster-hex | 93 of 96 |
| Secret | base64ct and base16ct, the rivals that claim constant time | 96 of 96 |

The three open cells are public functions in the default build:

| Host | Call | rscodec | Fastest rival | Ratio | Kernel alone |
| --- | --- | ---: | ---: | ---: | ---: |
| Xeon 6975P-C | `hex::decode`, 32 bytes of result | 2.928 ns | faster-hex, 2.836 ns | 1.032 | 2.371 ns |
| Xeon 6975P-C | `hex::decode`, 64 bytes of result | 3.951 ns | faster-hex, 3.823 ns | 1.033 | 3.277 ns |
| Graviton4 | `hex::encode`, 16 bytes of input | 2.296 ns | const-hex, 2.149 ns | 1.068 | 2.152 ns |

In each open cell, the kernel alone (`decode_with` or `encode_with`) passes. The public function
adds the time of its entry code before the kernel. The cause of that time is not proved. The entry
code of `src/hex/` owns these cells, and they stay open until one round passes all 192 cells.

One run gave each median, with 50 samples for each row. In earlier rounds, the same code moved by up
to 18% between two builds. Thus a ratio near 1.03 can pass in one round and fail in the next.

## Security

Report a vulnerability in private. See [SECURITY.md](SECURITY.md).

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) gives the setup, the commands, and the sign-off of a commit.
rscodec has the license [Apache-2.0](LICENSE).
