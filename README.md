# rscodec

**DO NOT USE IN PRODUCTION.**

rscodec is a set of codecs for Rust: base64, base64url, hex, and JSON. Each codec is a Cargo
feature (`base64`, `hex`, `json`), has a portable backend and SIMD backends, allocates no memory,
and has no C.

base64, base64url, and hex have a secret class for keys and tokens: its functions run only kernels
whose instructions have data independent timing.

- [base64, base64url, and hex](docs/content/base64-and-hex.md)
- [JSON](docs/content/json.md)

rscodec needs Rust 1.101, a nightly toolchain today. Before 1.0, a minor release can change the API.

## Speed

The final round of v0.1 (revision `e03f296`, 2026-11-27) compares rscodec with the fastest other
library of each cell, on the same input in the same process. The rivals are Rust crates, simdutf
through its Rust binding, and `core` for f64 read. The secret class runs against the constant-time
crates base64ct and base16ct.

- Hosts: AWS c8i.xlarge (Intel Xeon 6975P-C) and c8g.xlarge (Graviton4), Linux.
- Builds: shipped (the default target features) and native (`-C target-cpu=native`).
- Each build ran at 8 stack addresses, with ASLR off, and a row is the mean of its 8 medians.
- Win, tie, loss: a win when each of the 8 medians of rscodec is lower than each median of the
  rival, a loss when each is higher, and a tie when the two ranges overlap.

Each entry is wins–ties–losses.

| Cells | Xeon shipped | Xeon native | Graviton4 shipped | Graviton4 native | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| base64 encode | 4–1–1 | 5–0–1 | 6–0–0 | 5–1–0 | 20–2–2 |
| base64 decode | 6–0–0 | 6–0–0 | 5–0–1 | 4–0–2 | 21–0–3 |
| hex encode | 4–1–1 | 5–1–0 | 2–4–0 | 2–2–2 | 13–8–3 |
| hex decode | 4–0–2 | 6–0–0 | 6–0–0 | 6–0–0 | 22–0–2 |
| base64 and hex, secret class | 24–0–0 | 24–0–0 | 24–0–0 | 24–0–0 | 96–0–0 |
| JSON read | 8–1–1 | 8–0–2 | 9–0–1 | 9–0–1 | 34–1–5 |
| JSON typed read | 5–0–0 | 5–0–0 | 5–0–0 | 5–0–0 | 20–0–0 |
| JSON write | 10–0–0 | 10–0–0 | 9–1–0 | 8–2–0 | 37–3–0 |
| f64 print | 1–1–0 | 1–0–1 | 2–0–0 | 2–0–0 | 6–1–1 |
| f64 read | 2–0–0 | 2–0–0 | 2–0–0 | 2–0–0 | 8–0–0 |
| **All cells** | | | | | **277–15–16** |

Each loss, with the time of rscodec divided by the time of the rival:

| Loss | Build | Fastest rival | Time of rscodec / rival |
| --- | --- | --- | ---: |
| JSON read canada | Xeon shipped | sonic-rs | 1.165 |
| JSON read canada | Xeon native | sonic-rs | 1.133 |
| hex decode 32 | Xeon shipped | faster-hex | 1.113 |
| JSON read canada | Graviton4 native | sonic-rs | 1.099 |
| JSON read canada | Graviton4 shipped | sonic-rs | 1.087 |
| f64 print random | Xeon native | zmij | 1.054 |
| hex decode 64 | Xeon shipped | faster-hex | 1.050 |
| hex encode 256 | Graviton4 native | const-hex | 1.047 |
| hex encode 16 | Xeon shipped | const-hex | 1.036 |
| JSON read b1-sparse-index | Xeon native | sonic-rs | 1.033 |
| base64 encode std-65536 | Xeon shipped | base64-turbo | 1.021 |
| base64 encode std-65536 | Xeon native | base64-turbo | 1.015 |
| base64 decode url-32 | Graviton4 native | base64-simd | 1.012 |
| base64 decode url-32 | Graviton4 shipped | base64-simd | 1.007 |
| base64 decode url-4096 | Graviton4 native | encodify | 1.006 |
| hex encode 4096 | Graviton4 native | faster-hex | 1.002 |

rscodec is one part of a family of Rust libraries:
[rscrypto](https://github.com/loadingalias/rscrypto) (cryptography),
[rsasync](https://github.com/loadingalias/rsasync) (runtime),
[rsnetwork](https://github.com/loadingalias/rsnetwork) (networking),
[rsauthnz](https://github.com/loadingalias/rsauthnz) (auth), and rscodec (codecs).

License: Apache-2.0.
