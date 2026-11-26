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

The final round of v0.1 (revision `1b483bc`, 2026-11-26) compares rscodec with the fastest other
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
| base64 encode | 4–1–1 | 5–0–1 | 6–0–0 | 5–0–1 | 20–1–3 |
| base64 decode | 6–0–0 | 6–0–0 | 5–0–1 | 4–0–2 | 21–0–3 |
| hex encode | 5–0–1 | 5–1–0 | 2–4–0 | 2–2–2 | 14–7–3 |
| hex decode | 4–0–2 | 6–0–0 | 6–0–0 | 6–0–0 | 22–0–2 |
| base64 and hex, secret class | 24–0–0 | 24–0–0 | 24–0–0 | 24–0–0 | 96–0–0 |
| JSON read | 8–0–2 | 8–0–2 | 8–1–1 | 7–0–3 | 31–1–8 |
| JSON typed read | 5–0–0 | 5–0–0 | 5–0–0 | 5–0–0 | 20–0–0 |
| JSON write | 10–0–0 | 10–0–0 | 10–0–0 | 10–0–0 | 40–0–0 |
| f64 print | 1–1–0 | 1–0–1 | 2–0–0 | 2–0–0 | 6–1–1 |
| f64 read | 2–0–0 | 2–0–0 | 2–0–0 | 2–0–0 | 8–0–0 |
| **All cells** | | | | | **278–10–20** |

Each loss, with the time of rscodec divided by the time of the rival:

| Loss | Build | Fastest rival | Time of rscodec / rival |
| --- | --- | --- | ---: |
| JSON read canada | Xeon shipped | sonic-rs | 1.481 |
| JSON read canada | Graviton4 native | sonic-rs | 1.464 |
| JSON read canada | Graviton4 shipped | sonic-rs | 1.454 |
| JSON read canada | Xeon native | sonic-rs | 1.400 |
| JSON read b1-sparse-index | Xeon native | sonic-rs | 1.108 |
| JSON read b1-sparse-index | Xeon shipped | sonic-rs | 1.103 |
| hex decode 32 | Xeon shipped | faster-hex | 1.097 |
| f64 print random | Xeon native | zmij | 1.073 |
| hex decode 64 | Xeon shipped | faster-hex | 1.059 |
| hex encode 256 | Graviton4 native | const-hex | 1.046 |
| hex encode 16 | Xeon shipped | const-hex | 1.032 |
| base64 encode std-65536 | Xeon shipped | base64-turbo | 1.019 |
| JSON read b2-rustc-diagnostics | Graviton4 native | sonic-rs | 1.016 |
| JSON read b1-sparse-index | Graviton4 native | sonic-rs | 1.016 |
| base64 encode std-65536 | Xeon native | base64-turbo | 1.013 |
| base64 decode url-32 | Graviton4 native | base64-simd | 1.012 |
| base64 decode url-32 | Graviton4 shipped | base64-simd | 1.008 |
| base64 decode url-4096 | Graviton4 native | encodify | 1.006 |
| base64 encode url-16 | Graviton4 native | base64-simd | 1.006 |
| hex encode 4096 | Graviton4 native | faster-hex | 1.001 |

rscodec is one part of a family of Rust libraries:
[rscrypto](https://github.com/loadingalias/rscrypto) (cryptography),
[rsasync](https://github.com/loadingalias/rsasync) (runtime),
[rsnetwork](https://github.com/loadingalias/rsnetwork) (networking),
[rsauthnz](https://github.com/loadingalias/rsauthnz) (auth), and rscodec (codecs).

License: Apache-2.0.
