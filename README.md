# rscodec

Encode and decode base64, base64url, and lowercase hex. Read and write JSON in caller-owned buffers.
The library allocates no memory and has no dependencies.

```toml
[dependencies]
rscodec = { version = "0.1.0", features = ["base64"] }
```

```rust
use rscodec::base64::Profile;

fn main() -> Result<(), rscodec::base64::Error> {
    let mut bytes = [0; 4];
    assert_eq!(Profile::Url.decode(b"AQID_w", &mut bytes)?, [1, 2, 3, 255]);
    Ok(())
}
```

Enable the features you use. The default feature, `std`, enables runtime CPU detection.
Set `default-features = false` for a `core`-only build. The crate includes the guides and examples. The minimum Rust version is 1.101.
The repository tests `nightly-2026-11-02`.

| Feature | Operations |
| --- | --- |
| `base64` | Padded base64 and unpadded base64url. Separate functions handle public and secret data. |
| `hex` | Lowercase hex with the same two timing classes. |
| `json` | Read borrowed tokens or typed values. Reject duplicate keys. Write compact JSON. |

[Base64 and hex guide](https://github.com/loadingalias/rscodec/blob/main/docs/content/base64-and-hex.md) ·
[JSON guide](https://github.com/loadingalias/rscodec/blob/main/docs/content/json.md) ·
[Runnable examples](https://github.com/loadingalias/rscodec/tree/main/examples) ·
[Platform support](https://github.com/loadingalias/rscodec/blob/main/docs/content/support.md)

## Benchmarks

rscodec won 277 of 308 comparisons, tied 15, and lost 16. It won all 96 secret base64 and hex
comparisons and all 20 typed JSON comparisons. sonic-rs read `canada.json` faster in all four builds.

The November 27, 2026 run used revision [`e03f296`](https://github.com/loadingalias/rscodec/commit/e03f296).
Each operation used the same input as its fastest measured rival. The run preceded the secret-decoder corrections. Secret operations competed against base64ct or base16ct.

Both hosts ran Linux on AWS. The hosts were a Xeon 6975P-C (`c8i.xlarge`) and a Graviton4 (`c8g.xlarge`).
“Shipped” uses the default target features. “Native” uses `-C target-cpu=native`.
Each build ran at eight stack addresses with address randomization off. A win required all eight
rscodec medians to beat all eight rival medians. Overlapping ranges tied. Ratios use the mean of each set of medians.

Each entry gives wins–ties–losses.

| Operation | Xeon shipped | Xeon native | Graviton4 shipped | Graviton4 native | Total |
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
| **All cells** | **68–4–5** | **72–1–4** | **70–5–2** | **67–5–5** | **277–15–16** |

The ratio below is rscodec time divided by rival time. Base64 and hex sizes count decoded bytes.

| Operation and input | Build | Fastest rival | Time ratio |
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

[All 308 measurements (CSV)](https://github.com/loadingalias/rscodec/blob/main/benches/results/e03f296.csv) ·
[Run the benchmarks](https://github.com/loadingalias/rscodec/blob/main/benches/README.md) ·
[Timing evidence and limits](https://github.com/loadingalias/rscodec/blob/main/docs/content/base64-and-hex.md#timing-evidence) ·
[Report a vulnerability](https://github.com/loadingalias/rscodec/security/advisories/new) ·
[Apache-2.0 license](https://github.com/loadingalias/rscodec/blob/main/LICENSE)
