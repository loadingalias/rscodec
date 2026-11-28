# Benchmark Inputs

Each directory in `json/` is one input class for the JSON benchmarks. Each JSON Lines record is one document.

## Included Inputs

Synthetic identifiers, keys, and accounts use fixed test values. The generated inputs use the repository's Apache-2.0 license.

| Class | Files | Bytes for each document | Source of the shape |
| --- | --- | --- | --- |
| `a1-client-data` | 4 | 132–244 | WebAuthn Level 3 §5.8.1. Creation, authentication, cross-origin requests, and `topOrigin`. |
| `a2-jose-header` | 8 | 15–195 | JOSE headers from RFCs 7515, 7516, 7518, and 7797. Two retain their original CRLF breaks. |
| `a3-claims` | 3 | 300–1,518 | JWT and OIDC claims with nested addresses, language tags, and groups. |
| `a4-oidc-discovery` | 4 | 2,184–6,997 | Provider metadata from OpenID Connect Discovery §4.2 and RFC 8414 §2. |
| `a5-jwks` | 3 | 758–4,750 | JWK sets from RFCs 7517, 7518, and 8037, including certificate chains. |
| `b1-sparse-index` | 11 | 303–4,626 | Cargo registry index format. Each file holds 1–200 version records. |
| `b2-rustc-diagnostics` | 1 | 42–3,853 | Clippy JSON Lines output from Rust 1.101.0-nightly (`3e92e1a2d`), including 200 diagnostics. |

## Fetched Files

The upstream corpora have no stated license. Fetch the pinned files before tests or JSON benchmarks.

```sh
cd benches/data
mkdir -p fetched
for name in twitter citm_catalog canada; do
  curl -fsSL -o "fetched/$name.json" \
    "https://raw.githubusercontent.com/simdjson/simdjson-data/4197c425e857f0ec38e89822fdd0bd9ea21f4daf/jsonexamples/$name.json"
done
sha256sum -c corpora.sha256    # On macOS: shasum -a 256 -c corpora.sha256
```

| File | Bytes | Content | Source |
| --- | --- | --- | --- |
| `twitter.json` | 631,515 | Search results of the Twitter API: strings, CJK text, and integers | simdjson-data, from nativejson-benchmark |
| `citm_catalog.json` | 1,727,204 | A catalog with many keys, integers, and indentation | simdjson-data, from nativejson-benchmark |
| `canada.json` | 2,251,051 | The outline of Canada as GeoJSON: floating-point numbers | simdjson-data, from nativejson-benchmark |
