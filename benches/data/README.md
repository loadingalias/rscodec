# Benchmark inputs

`benches/json.rs` reads these files. Each directory in `json/` is one class of documents. Each line
of a `.jsonl` file is one document.

## Generated files

These files were written for this repository, under its license (Apache-2.0). Each value that looks
random, for example a challenge, a key, or a checksum, comes from a fixed pseudo-random sequence. No
value is a real key or a real account.

| Class | Files | Bytes for each document | Source of the shape |
| --- | --- | --- | --- |
| `a1-client-data` | 4 | 132–244 | clientDataJSON of WebAuthn Level 3, sections 5.8.1 and 5.8.1.1: the forms create, get, cross-origin, and `topOrigin`. `create.json` also has the extra member that Chrome adds. |
| `a2-jose-header` | 8 | 15–195 | JOSE headers of RFC 7515 (appendices A.1, A.2, and A.3, and section 4.1.11 with `crit`), RFC 7516 appendix A.1, RFC 7518 appendix C, and RFC 7797 section 3. The two headers of RFC 7515 keep their CRLF line breaks. |
| `a3-claims` | 3 | 300–1,518 | JWT and OIDC claims of RFC 7519 section 4 and OpenID Connect Core 1.0 sections 2, 5.1, 5.1.1, and 5.2: an ID token, a UserInfo response with an address and names with language tags, and an access token with group arrays. The names include non-ASCII text. |
| `a4-oidc-discovery` | 4 | 2,184–6,997 | Discovery documents of OpenID Connect Discovery 1.0 section 4.2 and RFC 8414 section 2, from the example of the specification to a large provider document. |
| `a5-jwks` | 3 | 758–4,750 | JWK sets of RFC 7517 appendices A.1 and A.2, RFC 7518 section 6, and RFC 8037 section 2: public keys, private keys, and a provider set with `x5c` chains. |
| `b1-sparse-index` | 11 | 303–4,626 | Files of the Cargo sparse index, in the format of the Cargo Book, "Registry Index", section "Index Format". The files have from 1 to 200 lines, one line for each version. |
| `b2-rustc-diagnostics` | 1 | 42–3,853 | The output of `cargo clippy --message-format=json-diagnostic-rendered-ansi -- -W clippy::pedantic -W clippy::nursery` on a scratch crate, with rustc 1.101.0-nightly (3e92e1a2d 2026-11-01): 200 diagnostics with ANSI-rendered text, the artifact message, and the build result. |

## Fetched files

These standard corpora state no license, so the repository does not hold them. Fetch them into
`fetched/`, which Git ignores, before a benchmark run:

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
