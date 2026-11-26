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

rscodec is one part of a family of Rust libraries:
[rscrypto](https://github.com/loadingalias/rscrypto) (cryptography),
[rsasync](https://github.com/loadingalias/rsasync) (runtime),
[rsnetwork](https://github.com/loadingalias/rsnetwork) (networking),
[rsauthnz](https://github.com/loadingalias/rsauthnz) (auth), and rscodec (codecs).

License: Apache-2.0.
