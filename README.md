# rscodec

**DO NOT USE IN PRODUCTION.**

rscodec is a set of codecs for Rust: base64, base64url, hex, and JSON. You select each codec as a
Cargo feature. Each codec has a portable backend and SIMD backends, allocates no memory, and has
no C.

rscodec is in development. Each release can change the API.

rscodec is one part of a family of Rust libraries:
[rscrypto](https://github.com/loadingalias/rscrypto) (cryptography),
[rsasync](https://github.com/loadingalias/rsasync) (runtime),
[rsnetwork](https://github.com/loadingalias/rsnetwork) (networking),
[rsauthnz](https://github.com/loadingalias/rsauthnz) (auth), and rscodec (codecs).

License: Apache-2.0.
