# rscodec

Codecs for Rust: each codec is a leaf feature with portable and SIMD backends and no C.

This crate is in development. Its codecs:

- `base64`: base64 with padding and base64url without padding. See
  [Encode and decode base64, base64url, and hex](docs/content/base64-and-hex.md).
- `hex`: lowercase hex, on the same page.

Each of these codecs has functions for public data and functions for secret data, for example a
key. The page tells you which functions to use.
