//! Allocation-free codecs for caller-owned buffers.
//!
//! Enable `base64` for padded base64 and unpadded base64url, `hex` for lowercase hex,
//! or `json` for the JSON reader and writer. The default feature, `std`, enables runtime
//! CPU detection. Without it, the crate uses only `core` and build-time target features.
//!
//! Base64 and hex provide separate functions for public and secret data.
//! [Guides, examples, and platform support](https://github.com/loadingalias/rscodec#rscodec)

#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(any(feature = "base64", feature = "hex", feature = "json"))]
mod cpu;
#[cfg(any(feature = "base64", feature = "hex"))]
mod ct;
#[cfg(all(
    any(feature = "base64", feature = "hex"),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
mod simd;
#[cfg(all(
    any(feature = "base64", feature = "hex", feature = "json"),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
mod vector;

#[cfg(any(feature = "base64", feature = "hex"))]
pub use cpu::{Dit, DitUnknown, with_dit};

#[cfg(feature = "base64")]
pub mod base64;
#[cfg(feature = "hex")]
pub mod hex;
#[cfg(feature = "json")]
pub mod json;
