//! Codecs for Rust. This release is an alpha: each alpha release can change the API.
//!
//! Each codec is a leaf feature, so a build compiles only the codecs that it selects:
//!
//! - `base64`: [`base64`], base64 with padding and base64url without padding.
//! - `hex`: [`hex`], lowercase hex.
//! - `json`: [`json`], a JSON reader and writer. This codec is in development.
//!
//! The `std` feature is on by default. Without it, the crate needs only `core`.
//!
//! Each codec writes into a slice that the caller lends, and it allocates no memory. Each codec has
//! a portable kernel and SIMD kernels, and each kernel gives the same result and the same error for
//! each input. A codec that can carry a secret has two timing classes: functions for public data,
//! which run the fastest kernel of the CPU, and functions for secret data, which run only the
//! kernels whose instructions have data independent timing. A secret function takes the proof
//! [`Dit`], which only the scope of [`with_dit`] gives: the scope sets `PSTATE.DIT` on aarch64. On
//! aarch64, a build that cannot tell whether the CPU has DIT gets [`DitUnknown`] in place of a scope.

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
