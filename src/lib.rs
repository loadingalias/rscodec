//! Codecs for Rust. This crate is in development and holds no codec yet.
//!
//! Each codec is a leaf feature, so a build compiles only the codecs that it selects. The `std`
//! feature is on by default. Without it, the crate needs only `core`.

#![no_std]

#[cfg(feature = "std")]
extern crate std;
