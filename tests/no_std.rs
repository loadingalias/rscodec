//! The build without `std`, which `cargo xtask test` runs after the build with `std`. The crate
//! detects no CPU there, so each kernel comes from `-C target-feature`.
#![cfg(not(feature = "std"))]

use rscodec::{base64, hex};

/// The names of the kernels whose CPU features the build turns on, the fastest first. Only base64
/// has a VBMI kernel.
fn built(vbmi: bool) -> Vec<&'static str> {
    let avx512bw = cfg!(all(target_feature = "avx512f", target_feature = "avx512bw"));
    [
        ("Avx512Vbmi", vbmi && avx512bw && cfg!(target_feature = "avx512vbmi")),
        ("Avx512Bw", avx512bw),
        ("Avx2", cfg!(target_feature = "avx2")),
        ("Ssse3", cfg!(target_feature = "ssse3")),
        ("Neon", cfg!(target_feature = "neon")),
        ("Simd128", cfg!(target_feature = "simd128")),
        ("Portable", true),
    ]
    .into_iter()
    .filter_map(|(name, built)| built.then_some(name))
    .collect()
}

/// Without `std`, a backend exists exactly for each kernel whose CPU features the build turns on.
/// A backend of another kernel runs instructions that the CPU of the user can lack.
#[test]
fn a_backend_exists_exactly_for_each_kernel_of_the_build() {
    let base64: Vec<_> = base64::Kernel::ALL
        .iter()
        .filter(|&&kernel| base64::Backend::new(kernel).is_some())
        .map(|kernel| format!("{kernel:?}"))
        .collect();
    let hex: Vec<_> = hex::Kernel::ALL
        .iter()
        .filter(|&&kernel| hex::Backend::new(kernel).is_some())
        .map(|kernel| format!("{kernel:?}"))
        .collect();
    assert_eq!(base64, built(true));
    assert_eq!(hex, built(false));
}
