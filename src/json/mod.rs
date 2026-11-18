//! JSON ([RFC 8259]). This module is in development. It holds the kernels and the block cursor of
//! the reader, which a later task builds on them.
//!
//! [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "the reader of task C203 is the first consumer of the cursor")
)]

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod block;
mod cursor;
mod portable;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod simd;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(target_arch = "x86_64")]
mod x86;

use crate::cpu::{self, Caps, Kept, caps_static};
use block::{Block, Carry};

/// A kernel of the reader: its code that classifies one block, for one instruction set. The
/// variants depend on the target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    /// The 64-bit SWAR reference. It runs on each CPU.
    Swar,
    /// AVX-512 BW, VL, and VBMI2, with PCLMULQDQ, BMI1, and BMI2. VBMI2 marks the CPUs that keep
    /// their clock with AVX-512.
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// AVX2, with PCLMULQDQ, BMI1, and BMI2.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Neon, with PMULL for the string mask.
    #[cfg(target_arch = "aarch64")]
    NeonPmull,
    /// Neon without PMULL.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// WebAssembly SIMD. It exists only with `-C target-feature=+simd128`, because WebAssembly has
    /// no run-time detection.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

/// The function of a kernel: the block of the first 64 bytes of its first argument, with spaces
/// after them when it has fewer, after the 4 bytes of its second argument in little-endian order and
/// after the carry. A call needs the CPU features of the kernel.
type Classify = unsafe fn(&[u8], u32, Carry) -> Block;

impl Kernel {
    /// Each kernel of the target architecture, the fastest first.
    const ALL: &[Self] = &[
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        #[cfg(target_arch = "aarch64")]
        Self::NeonPmull,
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Self::Simd128,
        Self::Swar,
    ];

    /// The CPU features that the kernel needs: the features of its `#[target_feature]` functions,
    /// also each feature that they imply and that [`Caps`] names. AVX-512BW implies AVX-512F, and
    /// AVX-512F implies AVX2, so the compiler can use AVX2 in the AVX-512 kernel.
    const fn needs(self) -> Caps {
        #[cfg(target_arch = "x86_64")]
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ};
        match self {
            Self::Swar => Caps::NONE,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => AVX512F
                .union(AVX512BW)
                .union(AVX512VL)
                .union(AVX512VBMI2)
                .union(AVX2)
                .union(PCLMULQDQ)
                .union(BMI1)
                .union(BMI2),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => AVX2.union(PCLMULQDQ).union(BMI1).union(BMI2),
            #[cfg(target_arch = "aarch64")]
            Self::NeonPmull => cpu::aarch64::NEON.union(cpu::aarch64::AES),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => cpu::aarch64::NEON,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => cpu::wasm32::SIMD128,
        }
    }

    /// The index in [`Kernel::ALL`] of the fastest kernel that a CPU with `caps` runs. The last
    /// kernel, the reference, runs on each CPU.
    const fn fastest(caps: Caps) -> usize {
        let mut index = 0;
        while index + 1 < Self::ALL.len() && !caps.has(Self::ALL[index].needs()) {
            index += 1;
        }
        index
    }

    /// The index of the kernel that the build fixes: if the build has no run-time detection, or if
    /// `caps_static()` holds the features of the fastest kernel.
    const fn fixed() -> Option<usize> {
        let index = Self::fastest(caps_static());
        if !cpu::DETECTS || index == Self::fastest(Caps::ALL) {
            Some(index)
        } else {
            None
        }
    }

    /// The function of the kernel.
    ///
    /// The cursor takes the function, not the kernel. In the `#[target_feature]` entry of a kernel,
    /// the compiler then makes each call of the function a direct call, and the kernel inlines into
    /// the entry. A `match` on the kernel in the cursor would call each kernel from each entry, and
    /// no kernel would inline.
    #[inline(always)]
    fn function(self) -> Classify {
        match self {
            Self::Swar => portable::block,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => x86::avx512,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => x86::avx2,
            #[cfg(target_arch = "aarch64")]
            Self::NeonPmull => aarch64::pmull::block,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => aarch64::shift::block,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => wasm32::simd128,
        }
    }
}

/// A kernel that this CPU runs.
///
/// Only [`Backend::new`] and [`Backend::chosen`] make a `Backend`, and each checks the CPU first.
/// Thus a `Backend` exists only for a kernel whose features the CPU has, and a cursor runs only
/// such a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Backend(Kernel);

impl Backend {
    /// The backend of `kernel`, or `None` if this CPU does not have the features of `kernel`.
    #[inline]
    fn new(kernel: Kernel) -> Option<Self> {
        cpu::has(kernel.needs()).then_some(Self(kernel))
    }

    /// The backend of the fastest kernel of this CPU. If the build does not fix it, the first call
    /// chooses it, and each later call reads the choice.
    #[inline(always)]
    fn chosen() -> Self {
        const FIXED: Option<usize> = Kernel::fixed();
        static KEPT: Kept = Kept::new();
        let index = FIXED.unwrap_or_else(|| KEPT.index(|| Kernel::fastest(cpu::caps())));
        Self(Kernel::ALL.get(index).copied().unwrap_or(Kernel::Swar))
    }
}
