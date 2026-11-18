//! The CPU detection of the codecs (selected on 2026-11-17, [CODEC Layers]).
//!
//! With `std`, the crate detects the CPU at run time, through `std::arch`, which detects it once for
//! each process. Without `std`, only the features of `-C target-feature` are on.

/// An instruction set of a codec backend.
///
/// The set of variants depends on the target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Scalar arithmetic with no table. It runs on each CPU.
    Portable,
    /// SSSE3, with 16-byte vectors.
    #[cfg(target_arch = "x86_64")]
    Ssse3,
    /// AVX2, with 32-byte vectors.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512F and AVX-512BW, with 64-byte vectors. The kernels use only the instructions that
    /// Intel lists with data operand independent timing (DOIT).
    #[cfg(target_arch = "x86_64")]
    Avx512Bw,
    /// AVX-512BW and AVX-512 VBMI. The base64 kernels use the byte permutes of VBMI, which the DOIT
    /// list of Intel does not hold. The hex kernels are those of [`Isa::Avx512Bw`].
    #[cfg(target_arch = "x86_64")]
    Avx512Vbmi,
    /// Neon (Advanced SIMD), with 16-byte vectors.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// WebAssembly SIMD, with 16-byte vectors. It exists only with `-C target-feature=+simd128`,
    /// because WebAssembly has no run-time detection.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

impl Isa {
    /// Each instruction set of the target architecture, the fastest first.
    pub const ALL: &[Self] = &[
        #[cfg(target_arch = "x86_64")]
        Self::Avx512Vbmi,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512Bw,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        #[cfg(target_arch = "x86_64")]
        Self::Ssse3,
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Self::Simd128,
        Self::Portable,
    ];

    /// Whether this CPU runs the instruction set.
    #[inline]
    fn is_available(self) -> bool {
        match self {
            Self::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Self::Ssse3 => x86!("ssse3"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => x86!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bw => x86!("avx512f", "avx512bw"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Vbmi => x86!("avx512f", "avx512bw", "avx512vbmi"),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => aarch64!("neon"),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => true,
        }
    }
}

/// An instruction set that this CPU runs. Each codec function with a `Backend` argument runs on its
/// instruction set.
///
/// Only [`Backend::detect`] and [`Backend::new`] make a `Backend`, and each checks the CPU first.
/// Thus a `Backend` exists only for an instruction set that the CPU has, and no safe call can run a
/// kernel that the CPU does not support.
///
/// ```
/// use rscodec::{Backend, Isa, hex};
///
/// // Each CPU runs the portable backend. `Backend::new` gives `None` for an instruction set that
/// // the CPU does not have.
/// let portable = Backend::new(Isa::Portable).expect("each CPU runs the portable backend");
/// let mut text = [0; 2];
/// assert_eq!(hex::encode_with(portable, &[0xc0], &mut text)?, b"c0");
/// # Ok::<(), hex::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Backend(Isa);

impl Backend {
    /// The fastest backend that this CPU runs. Each codec function without a `Backend` argument
    /// uses it. With `std`, the first call detects it, and each later call reads the result.
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        let fastest = || Isa::ALL.iter().position(|isa| isa.is_available());
        let isa = |index: usize| Self(Isa::ALL.get(index).copied().unwrap_or(Isa::Portable));
        cfg_select! {
            feature = "std" => {
                use core::sync::atomic::{AtomicU8, Ordering::Relaxed};
                /// One more than the index in `Isa::ALL` of the fastest instruction set, or zero
                /// before the first call. Each call that stores it stores the same value.
                static DETECTED: AtomicU8 = AtomicU8::new(0);
                if let Some(index) = usize::from(DETECTED.load(Relaxed)).checked_sub(1) {
                    return isa(index);
                }
                let index = fastest().unwrap_or(Isa::ALL.len() - 1);
                DETECTED.store(u8::try_from(index + 1).unwrap_or(0), Relaxed);
                isa(index)
            }
            _ => isa(fastest().unwrap_or(Isa::ALL.len() - 1)),
        }
    }

    /// The backend of `isa`, or `None` if this CPU does not run `isa`.
    #[inline]
    #[must_use]
    pub fn new(isa: Isa) -> Option<Self> {
        isa.is_available().then_some(Self(isa))
    }

    /// The instruction set of the backend.
    #[inline]
    #[must_use]
    pub const fn isa(self) -> Isa {
        self.0
    }
}

/// Whether this CPU has each of the x86 features.
#[cfg(target_arch = "x86_64")]
macro_rules! x86 {
    ($($feature:tt),+) => {
        $(has!(is_x86_feature_detected, $feature))&&+
    };
}
#[cfg(target_arch = "x86_64")]
use x86;

/// Whether this CPU has each of the aarch64 features.
#[cfg(target_arch = "aarch64")]
macro_rules! aarch64 {
    ($($feature:tt),+) => {
        $(has!(is_aarch64_feature_detected, $feature))&&+
    };
}
#[cfg(target_arch = "aarch64")]
use aarch64;

/// Whether this CPU has `feature`: from `std::arch` with `std`, and from `-C target-feature`
/// without it.
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
macro_rules! has {
    ($detect:ident, $feature:tt) => {
        std::arch::$detect!($feature)
    };
}
#[cfg(all(not(feature = "std"), any(target_arch = "x86_64", target_arch = "aarch64")))]
macro_rules! has {
    ($detect:ident, $feature:tt) => {
        cfg!(target_feature = $feature)
    };
}
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use has;

/// Runs `call` with data independent timing (DIT) on, if the CPU has it ([CODEC Base-N]). Arm
/// guarantees the timing of its DIT instructions only while `PSTATE.DIT` is 1.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn constant_time<T>(call: impl FnOnce() -> T) -> T {
    if aarch64!("dit") { dit::run(call) } else { call() }
}

/// Runs `call`. Only Arm has a timing mode that a program can turn on.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn constant_time<T>(call: impl FnOnce() -> T) -> T {
    call()
}

/// `PSTATE.DIT`, which has no intrinsic. The system register `S3_3_C4_C2_5` is DIT, and bit 24 is
/// its value.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "the asm reads and writes only PSTATE.DIT, which changes no value, and only on a CPU that has DIT"
)]
mod dit {
    use core::{arch::asm, mem::DropGuard};

    /// The bit of DIT in its register.
    const ON: u64 = 1 << 24;

    /// Runs `call` with `PSTATE.DIT` set, then restores its earlier value. If DIT is set already,
    /// for example by a caller that sets it for its thread, it writes nothing. The caller checks
    /// that the CPU has DIT.
    ///
    /// The asm can read and write memory for the compiler, so no load of the input moves before the
    /// write of DIT, and no store of the output moves after its restore.
    #[inline]
    pub(super) fn run<T>(call: impl FnOnce() -> T) -> T {
        let earlier: u64;
        // SAFETY: the caller checked that the CPU has DIT, so the instruction is defined.
        unsafe { asm!("mrs {earlier}, S3_3_C4_C2_5", earlier = out(reg) earlier, options(nostack, preserves_flags)) };
        if earlier & ON != 0 {
            return call();
        }
        // SAFETY: as above.
        unsafe { asm!("msr S3_3_C4_C2_5, {on}", on = in(reg) ON, options(nostack, preserves_flags)) };
        let _restore = DropGuard::new(earlier, |earlier| {
            // SAFETY: as above.
            unsafe { asm!("msr S3_3_C4_C2_5, {earlier}", earlier = in(reg) earlier, options(nostack, preserves_flags)) }
        });
        call()
    }
}
