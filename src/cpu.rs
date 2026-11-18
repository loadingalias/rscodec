//! The facts layer: the CPU features that the kernels of the codecs can use ([CODEC Layers]).
//!
//! It has the shape of `rscrypto::platform` (selected on 2026-11-18). [`Caps`] is a set of single
//! CPU features. [`caps_static`] is the set that `-C target-feature` turns on, as a constant. With
//! `std`, [`caps`] detects the CPU once for each process and includes the static set. Without
//! `std`, [`caps`] is [`caps_static`].
//!
//! Each codec owns the table of its kernels. Each kernel names the features that it needs and its
//! [`Timing`] class. A codec chooses its kernel for each class from these facts: at compile time
//! where [`caps_static`] holds the features of the fastest kernel of the class, and otherwise once
//! for each process, in a [`Kept`].

use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// A set of CPU features. Each constant of `x86`, `aarch64`, and `wasm32` holds one feature. A
/// build has the features of one architecture, so each architecture numbers its features from bit 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Caps(u16);

impl Caps {
    /// No feature.
    pub(crate) const NONE: Self = Self(0);

    /// Each feature: the set of a CPU that has all of them.
    pub(crate) const ALL: Self = Self(u16::MAX >> 1);

    /// The feature of bit `bit`. Bit 15 is not a feature, so that [`caps`] can mark its cache.
    const fn bit(bit: u8) -> Self {
        assert!(bit < 15);
        Self(1 << bit)
    }

    /// Whether the set holds each feature of `required`.
    #[inline(always)]
    pub(crate) const fn has(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// The features of either set.
    #[inline(always)]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// The features of x86_64.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::Caps;

    /// SSSE3: `pshufb` and `pmaddubsw` on 16-byte vectors.
    pub(crate) const SSSE3: Caps = Caps::bit(0);
    /// AVX2: the integer operations on 32-byte vectors.
    pub(crate) const AVX2: Caps = Caps::bit(1);
    /// BMI1: `blsr` and `tzcnt`.
    pub(crate) const BMI1: Caps = Caps::bit(2);
    /// BMI2: `pdep` and `pext`.
    pub(crate) const BMI2: Caps = Caps::bit(3);
    /// PCLMULQDQ: the carry-less multiply.
    pub(crate) const PCLMULQDQ: Caps = Caps::bit(4);
    /// AVX-512F: the foundation of AVX-512, with 64-byte vectors.
    pub(crate) const AVX512F: Caps = Caps::bit(5);
    /// AVX-512BW: the byte and word operations.
    pub(crate) const AVX512BW: Caps = Caps::bit(6);
    /// AVX-512VL: the AVX-512 operations on 16-byte and 32-byte vectors.
    pub(crate) const AVX512VL: Caps = Caps::bit(7);
    /// AVX-512 VBMI: the byte permutes `vpermb` and `vpermi2b`, and `vpmultishiftqb`.
    pub(crate) const AVX512VBMI: Caps = Caps::bit(8);
    /// AVX-512 VBMI2: `vpcompressb` and the double shifts.
    pub(crate) const AVX512VBMI2: Caps = Caps::bit(9);
}

/// The features of aarch64.
#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64 {
    use super::Caps;

    /// Neon (Advanced SIMD).
    pub(crate) const NEON: Caps = Caps::bit(0);
    /// FEAT_AES and FEAT_PMULL, which the `aes` feature of Rust holds together.
    pub(crate) const AES: Caps = Caps::bit(1);
    /// FEAT_DIT: data independent timing, with `PSTATE.DIT`.
    pub(crate) const DIT: Caps = Caps::bit(2);
}

/// The features of WebAssembly. WebAssembly has no run-time detection, so each feature is a
/// feature of the build.
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm32 {
    use super::Caps;

    /// WebAssembly SIMD, with 16-byte vectors.
    pub(crate) const SIMD128: Caps = Caps::bit(0);
}

/// Defines [`caps_static`] from one list: the name of each feature in `-C target-feature`, and its
/// constant. With the name of the detection macro of `std::arch`, it also defines `detect` for
/// `std`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! features {
    ($detected:ident: $($name:tt => $feature:expr),+ $(,)?) => {
        features!($($name => $feature),+);

        /// The features that `std::arch` detects on this CPU.
        #[cfg(feature = "std")]
        fn detect() -> Caps {
            Caps::NONE $(.union(if std::arch::$detected!($name) { $feature } else { Caps::NONE }))+
        }
    };
    ($($name:tt => $feature:expr),+) => {
        /// The features that `-C target-feature` turns on. The compiler can use them in each
        /// function, so each CPU that runs the program has them.
        #[inline(always)]
        pub(crate) const fn caps_static() -> Caps {
            Caps::NONE $(.union(if cfg!(target_feature = $name) { $feature } else { Caps::NONE }))+
        }
    };
}

#[cfg(target_arch = "x86_64")]
features!(is_x86_feature_detected:
    "ssse3" => x86::SSSE3,
    "avx2" => x86::AVX2,
    "bmi1" => x86::BMI1,
    "bmi2" => x86::BMI2,
    "pclmulqdq" => x86::PCLMULQDQ,
    "avx512f" => x86::AVX512F,
    "avx512bw" => x86::AVX512BW,
    "avx512vl" => x86::AVX512VL,
    "avx512vbmi" => x86::AVX512VBMI,
    "avx512vbmi2" => x86::AVX512VBMI2,
);

#[cfg(target_arch = "aarch64")]
features!(is_aarch64_feature_detected:
    "neon" => aarch64::NEON,
    "aes" => aarch64::AES,
    "dit" => aarch64::DIT,
);

#[cfg(target_arch = "wasm32")]
features!("simd128" => wasm32::SIMD128);

/// No feature: rscodec has no kernel for the other architectures.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
#[inline(always)]
pub(crate) const fn caps_static() -> Caps {
    Caps::NONE
}

/// Whether [`caps`] can hold more than [`caps_static`]: with `std`, on a target with run-time
/// detection. If not, the choice of each kernel is a constant.
pub(crate) const DETECTS: bool = cfg!(all(
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
));

/// The features of the CPU that runs the process. It includes [`caps_static`]. With `std`, the
/// first call detects them, and each later call reads the result.
#[inline]
pub(crate) fn caps() -> Caps {
    cfg_select! {
        all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")) => {
            use core::sync::atomic::AtomicU16;
            /// The features with bit 15 set, or zero before the first call. Each call that stores
            /// it stores the same value.
            static DETECTED: AtomicU16 = AtomicU16::new(0);
            const KNOWN: u16 = 1 << 15;
            let detected = DETECTED.load(Relaxed);
            if detected != 0 {
                return Caps(detected & !KNOWN);
            }
            let caps = caps_static().union(runnable(detect()));
            DETECTED.store(caps.0 | KNOWN, Relaxed);
            caps
        }
        _ => caps_static(),
    }
}

/// Whether the CPU has each feature of `required`. The compiler folds the check of a feature of
/// [`caps_static`], with no load.
#[inline(always)]
pub(crate) fn has(required: Caps) -> bool {
    caps_static().has(required) || caps().has(required)
}

/// The detected features without AVX-512 on an Intel hybrid CPU (Alder Lake to Arrow Lake): the
/// rule of `rscrypto::platform`. The efficient cores of these CPUs have no AVX-512, so a thread
/// that moves to one of them during an AVX-512 kernel faults. Thus the rule removes AVX-512 even if
/// CPUID reports it. It removes no feature of [`caps_static`], so CPUID runs only if the CPU
/// reports an AVX-512 feature that the build does not turn on.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn runnable(detected: Caps) -> Caps {
    use core::arch::x86_64::__cpuid;

    let avx512 = [
        x86::AVX512F,
        x86::AVX512BW,
        x86::AVX512VL,
        x86::AVX512VBMI,
        x86::AVX512VBMI2,
    ]
    .into_iter()
    .fold(Caps::NONE, Caps::union);
    let removable = detected.0 & avx512.0 & !caps_static().0;
    if removable == 0 {
        return detected;
    }
    let vendor = __cpuid(0);
    // "GenuineIntel" in EBX, EDX, and ECX.
    let intel = [vendor.ebx, vendor.edx, vendor.ecx] == [0x756e_6547, 0x4965_6e69, 0x6c65_746e];
    let signature = if intel && vendor.eax >= 1 { __cpuid(1).eax } else { 0 };
    let family = signature >> 8 & 0xf;
    // The model, with the extended model of family 6 in its high nibble.
    let model = (signature >> 4 & 0xf) | (signature >> 12 & 0xf0);
    let hybrid = family == 6
        && matches!(
            model,
            0x97 | 0x9a | 0x9c | 0xaa | 0xac | 0xb7 | 0xba | 0xbd | 0xbf | 0xc5 | 0xc6
        );
    if hybrid {
        Caps(detected.0 & !removable)
    } else {
        detected
    }
}

/// The detected features. Only x86 has a rule that removes a detected feature.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[inline(always)]
const fn runnable(detected: Caps) -> Caps {
    detected
}

/// The timing class of a kernel, and of a function of a codec (selected on 2026-11-18,
/// [CODEC Base-N]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timing {
    /// For public data, for example a digest or a signature. A function of the class runs the
    /// fastest kernel of the CPU. A kernel of the class can use an instruction whose time can
    /// depend on its operands, for example a byte permute of AVX-512 VBMI, which Intel does not
    /// list with data operand independent timing (DOIT).
    Public,
    /// For secret data, for example a key or a bearer token. A kernel of the class uses only the
    /// instructions that Intel lists with DOIT and Arm lists with data independent timing (DIT). A
    /// function of the class runs only these kernels, with `PSTATE.DIT` set on aarch64.
    Secret,
}

impl Timing {
    /// Whether a function of this class can run a kernel of class `kernel`.
    #[inline(always)]
    pub(crate) const fn runs(self, kernel: Self) -> bool {
        matches!((self, kernel), (Self::Public, _) | (Self::Secret, Self::Secret))
    }
}

/// The kernel that a codec chose for one timing class, kept for the process: one more than the
/// index of the kernel in the table of the codec, or zero before the first call.
#[derive(Debug)]
pub(crate) struct Kept(AtomicU8);

impl Kept {
    pub(crate) const fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    /// The index that the first call chose with `choose`. Each racing first call chooses and
    /// stores the same index.
    #[inline]
    pub(crate) fn index(&self, choose: impl FnOnce() -> usize) -> usize {
        if let Some(index) = usize::from(self.0.load(Relaxed)).checked_sub(1) {
            return index;
        }
        let index = choose();
        // A table has fewer than 255 kernels, so the index fits.
        if let Ok(kept) = u8::try_from(index + 1) {
            self.0.store(kept, Relaxed);
        }
        index
    }
}

/// Runs `call` with data independent timing (DIT) on, if the CPU has it ([CODEC Base-N]). Arm
/// guarantees the timing of its DIT instructions only while `PSTATE.DIT` is 1. Only the secret
/// functions of the codecs run in it.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn data_independent_timing<T>(call: impl FnOnce() -> T) -> T {
    if has(aarch64::DIT) { dit::run(call) } else { call() }
}

/// Runs `call`. Only Arm has a timing mode that a program can turn on.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn data_independent_timing<T>(call: impl FnOnce() -> T) -> T {
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
