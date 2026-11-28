use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Caps(u32);

impl Caps {
    pub(crate) const NONE: Self = Self(0);

    pub(crate) const ALL: Self = Self(u32::MAX >> 1);

    const fn bit(bit: u8) -> Self {
        assert!(bit < 31);
        Self(1 << bit)
    }

    #[inline(always)]
    pub(crate) const fn has(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    #[inline(always)]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub(crate) const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::Caps;

    pub(crate) const SSSE3: Caps = Caps::bit(0);
    pub(crate) const AVX2: Caps = Caps::bit(1);
    pub(crate) const BMI1: Caps = Caps::bit(2);
    pub(crate) const BMI2: Caps = Caps::bit(3);
    pub(crate) const PCLMULQDQ: Caps = Caps::bit(4);
    pub(crate) const AVX512F: Caps = Caps::bit(5);
    pub(crate) const AVX512BW: Caps = Caps::bit(6);
    pub(crate) const AVX512VL: Caps = Caps::bit(7);
    pub(crate) const AVX512VBMI: Caps = Caps::bit(8);
    pub(crate) const AVX512VBMI2: Caps = Caps::bit(9);
    pub(crate) const SSE3: Caps = Caps::bit(10);
    pub(crate) const SSE41: Caps = Caps::bit(11);
    pub(crate) const SSE42: Caps = Caps::bit(12);
    pub(crate) const AVX: Caps = Caps::bit(13);
    pub(crate) const FMA: Caps = Caps::bit(14);
    pub(crate) const F16C: Caps = Caps::bit(15);
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64 {
    use super::Caps;

    pub(crate) const NEON: Caps = Caps::bit(0);
    /// FEAT_AES and FEAT_PMULL, which the `aes` feature of Rust holds together.
    pub(crate) const AES: Caps = Caps::bit(1);
    pub(crate) const DIT: Caps = Caps::bit(2);
    /// FEAT_SB: the speculation barrier `sb`, which follows each write that sets DIT.
    pub(crate) const SB: Caps = Caps::bit(3);
}

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm32 {
    use super::Caps;

    pub(crate) const SIMD128: Caps = Caps::bit(0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! features {
    ($detected:ident: $($name:tt => $feature:expr),+ $(,)?) => {
        features!($($name => $feature),+);

        #[cfg(feature = "std")]
        fn detect() -> Caps {
            Caps::NONE $(.union(if std::arch::$detected!($name) { $feature } else { Caps::NONE }))+
        }
    };
    ($($name:tt => $feature:expr),+) => {
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
    "sse3" => x86::SSE3,
    "sse4.1" => x86::SSE41,
    "sse4.2" => x86::SSE42,
    "avx" => x86::AVX,
    "fma" => x86::FMA,
    "f16c" => x86::F16C,
);

#[cfg(target_arch = "aarch64")]
features!(is_aarch64_feature_detected:
    "neon" => aarch64::NEON,
    "aes" => aarch64::AES,
    "dit" => aarch64::DIT,
    "sb" => aarch64::SB,
);

#[cfg(target_arch = "wasm32")]
features!("simd128" => wasm32::SIMD128);

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
#[inline(always)]
pub(crate) const fn caps_static() -> Caps {
    Caps::NONE
}

pub(crate) const DETECTS: bool = cfg!(all(
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
));

#[inline]
pub(crate) fn caps() -> Caps {
    cfg_select! {
        all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")) => {
            use core::sync::atomic::AtomicU32;
            static DETECTED: AtomicU32 = AtomicU32::new(0);
            const KNOWN: u32 = 1 << 31;
            let detected = DETECTED.load(Relaxed);
            if detected != 0 {
                return Caps(detected & !KNOWN);
            }
            let caps = close_implications(caps_static().union(runnable(detect())));
            DETECTED.store(caps.0 | KNOWN, Relaxed);
            caps
        }
        _ => caps_static(),
    }
}

/// Remove a feature if any implied feature is absent. Rust can emit instructions for all implied features.
/// [`caps_static`] needs no closure, because the compiler closes `cfg(target_feature)`.
///
/// Each entry of `IMPLIED` pairs a feature with the features that rustc turns on with it:
/// `rustc --print cfg --target <target> -C target-feature=+<feature>` on the pinned toolchain
/// (1.101.0-nightly, 3e92e1a2d 2026-11-01), less the features of the target without it. The sets
/// are transitive, so one comparison with `caps` is enough. On x86_64, BMI1, BMI2, PCLMULQDQ, and
/// SSE3 imply no other feature. On aarch64, Neon and DIT imply no other feature, and AES implies
/// Neon (the target `aarch64-unknown-none-softfloat` shows it).
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn close_implications(caps: Caps) -> Caps {
    #[cfg(target_arch = "x86_64")]
    const IMPLIED: &[(Caps, Caps)] = {
        use x86::{
            AVX, AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, F16C, FMA, SSE3, SSE41, SSE42, SSSE3,
        };
        const THROUGH_SSE42: Caps = SSE3.union(SSSE3).union(SSE41).union(SSE42);
        const THROUGH_AVX: Caps = THROUGH_SSE42.union(AVX);
        const THROUGH_AVX512F: Caps = THROUGH_AVX.union(AVX2).union(FMA).union(F16C).union(AVX512F);
        const THROUGH_AVX512BW: Caps = THROUGH_AVX512F.union(AVX512BW);
        &[
            (SSSE3, SSE3),
            (SSE41, SSE3.union(SSSE3)),
            (SSE42, SSE3.union(SSSE3).union(SSE41)),
            (AVX, THROUGH_SSE42),
            (AVX2, THROUGH_AVX),
            (FMA, THROUGH_AVX),
            (F16C, THROUGH_AVX),
            (AVX512F, THROUGH_AVX.union(AVX2).union(FMA).union(F16C)),
            (AVX512BW, THROUGH_AVX512F),
            (AVX512VL, THROUGH_AVX512F),
            (AVX512VBMI, THROUGH_AVX512BW),
            (AVX512VBMI2, THROUGH_AVX512BW),
        ]
    };
    #[cfg(target_arch = "aarch64")]
    const IMPLIED: &[(Caps, Caps)] = &[(aarch64::AES, aarch64::NEON)];
    IMPLIED
        .iter()
        .filter(|&&(_, implied)| !caps.has(implied))
        .fold(caps, |closed, &(feature, _)| closed.difference(feature))
}

#[cfg(any(feature = "base64", feature = "hex"))]
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

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[inline(always)]
const fn runnable(detected: Caps) -> Caps {
    detected
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timing {
    Public,
    #[cfg_attr(
        not(any(feature = "base64", feature = "hex")),
        expect(dead_code, reason = "a build with only the JSON codec has no secret class")
    )]
    Secret,
}

impl Timing {
    #[inline(always)]
    pub(crate) const fn runs(self, kernel: Self) -> bool {
        matches!((self, kernel), (Self::Public, _) | (Self::Secret, Self::Secret))
    }
}

#[derive(Debug)]
pub(crate) struct Kept(AtomicU8);

impl Kept {
    pub(crate) const fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    #[inline(always)]
    pub(crate) fn holds(&self, index: usize) -> bool {
        usize::from(self.0.load(Relaxed)) == index + 1
    }

    #[inline]
    pub(crate) fn index(&self, choose: impl FnOnce() -> usize) -> usize {
        if let Some(index) = usize::from(self.0.load(Relaxed)).checked_sub(1) {
            return index;
        }
        let index = choose();
        if let Ok(kept) = u8::try_from(index + 1) {
            self.0.store(kept, Relaxed);
        }
        index
    }
}

macro_rules! choice {
    ($reference:ident) => {
        const fn fastest(caps: $crate::cpu::Caps, class: $crate::cpu::Timing) -> usize {
            let mut index = 0;
            while index + 1 < Self::ALL.len() {
                let (features, timing) = Self::ALL[index].needs();
                if caps.has(features) && class.runs(timing) {
                    break;
                }
                index += 1;
            }
            index
        }

        const fn fixed(class: $crate::cpu::Timing) -> Option<usize> {
            let index = Self::fastest($crate::cpu::caps_static(), class);
            if !$crate::cpu::DETECTS || index == Self::fastest($crate::cpu::Caps::ALL, class) {
                Some(index)
            } else {
                None
            }
        }

        #[inline(always)]
        fn chosen(class: $crate::cpu::Timing) -> Self {
            use $crate::cpu::{Caps, Kept, Timing};
            static KEPT: [Kept; 2] = [Kept::new(), Kept::new()];
            let (fixed, kept, top) = match class {
                Timing::Public => (
                    const { Self::fixed(Timing::Public) },
                    &KEPT[0],
                    const { Self::fastest(Caps::ALL, Timing::Public) },
                ),
                Timing::Secret => (
                    const { Self::fixed(Timing::Secret) },
                    &KEPT[1],
                    const { Self::fastest(Caps::ALL, Timing::Secret) },
                ),
            };
            let index = match fixed {
                Some(index) => index,
                None if kept.holds(top) => return Self::ALL[top],
                None => kept.index(|| Self::detected(class)),
            };
            Self::ALL.get(index).copied().unwrap_or(Self::$reference)
        }

        #[cold]
        #[inline(never)]
        fn detected(class: $crate::cpu::Timing) -> usize {
            Self::fastest($crate::cpu::caps(), class)
        }
    };
}
pub(crate) use choice;

/// A thread-bound token for secret codec calls. [`with_dit`] lends it for one scope.
/// On Arm CPUs with DIT, the token requires `PSTATE.DIT` to remain set.
/// Other CPUs have no corresponding hardware guarantee.
///
/// ```compile_fail
/// rscodec::with_dit(|dit| std::thread::scope(|scope| scope.spawn(|| dit).join().ok()));
/// ```
#[cfg(any(feature = "base64", feature = "hex"))]
#[derive(Debug)]
pub struct Dit(core::marker::PhantomData<*const ()>);

/// The refusal of [`with_dit`]: this build cannot tell whether the CPU has DIT, so it cannot hold
/// `PSTATE.DIT`. It occurs only on aarch64. Recovery: build with `-C target-feature=+dit` for CPUs
/// that have DIT, or with the `std` feature on Linux, Android, or an Apple OS, where `std` reports
/// DIT.
#[cfg(any(feature = "base64", feature = "hex"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DitUnknown;

#[cfg(any(feature = "base64", feature = "hex"))]
impl core::fmt::Display for DitUnknown {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            "this build cannot tell whether the CPU has DIT: build with `-C target-feature=+dit`, or with \
             the `std` feature on Linux, Android, or an Apple OS",
        )
    }
}

#[cfg(any(feature = "base64", feature = "hex"))]
impl core::error::Error for DitUnknown {}

/// Runs secret operations in one scope. On Arm CPUs with DIT, sets `PSTATE.DIT` and
/// restores the previous state on return or unwind. Code inside the scope must keep DIT set.
/// Nested scopes leave an already-set bit alone.
///
/// On AArch64, unknown DIT support gives [`DitUnknown`] before the closure runs.
/// On CPUs known to lack DIT and on other architectures, the closure runs directly.
/// The compiler barriers order the codec's input loads and output stores.
/// They do not order arbitrary register-only computations.
///
/// ```
/// # #[cfg(feature = "hex")]
/// # {
/// use rscodec::{hex, with_dit};
///
/// let mut out = [0; 2];
/// match with_dit(|dit| hex::decode_secret(dit, b"c0de", &mut out)) {
///     Ok(result) => assert_eq!(result?, [0xc0, 0xde]),
///     Err(_) => assert!(cfg!(target_arch = "aarch64")),
/// }
/// # }
/// # Ok::<(), Box<dyn core::error::Error>>(())
/// ```
#[cfg(any(feature = "base64", feature = "hex"))]
#[inline]
pub fn with_dit<T>(scope: impl FnOnce(&Dit) -> T) -> Result<T, DitUnknown> {
    let proof = Dit(core::marker::PhantomData);
    #[cfg(target_arch = "aarch64")]
    {
        // `std` of the pinned toolchain reports DIT only on Linux, Android, and the Apple systems.
        const KNOWN: bool = cfg!(target_feature = "dit")
            || cfg!(all(
                feature = "std",
                any(target_os = "linux", target_os = "android", target_vendor = "apple")
            ));
        if !KNOWN {
            return Err(DitUnknown);
        }
        if has(aarch64::DIT) {
            return Ok(dit::hold(|| scope(&proof)));
        }
    }
    Ok(scope(&proof))
}

/// `PSTATE.DIT`, which has no intrinsic. The system register `S3_3_C4_C2_5` is DIT, and bit 24 is
/// its value.
#[cfg(all(target_arch = "aarch64", any(feature = "base64", feature = "hex")))]
#[expect(
    unsafe_code,
    reason = "the asm reads and writes only PSTATE.DIT and orders speculation, which changes no value, and only on a CPU that has DIT"
)]
mod dit {
    use core::{arch::asm, mem::DropGuard};

    use super::{aarch64, has};

    const ON: u64 = 1 << 24;

    // The assembly must order input loads and output stores with respect to changes to PSTATE.DIT.
    #[inline]
    pub(super) fn hold<T>(call: impl FnOnce() -> T) -> T {
        let earlier: u64;
        // SAFETY: the caller checked that the CPU has DIT, so the instruction is defined.
        unsafe { asm!("mrs {earlier}, S3_3_C4_C2_5", earlier = out(reg) earlier, options(nostack, preserves_flags)) };
        if earlier & ON != 0 {
            return call();
        }
        // SAFETY: as above.
        unsafe { asm!("msr S3_3_C4_C2_5, {on}", on = in(reg) ON, options(nostack, preserves_flags)) };
        // A later instruction can run speculatively before the write of DIT takes effect, so a
        // barrier comes first (Apple: "add a speculation barrier after turning on DIT").
        if has(aarch64::SB) {
            // SAFETY: the CPU has FEAT_SB. `.inst 0xd50330ff` encodes `sb`, which the assembler
            // accepts only with the `sb` target feature.
            unsafe { asm!(".inst 0xd50330ff", options(nostack, preserves_flags)) };
        } else {
            // SAFETY: each Armv8-A CPU has DSB and ISB: the barrier of Apple for a CPU without SB.
            unsafe { asm!("dsb nsh", "isb", options(nostack, preserves_flags)) };
        }
        let _restore = DropGuard::new(earlier, |earlier| {
            // SAFETY: as above.
            unsafe { asm!("msr S3_3_C4_C2_5, {earlier}", earlier = in(reg) earlier, options(nostack, preserves_flags)) }
        });
        call()
    }
}

#[cfg(all(test, feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod tests {
    use super::{Caps, close_implications};

    #[test]
    fn a_feature_without_a_feature_that_it_implies_is_not_reported() {
        let without = |missing: Caps| close_implications(Caps::ALL.difference(missing));
        assert_eq!(without(Caps::NONE), Caps::ALL);
        #[cfg(target_arch = "x86_64")]
        {
            use super::x86::{
                AVX, AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, BMI1, BMI2, F16C, FMA, PCLMULQDQ,
                SSE3, SSE41, SSE42, SSSE3,
            };
            for missing in [AVX2, FMA, F16C] {
                let caps = without(missing);
                for avx512 in [AVX512F, AVX512BW, AVX512VL, AVX512VBMI, AVX512VBMI2] {
                    assert!(!caps.has(avx512), "{avx512:?} without {missing:?}");
                }
            }
            assert!(!without(AVX512BW).has(AVX512VBMI2));
            assert!(without(AVX512VBMI).has(AVX512VBMI2));
            let caps = without(SSE3);
            for vector in [
                SSSE3,
                SSE41,
                SSE42,
                AVX,
                AVX2,
                FMA,
                F16C,
                AVX512F,
                AVX512BW,
                AVX512VL,
                AVX512VBMI,
                AVX512VBMI2,
            ] {
                assert!(!caps.has(vector), "{vector:?} without SSE3");
            }
            assert!(caps.has(BMI1.union(BMI2).union(PCLMULQDQ)));
        }
        #[cfg(target_arch = "aarch64")]
        {
            use super::aarch64::{AES, DIT, NEON};
            assert!(!without(NEON).has(AES));
            assert!(without(NEON).has(DIT));
            assert!(without(DIT).has(AES));
        }
    }
}
