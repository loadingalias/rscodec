//! The helpers of the codec tests.

/// Deterministic pseudo-random bytes (xorshift64*), so that each run tests the same inputs.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d).to_be_bytes()[0]
    }

    pub(crate) fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }
}

/// `PSTATE.DIT` of this thread: the system register `S3_3_C4_C2_5`, whose bit 24 is DIT.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "the asm reads or writes only PSTATE.DIT, and only on a CPU that has DIT"
)]
pub(crate) mod dit {
    use core::arch::asm;

    /// The bit of DIT in its register.
    const ON: u64 = 1 << 24;

    /// Whether DIT is set, or `None` if the CPU has no DIT.
    pub(crate) fn get() -> Option<bool> {
        if !std::arch::is_aarch64_feature_detected!("dit") {
            return None;
        }
        let value: u64;
        // SAFETY: the CPU has DIT, so the instruction is defined. It reads only DIT.
        unsafe { asm!("mrs {value}, S3_3_C4_C2_5", value = out(reg) value, options(nomem, nostack, preserves_flags)) };
        Some(value & ON != 0)
    }

    /// Sets DIT to `on`.
    pub(crate) fn set(on: bool) {
        assert!(std::arch::is_aarch64_feature_detected!("dit"));
        let value = if on { ON } else { 0 };
        // SAFETY: the assertion proves that the CPU has DIT, so the instruction is defined. It
        // writes only DIT, which changes no value.
        unsafe { asm!("msr S3_3_C4_C2_5, {value}", value = in(reg) value, options(nomem, nostack, preserves_flags)) };
    }
}
