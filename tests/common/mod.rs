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
