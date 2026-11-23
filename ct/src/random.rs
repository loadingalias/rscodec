//! SplitMix64: random inputs and the order of the DudeCT classes. The checks need no
//! cryptographic strength, only inputs that do not repeat.

/// A SplitMix64 generator (Steele, Lea, and Flood, 2014).
#[derive(Debug)]
pub(crate) struct Random(u64);

impl Random {
    /// A generator that starts at `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub(crate) const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A random value below `bound`, which is not zero. The bias is below 2^-32 for each bound
    /// of the harness.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        let wide = u128::from(self.next()) * bound as u128;
        usize::try_from(wide >> 64).unwrap_or(0)
    }

    /// Fills `bytes` with random bytes.
    pub(crate) fn fill(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(word.get(..chunk.len()).unwrap_or_default());
        }
    }
}
