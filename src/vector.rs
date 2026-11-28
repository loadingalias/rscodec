#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "each load or store accesses only bytes of the reference that it takes"
)]
pub(crate) mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_storeu_si128, _mm256_loadu_si256, _mm512_mask_loadu_epi8,
    };
    #[cfg(any(feature = "base64", feature = "hex"))]
    use core::arch::x86_64::{_mm256_storeu_si256, _mm512_loadu_si512, _mm512_mask_storeu_epi8, _mm512_storeu_si512};

    #[inline]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> __m128i {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { _mm_loadu_si128(src.as_ptr().add(AT).cast()) }
    }

    #[inline]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m128i) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    #[cfg(feature = "hex")]
    #[inline]
    pub(crate) fn store_low(dst: &mut [u8; 8], value: __m128i) {
        // SAFETY: the unaligned store writes exactly the eight bytes of `dst`.
        unsafe { core::arch::x86_64::_mm_storel_epi64(dst.as_mut_ptr().cast(), value) }
    }

    #[inline]
    /// # Safety
    /// The CPU must support AVX.
    #[target_feature(enable = "avx")]
    pub(crate) fn load32<const AT: usize, const N: usize>(src: &[u8; N]) -> __m256i {
        const { assert!(AT + 32 <= N) };
        // SAFETY: the bytes `AT..AT + 32` are in `src`, and the load needs no alignment.
        unsafe { _mm256_loadu_si256(src.as_ptr().add(AT).cast()) }
    }

    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    /// # Safety
    /// The CPU must support AVX.
    #[target_feature(enable = "avx")]
    pub(crate) fn store32<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m256i) {
        const { assert!(AT + 32 <= N) };
        // SAFETY: the bytes `AT..AT + 32` are in `dst`, and the store needs no alignment.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    /// # Safety
    /// The CPU must support AVX512F.
    #[target_feature(enable = "avx512f")]
    pub(crate) fn load64<const AT: usize, const N: usize>(src: &[u8; N]) -> __m512i {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `src`, and the load needs no alignment.
        unsafe { _mm512_loadu_si512(src.as_ptr().add(AT).cast()) }
    }

    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    /// # Safety
    /// The CPU must support AVX512F.
    #[target_feature(enable = "avx512f")]
    pub(crate) fn store64<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m512i) {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `dst`, and the store needs no alignment.
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    #[inline]
    fn lanes(len: usize) -> u64 {
        let lanes = len.min(64);
        if lanes == 0 { 0 } else { u64::MAX >> (64 - lanes) }
    }

    #[cfg(feature = "base64")]
    #[inline]
    pub(crate) fn lane(at: usize) -> u64 {
        u32::try_from(at).ok().and_then(|at| 1_u64.checked_shl(at)).unwrap_or(0)
    }

    #[inline]
    /// # Safety
    /// The CPU must support AVX512BW.
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn load_masked(src: &[u8], fill: __m512i) -> __m512i {
        // SAFETY: the mask holds only the lanes of `src`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_loadu_epi8(fill, lanes(src.len()), src.as_ptr().cast()) }
    }

    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    /// # Safety
    /// The CPU must support AVX512BW.
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn store_masked(dst: &mut [u8], value: __m512i) {
        // SAFETY: the mask holds only the lanes of `dst`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_storeu_epi8(dst.as_mut_ptr().cast(), lanes(dst.len()), value) }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn the_mask_holds_only_the_lanes_of_the_slice() {
            for len in (0..=66).chain([usize::MAX]) {
                let lanes = (0..len.min(64)).fold(0_u64, |mask, lane| mask | (1 << lane));
                assert_eq!(super::lanes(len), lanes, "{len} lanes");
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "each memory access stays in its reference, and the validity check uses only registers"
)]
pub(crate) mod aarch64 {
    use core::arch::aarch64::{uint8x16_t, uint8x16x4_t, vld1q_u8, vld1q_u8_x4};
    #[cfg(feature = "base64")]
    use core::arch::aarch64::{uint8x16x3_t, vld3q_u8, vld4q_u8, vst3q_u8};

    /// Tests bit 7 of every byte. The other bits of the minimum can still hold secret data.
    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline(always)]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn all_valid(flags: uint8x16_t) -> bool {
        let minimum: u32;
        // SAFETY: these Neon instructions use only registers. UMOV transfers exactly one byte under DIT.
        unsafe {
            core::arch::asm!(
                "uminv {flags:b}, {flags:v}.16b",
                "umov {minimum:w}, {flags:v}.b[0]",
                flags = inout(vreg) flags => _,
                minimum = lateout(reg) minimum,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        minimum & 0x80 != 0
    }

    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> uint8x16_t {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { vld1q_u8(src.as_ptr().add(AT)) }
    }

    #[cfg(any(feature = "base64", feature = "hex", target_feature = "neon"))]
    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: uint8x16_t) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { core::arch::aarch64::vst1q_u8(dst.as_mut_ptr().add(AT), value) }
    }

    /// LLVM can lower a lane intrinsic to general FMOV, which Arm excludes from DIT.
    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn store_first<const N: usize>(dst: &mut [u8; N], value: uint8x16_t) {
        use core::arch::asm;
        const { assert!(matches!(N, 1 | 2 | 4 | 8)) };
        let dst = dst.as_mut_ptr();
        // SAFETY: each store writes the `N` bytes of `dst`, and needs no alignment.
        unsafe {
            match N {
                8 => {
                    asm!("str {value:d}, [{dst}]", value = in(vreg) value, dst = in(reg) dst, options(nostack, preserves_flags))
                }
                4 => {
                    asm!("str {value:s}, [{dst}]", value = in(vreg) value, dst = in(reg) dst, options(nostack, preserves_flags))
                }
                2 => {
                    asm!("str {value:h}, [{dst}]", value = in(vreg) value, dst = in(reg) dst, options(nostack, preserves_flags))
                }
                _ => {
                    asm!("str {value:b}, [{dst}]", value = in(vreg) value, dst = in(reg) dst, options(nostack, preserves_flags))
                }
            }
        }
    }

    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn table64<const AT: usize, const N: usize>(src: &[u8; N]) -> uint8x16x4_t {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `src`, and the load needs no alignment.
        unsafe { vld1q_u8_x4(src.as_ptr().add(AT)) }
    }

    #[cfg(feature = "base64")]
    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn load_triples(src: &[u8; 48]) -> uint8x16x3_t {
        // SAFETY: `src` has 48 bytes, and the load needs no alignment.
        unsafe { vld3q_u8(src.as_ptr()) }
    }

    #[cfg(feature = "base64")]
    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn store_triples(dst: &mut [u8; 48], value: uint8x16x3_t) {
        // SAFETY: `dst` has 48 bytes, and the store needs no alignment.
        unsafe { vst3q_u8(dst.as_mut_ptr(), value) }
    }

    #[cfg(feature = "base64")]
    #[inline]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    pub(crate) fn load_quads(src: &[u8; 64]) -> uint8x16x4_t {
        // SAFETY: `src` has 64 bytes, and the load needs no alignment.
        unsafe { vld4q_u8(src.as_ptr()) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[expect(
    unsafe_code,
    reason = "each load or store accesses only bytes of the reference that it takes"
)]
pub(crate) mod wasm32 {
    use core::arch::wasm32::{v128, v128_load};

    #[inline]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> v128 {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { v128_load(src.as_ptr().add(AT).cast()) }
    }

    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: v128) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { core::arch::wasm32::v128_store(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    #[cfg(feature = "hex")]
    #[inline]
    pub(crate) fn store_low(dst: &mut [u8; 8], value: v128) {
        // SAFETY: the unaligned lane store writes exactly the eight bytes of `dst`.
        unsafe { core::arch::wasm32::v128_store64_lane::<0>(value, dst.as_mut_ptr().cast()) }
    }
}
