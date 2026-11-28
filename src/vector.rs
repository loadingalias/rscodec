//! The loads and stores of vector registers, which are the only unsafe memory operations of the
//! SIMD kernels of each codec. Each takes an array and an offset in it, or a slice for a masked
//! access, and a constant assertion or the mask proves that the access is in it.

/// The loads and stores of x86_64. Each takes an array and an offset in it, and a constant
/// assertion proves that the access is in the array. A masked access takes a slice.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
    not(any(feature = "base64", feature = "hex")),
    expect(dead_code, reason = "a build with only the JSON codec uses only the loads of JSON")
)]
#[expect(
    unsafe_code,
    reason = "each load or store accesses only bytes of the reference that it takes"
)]
#[expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_storeu_si128, _mm256_loadu_si256, _mm256_storeu_si256,
        _mm512_loadu_si512, _mm512_mask_loadu_epi8, _mm512_mask_storeu_epi8, _mm512_storeu_si512,
    };

    /// The 16 bytes at `AT` in `src`.
    #[inline]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> __m128i {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { _mm_loadu_si128(src.as_ptr().add(AT).cast()) }
    }

    /// Stores `value` at `AT` in `dst`.
    #[inline]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m128i) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    /// Stores the low eight bytes directly in the output.
    #[cfg(feature = "hex")]
    #[inline]
    pub(crate) fn store_low(dst: &mut [u8; 8], value: __m128i) {
        // SAFETY: the unaligned store writes exactly the eight bytes of `dst`.
        unsafe { core::arch::x86_64::_mm_storel_epi64(dst.as_mut_ptr().cast(), value) }
    }

    /// The 32 bytes at `AT` in `src`.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn load32<const AT: usize, const N: usize>(src: &[u8; N]) -> __m256i {
        const { assert!(AT + 32 <= N) };
        // SAFETY: the bytes `AT..AT + 32` are in `src`, and the load needs no alignment.
        unsafe { _mm256_loadu_si256(src.as_ptr().add(AT).cast()) }
    }

    /// Stores `value` at `AT` in `dst`.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn store32<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m256i) {
        const { assert!(AT + 32 <= N) };
        // SAFETY: the bytes `AT..AT + 32` are in `dst`, and the store needs no alignment.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    /// The 64 bytes at `AT` in `src`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn load64<const AT: usize, const N: usize>(src: &[u8; N]) -> __m512i {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `src`, and the load needs no alignment.
        unsafe { _mm512_loadu_si512(src.as_ptr().add(AT).cast()) }
    }

    /// Stores `value` at `AT` in `dst`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn store64<const AT: usize, const N: usize>(dst: &mut [u8; N], value: __m512i) {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `dst`, and the store needs no alignment.
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    /// The mask of the first `len` lanes of 64, or of all 64 lanes for a longer `len`. Thus a mask
    /// of a slice holds only lanes of the slice, with no check that can panic.
    #[inline]
    fn lanes(len: usize) -> u64 {
        // A shift of 0 to 63 bits, and no lane for an empty slice.
        let lanes = len.min(64);
        if lanes == 0 { 0 } else { u64::MAX >> (64 - lanes) }
    }

    /// The mask of lane `at` alone, or no lane from 64.
    #[cfg(feature = "base64")]
    #[inline]
    pub(crate) fn lane(at: usize) -> u64 {
        u32::try_from(at).ok().and_then(|at| 1_u64.checked_shl(at)).unwrap_or(0)
    }

    /// The first bytes of `src`, at most 64, in the first lanes, and `fill` in the other lanes.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn load_masked(src: &[u8], fill: __m512i) -> __m512i {
        // SAFETY: the mask holds only the lanes of `src`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_loadu_epi8(fill, lanes(src.len()), src.as_ptr().cast()) }
    }

    /// Stores the first lanes of `value` into `dst`: `dst.len()` lanes, at most 64.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn store_masked(dst: &mut [u8], value: __m512i) {
        // SAFETY: the mask holds only the lanes of `dst`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_storeu_epi8(dst.as_mut_ptr().cast(), lanes(dst.len()), value) }
    }

    #[cfg(test)]
    mod tests {
        /// The mask of `len` lanes holds the lanes `0..len`, at most 64, and no other lane, so a
        /// masked load or store of a slice accesses only the bytes of the slice. The test checks
        /// each length to 66, and the largest.
        #[test]
        fn the_mask_holds_only_the_lanes_of_the_slice() {
            for len in (0..=66).chain([usize::MAX]) {
                let lanes = (0..len.min(64)).fold(0_u64, |mask, lane| mask | (1 << lane));
                assert_eq!(super::lanes(len), lanes, "{len} lanes");
            }
        }
    }
}

/// The Neon validity check and memory operations. Each memory access stays inside its input or output.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(
    not(any(feature = "base64", feature = "hex")),
    expect(dead_code, reason = "a build with only the JSON codec uses only the loads of JSON")
)]
#[expect(
    unsafe_code,
    reason = "each memory access stays in its reference, and the validity check uses only registers"
)]
#[expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) mod aarch64 {
    use core::arch::aarch64::{uint8x16_t, uint8x16x4_t, vld1q_u8, vld1q_u8_x4, vst1q_u8};
    #[cfg(feature = "base64")]
    use core::arch::aarch64::{uint8x16x3_t, vld3q_u8, vld4q_u8, vst3q_u8};

    /// Tests bit 7 of every byte. The other bits of the minimum can still hold secret data.
    #[cfg(any(feature = "base64", feature = "hex"))]
    #[inline(always)]
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

    /// The 16 bytes at `AT` in `src`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> uint8x16_t {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { vld1q_u8(src.as_ptr().add(AT)) }
    }

    /// Stores `value` at `AT` in `dst`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: uint8x16_t) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { vst1q_u8(dst.as_mut_ptr().add(AT), value) }
    }

    /// Stores the first `N` bytes of `value`, 8, 4, 2, or 1, in `dst` with one store from the
    /// vector register (`str d`, `str s`, `str h`, `str b`). The asm fixes the instruction: from a
    /// lane of an intrinsic, LLVM can move the bytes to a general register with `fmov`, which is not
    /// on the Arm DIT list.
    #[inline]
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

    /// The 64 bytes at `AT` in `src` as four vectors, for example for a table lookup.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn table64<const AT: usize, const N: usize>(src: &[u8; N]) -> uint8x16x4_t {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `src`, and the load needs no alignment.
        unsafe { vld1q_u8_x4(src.as_ptr().add(AT)) }
    }

    #[cfg(feature = "base64")]
    /// The 48 bytes of `src`, with byte `3i + k` in lane `i` of vector `k`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load_triples(src: &[u8; 48]) -> uint8x16x3_t {
        // SAFETY: `src` has 48 bytes, and the load needs no alignment.
        unsafe { vld3q_u8(src.as_ptr()) }
    }

    #[cfg(feature = "base64")]
    /// Stores the bytes of the three vectors in turn.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn store_triples(dst: &mut [u8; 48], value: uint8x16x3_t) {
        // SAFETY: `dst` has 48 bytes, and the store needs no alignment.
        unsafe { vst3q_u8(dst.as_mut_ptr(), value) }
    }

    #[cfg(feature = "base64")]
    /// The 64 bytes of `src`, with byte `4i + k` in lane `i` of vector `k`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load_quads(src: &[u8; 64]) -> uint8x16x4_t {
        // SAFETY: `src` has 64 bytes, and the load needs no alignment.
        unsafe { vld4q_u8(src.as_ptr()) }
    }
}

/// The loads and stores of WebAssembly. Each takes an array and an offset in it, and a constant
/// assertion proves that the access is in the array.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[cfg_attr(
    not(any(feature = "base64", feature = "hex")),
    expect(dead_code, reason = "a build with only the JSON codec uses only the loads of JSON")
)]
#[expect(
    unsafe_code,
    reason = "each load or store accesses only bytes of the reference that it takes"
)]
pub(crate) mod wasm32 {
    use core::arch::wasm32::{v128, v128_load, v128_store};

    /// The 16 bytes at `AT` in `src`.
    #[inline]
    pub(crate) fn load16<const AT: usize, const N: usize>(src: &[u8; N]) -> v128 {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `src`, and the load needs no alignment.
        unsafe { v128_load(src.as_ptr().add(AT).cast()) }
    }

    /// Stores `value` at `AT` in `dst`.
    #[inline]
    pub(crate) fn store16<const AT: usize, const N: usize>(dst: &mut [u8; N], value: v128) {
        const { assert!(AT + 16 <= N) };
        // SAFETY: the bytes `AT..AT + 16` are in `dst`, and the store needs no alignment.
        unsafe { v128_store(dst.as_mut_ptr().add(AT).cast(), value) }
    }

    /// Stores the low eight bytes directly in the output.
    #[cfg(feature = "hex")]
    #[inline]
    pub(crate) fn store_low(dst: &mut [u8; 8], value: v128) {
        // SAFETY: the unaligned lane store writes exactly the eight bytes of `dst`.
        unsafe { core::arch::wasm32::v128_store64_lane::<0>(value, dst.as_mut_ptr().cast()) }
    }
}
