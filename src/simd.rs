//! The parts that the SIMD kernels of each codec share: the loop over the blocks, and the loads
//! and stores, which are the only unsafe operations of the kernels.

use crate::ct::Class;

/// Runs `block` over `input` and `output`. Each block reads `IN` bytes of input and uses the first
/// `step`. It writes `OUT` bytes of output, and the first `out_len(step)` of them are the result.
///
/// The last blocks run on arrays on the stack: the input after its end is `fill`, and only the
/// result goes to `output`. Thus no SIMD load or store passes the end of a slice, and no scalar
/// code decodes a tail. `output` has the length `out_len` of all of `input`. In the secret
/// `class`, each array is zero before the function returns.
///
/// The arrays take and give their bytes in 16-byte pieces, each with one or two loads or stores of
/// a fixed size. Thus the compiler keeps an array of 16 bytes in registers: a copy of a variable
/// length is a call to `memcpy`, and a vector load of bytes that smaller stores wrote waits for
/// them. A kernel with wider vectors runs only [`whole_blocks`], and gives the rest to a kernel of
/// 16-byte vectors.
#[inline(always)]
pub(crate) fn blocks<const IN: usize, const OUT: usize>(
    class: impl Class,
    input: &[u8],
    output: &mut [u8],
    step: usize,
    fill: u8,
    out_len: impl Fn(usize) -> usize,
    mut block: impl FnMut(&[u8; IN], &mut [u8; OUT]),
) {
    const { assert!(IN.is_multiple_of(16) && OUT.is_multiple_of(16)) };
    let (mut read, mut written) = whole_blocks(input, output, step, &out_len, &mut block);
    while let Some(rest) = input.get(read..).filter(|rest| !rest.is_empty()) {
        let len = rest.len().min(step);
        let mut src = [fill; IN];
        for (piece, chunk) in src.as_chunks_mut::<16>().0.iter_mut().zip(rest[..len].chunks(16)) {
            *piece = read16(chunk, fill).to_le_bytes();
        }
        let mut dst = [0; OUT];
        block(&src, &mut dst);
        let len_out = out_len(len);
        for (chunk, piece) in output[written..written + len_out]
            .chunks_mut(16)
            .zip(dst.as_chunks::<16>().0)
        {
            write16(class, chunk, u128::from_le_bytes(*piece));
        }
        class.clear(&mut src);
        class.clear(&mut dst);
        read += len;
        written += len_out;
    }
}

/// The loop of [`blocks`] over each block whose load and store fit in `input` and `output`. Gives
/// the bytes that it read and wrote.
#[inline(always)]
pub(crate) fn whole_blocks<const IN: usize, const OUT: usize>(
    input: &[u8],
    output: &mut [u8],
    step: usize,
    out_len: impl Fn(usize) -> usize,
    mut block: impl FnMut(&[u8; IN], &mut [u8; OUT]),
) -> (usize, usize) {
    let step_out = out_len(step);
    let (mut read, mut written) = (0, 0);
    while let (Some(src), Some(dst)) = (
        input.get(read..).and_then(<[u8]>::first_chunk),
        output.get_mut(written..).and_then(<[u8]>::first_chunk_mut),
    ) {
        block(src, dst);
        read += step;
        written += step_out;
    }
    (read, written)
}

/// The bytes of `src`, at most 16, in the low bytes of the result, and `fill` in the other bytes.
/// Two overlapping loads of 8, 4, or 1 bytes read each byte, and the OR joins them.
#[inline(always)]
fn read16(src: &[u8], fill: u8) -> u128 {
    let len = src.len();
    assert!(len <= 16);
    let pair = |first: u128, last: u128, last_at: usize| first | last << (8 * last_at);
    let bytes = if let (Some(first), Some(last)) = (src.first_chunk::<8>(), src.last_chunk::<8>()) {
        let word = |bytes: &[u8; 8]| u128::from(u64::from_le_bytes(*bytes));
        pair(word(first), word(last), len - 8)
    } else if let (Some(first), Some(last)) = (src.first_chunk::<4>(), src.last_chunk::<4>()) {
        let word = |bytes: &[u8; 4]| u128::from(u32::from_le_bytes(*bytes));
        pair(word(first), word(last), len - 4)
    } else if let [first, ..] = src {
        let byte = |at: usize| u128::from(src[at]) << (8 * at);
        u128::from(*first) | byte(len / 2) | byte(len - 1)
    } else {
        0
    };
    // The bytes from `len`: no shift of 128 bits exists, so 16 bytes give none.
    let after = u32::try_from(8 * len)
        .ok()
        .and_then(|bits| u128::MAX.checked_shl(bits))
        .unwrap_or(0);
    bytes | (after & u128::from_le_bytes([fill; 16]))
}

/// Writes the low `dst.len()` bytes, at most 16, of `value` into `dst`, with two overlapping stores
/// of 8, 4, or 1 bytes.
#[inline(always)]
fn write16(class: impl Class, dst: &mut [u8], value: u128) {
    let len = dst.len();
    assert!(len <= 16);
    let mut bytes = value.to_le_bytes();
    match len {
        8.. => {
            dst[..8].copy_from_slice(&bytes[..8]);
            dst[len - 8..].copy_from_slice(&bytes[len - 8..len]);
        }
        4.. => {
            dst[..4].copy_from_slice(&bytes[..4]);
            dst[len - 4..].copy_from_slice(&bytes[len - 4..len]);
        }
        1.. => {
            dst[0] = bytes[0];
            dst[len / 2] = bytes[len / 2];
            dst[len - 1] = bytes[len - 1];
        }
        0 => {}
    }
    class.clear(&mut bytes);
}

/// The loads and stores of x86_64. Each takes an array and an offset in it, and a constant
/// assertion proves that the access is in the array. A masked access takes a slice.
#[cfg(target_arch = "x86_64")]
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

    /// The mask of the first `len` lanes of 64.
    #[inline]
    fn lanes(len: usize) -> u64 {
        assert!(len <= 64);
        u64::MAX
            .checked_shr(64_u32.saturating_sub(len.try_into().unwrap_or(64)))
            .unwrap_or(0)
    }

    /// The bytes of `src`, at most 64, in the first lanes, and `fill` in the other lanes.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn load_masked(src: &[u8], fill: __m512i) -> __m512i {
        // SAFETY: the mask holds only the lanes of `src`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_loadu_epi8(fill, lanes(src.len()), src.as_ptr().cast()) }
    }

    /// Stores the first `dst.len()` lanes, at most 64, of `value` into `dst`.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(crate) fn store_masked(dst: &mut [u8], value: __m512i) {
        // SAFETY: the mask holds only the lanes of `dst`, and a lane outside the mask accesses no
        // memory.
        unsafe { _mm512_mask_storeu_epi8(dst.as_mut_ptr().cast(), lanes(dst.len()), value) }
    }

    /// The proof of the bounds of the masked loads and stores, for `cargo kani`.
    #[cfg(kani)]
    mod proofs {
        /// The mask of `len` lanes holds lanes `0..len` and no other lane, so a masked load or
        /// store of a slice accesses only the bytes of the slice.
        #[kani::proof]
        fn the_mask_holds_only_the_lanes_of_the_slice() {
            let len = kani::any_where(|&len: &usize| len <= 64);
            let mask = super::lanes(len);
            if len == 64 {
                assert!(mask == u64::MAX);
            } else {
                assert!(mask == (1 << len) - 1);
            }
        }
    }
}

/// The loads and stores of aarch64. Each takes an array and an offset in it, and a constant
/// assertion proves that the access is in the array.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "each load or store accesses only bytes of the reference that it takes"
)]
#[expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) mod aarch64 {
    use core::arch::aarch64::{uint8x16_t, vld1q_u8, vst1q_u8};
    #[cfg(feature = "hex")]
    use core::arch::aarch64::{uint8x16x2_t, vst2q_u8};
    #[cfg(feature = "base64")]
    use core::arch::aarch64::{uint8x16x3_t, uint8x16x4_t, vld1q_u8_x4, vld3q_u8, vld4q_u8, vst3q_u8, vst4q_u8};

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

    #[cfg(feature = "base64")]
    /// The 64 bytes at `AT` in `src` as four vectors, for a table lookup.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn table64<const AT: usize, const N: usize>(src: &[u8; N]) -> uint8x16x4_t {
        const { assert!(AT + 64 <= N) };
        // SAFETY: the bytes `AT..AT + 64` are in `src`, and the load needs no alignment.
        unsafe { vld1q_u8_x4(src.as_ptr().add(AT)) }
    }

    #[cfg(feature = "hex")]
    /// Stores the bytes of the two vectors in turn.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn store_pairs(dst: &mut [u8; 32], value: uint8x16x2_t) {
        // SAFETY: `dst` has 32 bytes, and the store needs no alignment.
        unsafe { vst2q_u8(dst.as_mut_ptr(), value) }
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

    #[cfg(feature = "base64")]
    /// Stores the bytes of the four vectors in turn.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn store_quads(dst: &mut [u8; 64], value: uint8x16x4_t) {
        // SAFETY: `dst` has 64 bytes, and the store needs no alignment.
        unsafe { vst4q_u8(dst.as_mut_ptr(), value) }
    }
}

/// The loads and stores of WebAssembly. Each takes an array and an offset in it, and a constant
/// assertion proves that the access is in the array.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
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
}
