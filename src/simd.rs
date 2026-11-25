//! The parts that the SIMD kernels of each codec share: the loops over the blocks, the tails, and
//! the loads and stores, which are the only unsafe operations of the kernels.
//!
//! A tail overlaps: after the whole blocks, the last block reads one whole vector that ends at the
//! end of the input, and writes some bytes again, with the same values. Only an input shorter than
//! one vector runs on arrays on the stack ([`short_blocks`]).

use crate::ct::Class;

/// The largest input of the short path of a codec on aarch64, in bytes: it runs inline in the
/// caller, and a longer input runs in a function of its own.
#[cfg(target_arch = "aarch64")]
pub(crate) const SHORT_MAX: usize = 64;

/// The lanes 0 to 15, the index of the identity shuffle.
#[cfg(any(feature = "base64", target_arch = "aarch64"))]
const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Runs `block` over an input shorter than one vector, on arrays on the stack. Each block reads `IN`
/// bytes of input and uses the first `step`. It writes `OUT` bytes of output, and the first
/// `out_len(step)` of them are the result.
///
/// The input after its end is `fill`, and only the result goes to `output`. Thus no SIMD load or
/// store passes the end of a slice, and no scalar code decodes a tail. `output` has the length
/// `out_len` of all of `input`. In the secret `class`, each array is zero before the function
/// returns. Gives the byte after the result in the output of the last block, for example the
/// trailing bits of base64.
///
/// The arrays take and give their bytes in 16-byte pieces, each with loads or stores of a fixed
/// size. Thus the compiler keeps an array of 16 bytes in registers: a copy of a variable length is
/// a call to `memcpy`, and a vector load of bytes that smaller stores wrote waits for them.
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(
    target_arch = "aarch64",
    expect(
        clippy::missing_safety_doc,
        reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
    )
)]
pub(crate) fn short_blocks<const IN: usize, const OUT: usize>(
    class: impl Class,
    input: &[u8],
    output: &mut [u8],
    step: usize,
    fill: u8,
    out_len: impl Fn(usize) -> usize,
    mut block: impl FnMut(&[u8; IN], &mut [u8; OUT]),
) -> u8 {
    const { assert!(IN.is_multiple_of(16) && OUT.is_multiple_of(16)) };
    let (mut read, mut written, mut after) = (0, 0, 0);
    while let Some(rest) = input.get(read..).filter(|rest| !rest.is_empty()) {
        let len = rest.len().min(step);
        let mut src = [fill; IN];
        for (piece, chunk) in src.as_chunks_mut::<16>().0.iter_mut().zip(rest[..len].chunks(16)) {
            *piece = read16(chunk, fill);
        }
        let mut dst = [0; OUT];
        block(&src, &mut dst);
        // A whole step gives a constant, and the last block gives the rest of `output`. Thus no
        // division of the length waits before the stores.
        let len_out = if len == step {
            out_len(step)
        } else {
            output.len() - written
        };
        for (chunk, piece) in output[written..written + len_out]
            .chunks_mut(16)
            .zip(dst.as_chunks::<16>().0)
        {
            write16(chunk, piece);
        }
        after = dst.get(len_out).copied().unwrap_or(0);
        class.clear(&mut src);
        class.clear(&mut dst);
        read += len;
        written += len_out;
    }
    after
}

/// Runs `block` over the whole blocks of `input`, at least one, and then over the last `IN` bytes,
/// which overlap the block before them, if `input` is not a multiple of `IN`. Each block writes
/// `OUT` bytes, and `output` has `OUT` bytes for each `IN` bytes of `input`. Thus the last block
/// writes again some bytes of the block before it, with the same values.
#[cfg(feature = "hex")]
#[inline(always)]
pub(crate) fn overlapping<const IN: usize, const OUT: usize>(
    input: &[u8],
    output: &mut [u8],
    mut block: impl FnMut(&[u8; IN], &mut [u8; OUT]),
) {
    let (chunks, rest) = input.as_chunks::<IN>();
    for (src, dst) in chunks.iter().zip(output.as_chunks_mut::<OUT>().0) {
        block(src, dst);
    }
    if !rest.is_empty()
        && let (Some(src), Some(dst)) = (input.last_chunk(), output.last_chunk_mut())
    {
        block(src, dst);
    }
}

/// Runs `block` over the first `IN` bytes of `input`, and then, if more bytes follow, over the last
/// `IN` bytes, which overlap the first block if `input` is shorter than `2 * IN` bytes. `input` has
/// `IN` to `2 * IN` bytes, and `output` has `OUT` bytes, a multiple of `IN`, for each `IN` bytes of
/// `input`. Thus the path
/// has no loop and no count of blocks. Slice patterns give the chunks with no `Option`, and thus
/// with no test of their address for null.
#[cfg(feature = "hex")]
#[inline(always)]
pub(crate) fn ends<const IN: usize, const OUT: usize>(
    input: &[u8],
    output: &mut [u8],
    mut block: impl FnMut(&[u8; IN], &mut [u8; OUT]),
) {
    if let ([first, ..], [dst, ..]) = (input.as_chunks().0, output.as_chunks_mut().0) {
        block(first, dst);
    }
    const { assert!(OUT.is_multiple_of(IN)) };
    // One subtraction gives the place of the last chunk of `input`, and one shift the place in
    // `output`: `as_rchunks` computes each from the count and the rest of the chunks.
    if let Some(at) = input.len().checked_sub(IN).filter(|&at| at > 0)
        && let ([last, ..], [dst, ..]) = (input[at..].as_chunks().0, output[at * (OUT / IN)..].as_chunks_mut().0)
    {
        block(last, dst);
    }
}

/// The bytes of `src`, at most 16, in the first bytes of the result, and `fill` in the other bytes.
/// Two overlapping loads of 8, 4, or 1 bytes read each byte, and the OR joins them.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn read16(src: &[u8], fill: u8) -> [u8; 16] {
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
    (bytes | (after & u128::from_le_bytes([fill; 16]))).to_le_bytes()
}

/// Writes the first `dst.len()` bytes, at most 16, of `value` into `dst`, with two overlapping
/// stores of 8, 4, or 1 bytes. Shifts of `value` give the bytes of each store, so the value stays in
/// registers: a load at a variable offset in a copy of it on the stack waits for the copy.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn write16(dst: &mut [u8], value: &[u8; 16]) {
    let len = dst.len();
    assert!(len <= 16);
    let value = u128::from_le_bytes(*value);
    // The bytes from `at`, which is less than 16, in the low bits.
    let from = |at: usize| value >> (8 * at);
    let word = |value: u128| u64::try_from(value & u128::from(u64::MAX)).unwrap_or_default();
    let half = |value: u128| u32::try_from(value & u128::from(u32::MAX)).unwrap_or_default();
    let byte = |value: u128| value.to_le_bytes()[0];
    match len {
        8.. => {
            dst[..8].copy_from_slice(&word(value).to_le_bytes());
            dst[len - 8..].copy_from_slice(&word(from(len - 8)).to_le_bytes());
        }
        4.. => {
            dst[..4].copy_from_slice(&half(value).to_le_bytes());
            dst[len - 4..].copy_from_slice(&half(from(len - 4)).to_le_bytes());
        }
        1.. => {
            dst[0] = byte(value);
            dst[len / 2] = byte(from(len / 2));
            dst[len - 1] = byte(from(len - 1));
        }
        0 => {}
    }
}

/// The bytes of `src`, at most 16, in the first bytes of the result, and `fill` in the other bytes.
/// The bits of the length give pieces of 8, 4, 2, and 1 bytes. From the last piece to the first,
/// a load fills each lane of a vector with the piece (`ld1r`), and `ext` puts the piece before the
/// bytes after it. Thus the bytes go from memory to vector registers, with no general register.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[target_feature(enable = "neon")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
fn read16(src: &[u8], fill: u8) -> [u8; 16] {
    use core::arch::aarch64::*;
    if let Ok(full) = <[u8; 16]>::try_from(src) {
        return full;
    }
    let len = src.len();
    assert!(len < 16);
    let piece = |bits: usize| src.get(len & !(2 * bits - 1)..).unwrap_or_default();
    let mut value = vdupq_n_u8(fill);
    if let (true, [byte, ..]) = (len & 1 != 0, piece(1)) {
        value = vextq_u8::<15>(vdupq_n_u8(*byte), value);
    }
    if let (true, Some(half)) = (len & 2 != 0, piece(2).first_chunk()) {
        value = vextq_u8::<14>(vreinterpretq_u8_u16(vdupq_n_u16(u16::from_le_bytes(*half))), value);
    }
    if let (true, Some(word)) = (len & 4 != 0, piece(4).first_chunk()) {
        value = vextq_u8::<12>(vreinterpretq_u8_u32(vdupq_n_u32(u32::from_le_bytes(*word))), value);
    }
    if let (true, Some(word)) = (len & 8 != 0, src.first_chunk()) {
        value = vextq_u8::<8>(vreinterpretq_u8_u64(vdupq_n_u64(u64::from_le_bytes(*word))), value);
    }
    let mut bytes = [0; 16];
    aarch64::store16::<0, _>(&mut bytes, value);
    bytes
}

/// Writes the first `dst.len()` bytes, at most 16, of `value` into `dst`, with two overlapping
/// stores of 8, 4, 2, or 1 bytes from the first lanes of vector registers (`str d`, `str s`,
/// `str h`, `str b`). A `tbl` with an index from the length moves the bytes of the second store to
/// the first lanes. Thus the bytes go from vector registers to memory, with no general register.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[target_feature(enable = "neon")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn write16(dst: &mut [u8], value: &[u8; 16]) {
    use core::arch::aarch64::*;
    if let Ok(full) = <&mut [u8; 16]>::try_from(&mut *dst) {
        *full = *value;
        return;
    }
    let len = dst.len();
    let value = aarch64::load16::<0, _>(value);
    // The bytes from `at`, which is less than 16, in the first lanes.
    let from = |at: usize| {
        let at = u8::try_from(at).unwrap_or_default();
        vqtbl1q_u8(value, vaddq_u8(aarch64::load16::<0, _>(&IOTA), vdupq_n_u8(at)))
    };
    match len {
        8.. => {
            if let Some(first) = dst.first_chunk_mut::<8>() {
                aarch64::store_first(first, value);
            }
            if let Some(last) = dst.last_chunk_mut::<8>() {
                aarch64::store_first(last, from(len - 8));
            }
        }
        4.. => {
            if let Some(first) = dst.first_chunk_mut::<4>() {
                aarch64::store_first(first, value);
            }
            if let Some(last) = dst.last_chunk_mut::<4>() {
                aarch64::store_first(last, from(len - 4));
            }
        }
        2.. => {
            if let Some(first) = dst.first_chunk_mut::<2>() {
                aarch64::store_first(first, value);
            }
            if let Some(last) = dst.last_chunk_mut::<2>() {
                aarch64::store_first(last, from(len - 2));
            }
        }
        1 => {
            if let Some(first) = dst.first_chunk_mut::<1>() {
                aarch64::store_first(first, value);
            }
        }
        0 => {}
    }
}

/// The number of the last bytes of a vector to keep, from a length of at most 16.
#[cfg(feature = "base64")]
#[inline(always)]
fn kept(len: usize) -> u8 {
    u8::try_from(len.min(16)).unwrap_or(16)
}

/// The last `len` bytes of `last`, at most 16, in the first lanes, and `fill` in the other lanes.
/// One shuffle with an index from the length moves them, so the bytes stay in vector registers.
#[cfg(all(feature = "base64", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "ssse3")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn from_end(last: &[u8; 16], len: usize, fill: u8) -> [u8; 16] {
    use core::arch::x86_64::*;
    // Lane `i` takes byte `16 - len + i`. `pshufb` gives zero to a lane whose index has bit 7, so
    // the `0x70` in the index sets bit 7 for each lane from `len`, and keeps the low four bits of
    // the other lanes.
    let index = _mm_add_epi8(
        x86::load16::<0, _>(&IOTA),
        _mm_set1_epi8((0x80 - kept(len)).cast_signed()),
    );
    let fill = _mm_set1_epi8(fill.cast_signed());
    let bytes = _mm_add_epi8(
        _mm_shuffle_epi8(_mm_sub_epi8(x86::load16::<0, _>(last), fill), index),
        fill,
    );
    let mut out = [0; 16];
    x86::store16::<0, _>(&mut out, bytes);
    out
}

/// The first `len` bytes of `first`, at most 16, in the last lanes, and zero in the other lanes:
/// the inverse of [`from_end`].
#[cfg(all(feature = "base64", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "ssse3")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn to_end(first: &[u8; 16], len: usize) -> [u8; 16] {
    use core::arch::x86_64::*;
    // Lane `i` takes byte `i + len - 16`, and a lane before `16 - len` has an index that wraps to
    // bit 7, so `pshufb` gives it zero.
    let index = _mm_add_epi8(
        x86::load16::<0, _>(&IOTA),
        _mm_set1_epi8(kept(len).wrapping_sub(16).cast_signed()),
    );
    let mut out = [0; 16];
    x86::store16::<0, _>(&mut out, _mm_shuffle_epi8(x86::load16::<0, _>(first), index));
    out
}

/// The last `len` bytes of `last`, at most 16, in the first lanes, and `fill` in the other lanes.
/// One `tbx` with an index from the length moves them, so the bytes stay in vector registers.
#[cfg(all(feature = "base64", target_arch = "aarch64"))]
#[inline(always)]
#[target_feature(enable = "neon")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn from_end(last: &[u8; 16], len: usize, fill: u8) -> [u8; 16] {
    use core::arch::aarch64::*;
    // Lane `i` takes byte `16 - len + i`, and a lane from `len` has an index past the table, so
    // `tbx` keeps its `fill`.
    let index = vaddq_u8(aarch64::load16::<0, _>(&IOTA), vdupq_n_u8(16 - kept(len)));
    let last = aarch64::load16::<0, _>(last);
    let bytes = if fill == 0 {
        vqtbl1q_u8(last, index)
    } else {
        vqtbx1q_u8(vdupq_n_u8(fill), last, index)
    };
    let mut out = [0; 16];
    aarch64::store16::<0, _>(&mut out, bytes);
    out
}

/// The first `len` bytes of `first`, at most 16, in the last lanes, and zero in the other lanes:
/// the inverse of [`from_end`].
#[cfg(all(feature = "base64", target_arch = "aarch64"))]
#[inline(always)]
#[target_feature(enable = "neon")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn to_end(first: &[u8; 16], len: usize) -> [u8; 16] {
    use core::arch::aarch64::*;
    // Lane `i` takes byte `i + len - 16`, and a lane before `16 - len` has an index that wraps past
    // the table, so `tbl` gives it zero.
    let index = vaddq_u8(aarch64::load16::<0, _>(&IOTA), vdupq_n_u8(kept(len).wrapping_sub(16)));
    let mut out = [0; 16];
    aarch64::store16::<0, _>(&mut out, vqtbl1q_u8(aarch64::load16::<0, _>(first), index));
    out
}

/// The last `len` bytes of `last`, at most 16, in the first lanes, and `fill` in the other lanes.
/// One `i8x16.swizzle` with an index from the length moves them, so the bytes stay in vector
/// registers.
#[cfg(all(feature = "base64", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(crate) fn from_end(last: &[u8; 16], len: usize, fill: u8) -> [u8; 16] {
    use core::arch::wasm32::*;
    // Lane `i` takes byte `16 - len + i`, and `swizzle` gives zero to a lane from `len`, whose index
    // is past the vector.
    let index = i8x16_add(wasm32::load16::<0, _>(&IOTA), u8x16_splat(16 - kept(len)));
    let fill = u8x16_splat(fill);
    let bytes = i8x16_add(
        i8x16_swizzle(i8x16_sub(wasm32::load16::<0, _>(last), fill), index),
        fill,
    );
    let mut out = [0; 16];
    wasm32::store16::<0, _>(&mut out, bytes);
    out
}

/// The first `len` bytes of `first`, at most 16, in the last lanes, and zero in the other lanes:
/// the inverse of [`from_end`].
#[cfg(all(feature = "base64", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(crate) fn to_end(first: &[u8; 16], len: usize) -> [u8; 16] {
    use core::arch::wasm32::*;
    // Lane `i` takes byte `i + len - 16`, and a lane before `16 - len` has an index that wraps past
    // the vector, so `swizzle` gives it zero.
    let index = i8x16_add(wasm32::load16::<0, _>(&IOTA), u8x16_splat(kept(len).wrapping_sub(16)));
    let mut out = [0; 16];
    wasm32::store16::<0, _>(&mut out, i8x16_swizzle(wasm32::load16::<0, _>(first), index));
    out
}

/// The byte at `at` of `value`, or zero from 16. A shuffle moves it to the first lane, so the bytes
/// stay in vector registers until the last move.
#[cfg(all(feature = "base64", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "ssse3")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn byte_at(value: &[u8; 16], at: usize) -> u8 {
    use core::arch::x86_64::*;
    // The `0x70` sets bit 7 of the index from 16, as in `from_end`.
    let index = _mm_set1_epi8((0x70 + kept(at)).cast_signed());
    _mm_cvtsi128_si32(_mm_shuffle_epi8(x86::load16::<0, _>(value), index)).to_le_bytes()[0]
}

/// The byte at `at` of `value`, or zero from 16. A `tbl` moves it to the first lane, so the bytes
/// stay in vector registers until the last move.
#[cfg(all(feature = "base64", target_arch = "aarch64"))]
#[inline(always)]
#[target_feature(enable = "neon")]
#[expect(
    clippy::missing_safety_doc,
    reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
)]
pub(crate) fn byte_at(value: &[u8; 16], at: usize) -> u8 {
    use core::arch::aarch64::*;
    vgetq_lane_u8::<0>(vqtbl1q_u8(aarch64::load16::<0, _>(value), vdupq_n_u8(kept(at))))
}

/// The byte at `at` of `value`, or zero from 16. A `swizzle` moves it to the first lane, so the
/// bytes stay in vector registers until the last move.
#[cfg(all(feature = "base64", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(crate) fn byte_at(value: &[u8; 16], at: usize) -> u8 {
    use core::arch::wasm32::*;
    u8x16_extract_lane::<0>(i8x16_swizzle(wasm32::load16::<0, _>(value), u8x16_splat(kept(at))))
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
    use core::arch::aarch64::{uint8x16_t, uint8x16x4_t, vld1q_u8, vld1q_u8_x4, vst1q_u8};
    #[cfg(feature = "base64")]
    use core::arch::aarch64::{uint8x16x3_t, vld3q_u8, vld4q_u8, vst3q_u8};

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
    /// on the DIT list of Arm (C103).
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

    /// The 64 bytes at `AT` in `src` as four vectors, for a table lookup.
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
