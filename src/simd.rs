//! The parts that the SIMD kernels of the base-N codecs share: the loops over the blocks and the
//! tails. The loads and stores are in [`crate::vector`].
//!
//! A tail overlaps: after the whole blocks, the last block reads one whole vector that ends at the
//! end of the input, and writes some bytes again, with the same values. Only an input shorter than
//! one vector runs on arrays on the stack ([`short_blocks`]).

use crate::ct::Class;
#[cfg(target_arch = "aarch64")]
use crate::vector::aarch64;
#[cfg(all(feature = "base64", target_arch = "wasm32", target_feature = "simd128"))]
use crate::vector::wasm32;
#[cfg(all(feature = "base64", target_arch = "x86_64"))]
use crate::vector::x86;

/// Input threshold in bytes for AArch64 short kernels.
#[cfg(target_arch = "aarch64")]
pub(crate) const SHORT_MAX: usize = 64;

/// The lanes 0 to 15, the index of the identity shuffle.
#[cfg(any(feature = "base64", target_arch = "aarch64"))]
const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Stage up to `step` input bytes in an `IN`-byte array padded with `fill`.
/// Each call writes an `OUT`-byte block. Copy only the encoded or decoded result to `output`.
/// Return the byte after the final result for the base64 trailing-bit check.
/// Fixed-width transfers avoid variable-size copies. Clear both staging arrays for secret data.
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
