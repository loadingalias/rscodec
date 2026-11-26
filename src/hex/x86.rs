//! The x86_64 kernels: SSSE3, AVX2, and AVX-512BW, and the short decode path on SSE2. In the
//! kernels, `pshufb` maps each nibble to its digit. The decoders map each character to its nibble
//! with saturating arithmetic, which gives 16 or more for a byte that is not a digit, and OR the
//! nibbles of each block into one vector, which they test once after the last block. The
//! AVX-512BW decoder compares into masks instead.
//!
//! An encode of up to 64 bytes runs one block, or two, at each end of the slices ([`ends`]). A
//! longer input runs whole blocks, and a last block that ends at the end of the slices and writes
//! some bytes again ([`overlapping`]). The AVX2 kernel gives an input shorter than one block to the
//! SSSE3 kernel. An input shorter than one vector runs on arrays on the stack, or on AVX-512BW in
//! one vector with masked loads and stores. Thus a call has no count of a tail, and no loop for an
//! input of up to two blocks.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::x86_64::*;

use super::encode_steps;
use crate::{
    ct::Class,
    simd::{ends, overlapping, short_blocks},
    vector::x86::{load_masked, load16, load32, load64, store_masked, store16, store32, store64},
};

/// The digit of each nibble.
const DIGITS: [u8; 16] = *b"0123456789abcdef";

/// The multipliers of `pmaddubsw` that join the high and the low nibble of each pair: `hi * 16 + lo`.
const PAIR: i16 = 0x0110;

/// The step from `c - '0'` to the nibble of a letter `a`–`f`.
const LETTER_STEP: i8 = 39;

/// The first 8 bytes of `value`, as 8 bytes of the output.
#[inline(always)]
#[target_feature(enable = "sse2")]
fn store_low(dst: &mut [u8; 8], value: __m128i) {
    let mut bytes = [0; 16];
    store16::<0, _>(&mut bytes, value);
    if let Some(low) = bytes.first_chunk() {
        *dst = *low;
    }
}

/// The short decode path: SSE2, which each x86_64 CPU has, for a short result. It runs inline in
/// the caller, with no choice of kernel. SSE2 has no `pmaddubsw`, so shifts and `packuswb` join the
/// nibbles.
pub(super) mod sse2 {
    use super::*;

    /// The nibble of each of 16 characters: 0 to 15 for a digit, and 16 or more for another byte.
    /// `c - 58` with wrapping, minus 6 with saturation, plus 16 gives the value of `0`–`9`, and 16
    /// or more for each other byte. `c - 'a'` with wrapping, plus 10 with saturation, gives the
    /// value of `a`–`f`, and 16 or more for each other byte. The unsigned minimum takes the valid
    /// one.
    #[inline(always)]
    #[target_feature(enable = "sse2")]
    pub(in super::super) fn nibbles(chars: __m128i) -> __m128i {
        let digit = _mm_sub_epi8(
            _mm_subs_epu8(
                _mm_add_epi8(chars, _mm_set1_epi8(0xc6_u8.cast_signed())),
                _mm_set1_epi8(6),
            ),
            _mm_set1_epi8(0xf0_u8.cast_signed()),
        );
        let letter = _mm_adds_epu8(
            _mm_sub_epi8(chars, _mm_set1_epi8(b'a'.cast_signed())),
            _mm_set1_epi8(10),
        );
        _mm_min_epu8(digit, letter)
    }

    /// Whether each lane of the OR of the nibbles is at most 15: the sign bit of each lane plus
    /// `0x70` with saturation.
    #[inline(always)]
    #[target_feature(enable = "sse2")]
    pub(in super::super) fn digits_only(nibbles: __m128i) -> bool {
        _mm_movemask_epi8(_mm_adds_epu8(nibbles, _mm_set1_epi8(0x70))) == 0
    }

    /// The bytes of the nibbles of 32 characters: the first, the high nibble, of each pair is in the
    /// low byte of its 16-bit lane. The even and the odd lanes go to two vectors of 16 nibbles, and a
    /// shift of 16-bit lanes moves each high nibble, which is below 16, within its byte.
    #[inline(always)]
    #[target_feature(enable = "sse2")]
    fn join(first: __m128i, second: __m128i) -> __m128i {
        let low = _mm_set1_epi16(0x00ff);
        let hi = _mm_packus_epi16(_mm_and_si128(first, low), _mm_and_si128(second, low));
        let lo = _mm_packus_epi16(_mm_srli_epi16(first, 8), _mm_srli_epi16(second, 8));
        _mm_or_si128(_mm_slli_epi16(hi, 4), lo)
    }

    #[inline(always)]
    #[target_feature(enable = "sse2")]
    pub(in super::super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
        let mut values = _mm_setzero_si128();
        if text.len() >= 32 {
            overlapping::<32, 16>(text, out, |src, dst| {
                let (first, second) = (nibbles(load16::<0, _>(src)), nibbles(load16::<16, _>(src)));
                values = _mm_or_si128(values, _mm_or_si128(first, second));
                store16::<0, _>(dst, join(first, second));
            });
        } else if text.len() >= 16 {
            overlapping::<16, 8>(text, out, |src, dst| {
                let chars = nibbles(load16::<0, _>(src));
                values = _mm_or_si128(values, chars);
                store_low(dst, join(chars, chars));
            });
        } else {
            short_blocks::<32, 16>(
                class,
                text,
                out,
                32,
                b'0',
                |len| len / 2,
                |src, dst| {
                    let (first, second) = (nibbles(load16::<0, _>(src)), nibbles(load16::<16, _>(src)));
                    values = _mm_or_si128(values, _mm_or_si128(first, second));
                    store16::<0, _>(dst, join(first, second));
                },
            );
        }
        digits_only(values)
    }
}

pub(super) mod ssse3 {
    use super::{
        sse2::{digits_only, nibbles},
        *,
    };

    /// The bytes of the nibbles of 32 characters: `pmaddubsw` joins each pair.
    #[inline(always)]
    #[target_feature(enable = "ssse3")]
    fn join(first: __m128i, second: __m128i) -> __m128i {
        let pair = _mm_set1_epi16(PAIR);
        _mm_packus_epi16(_mm_maddubs_epi16(first, pair), _mm_maddubs_epi16(second, pair))
    }

    // The AVX2 kernel takes its short inputs, inline.
    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        let digits = load16::<0, _>(&DIGITS);
        let nibble = _mm_set1_epi8(0x0f);
        let block = |src: &[u8; 16], dst: &mut [u8; 32]| {
            let src = load16::<0, _>(src);
            let hi = _mm_shuffle_epi8(digits, _mm_and_si128(_mm_srli_epi16(src, 4), nibble));
            let lo = _mm_shuffle_epi8(digits, _mm_and_si128(src, nibble));
            store16::<0, _>(dst, _mm_unpacklo_epi8(hi, lo));
            store16::<16, _>(dst, _mm_unpackhi_epi8(hi, lo));
        };
        encode_steps(class, bytes, out, block, |bytes, out| overlapping(bytes, out, block));
    }

    // The AVX2 kernel takes its short inputs, inline.
    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
        let mut values = _mm_setzero_si128();
        if text.len() >= 32 {
            overlapping::<32, 16>(text, out, |src, dst| {
                let (first, second) = (nibbles(load16::<0, _>(src)), nibbles(load16::<16, _>(src)));
                values = _mm_or_si128(values, _mm_or_si128(first, second));
                store16::<0, _>(dst, join(first, second));
            });
        } else if text.len() >= 16 {
            overlapping::<16, 8>(text, out, |src, dst| {
                let chars = nibbles(load16::<0, _>(src));
                values = _mm_or_si128(values, chars);
                store_low(dst, join(chars, chars));
            });
        } else {
            short_blocks::<32, 16>(
                class,
                text,
                out,
                32,
                b'0',
                |len| len / 2,
                |src, dst| {
                    let (first, second) = (nibbles(load16::<0, _>(src)), nibbles(load16::<16, _>(src)));
                    values = _mm_or_si128(values, _mm_or_si128(first, second));
                    store16::<0, _>(dst, join(first, second));
                },
            );
        }
        digits_only(values)
    }
}

pub(super) mod avx2 {
    use super::*;

    /// The nibble of each of 32 characters, as [`super::sse2::nibbles`].
    #[inline]
    #[target_feature(enable = "avx2")]
    fn nibbles(chars: __m256i) -> __m256i {
        let digit = _mm256_sub_epi8(
            _mm256_subs_epu8(
                _mm256_add_epi8(chars, _mm256_set1_epi8(0xc6_u8.cast_signed())),
                _mm256_set1_epi8(6),
            ),
            _mm256_set1_epi8(0xf0_u8.cast_signed()),
        );
        let letter = _mm256_adds_epu8(
            _mm256_sub_epi8(chars, _mm256_set1_epi8(b'a'.cast_signed())),
            _mm256_set1_epi8(10),
        );
        _mm256_min_epu8(digit, letter)
    }

    /// Decodes 32 to 63 characters with one 32-byte vector at each end, and gives whether each is a
    /// digit. `vptest` gives the validity in the flags: the masked vector of the AVX-512BW kernel
    /// moved its mask to a general register, and took 1.32 times as long as faster-hex at 32
    /// characters (C109).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn decode_32(text: &[u8], out: &mut [u8]) -> bool {
        let pair = _mm256_set1_epi16(PAIR);
        let mut values = _mm256_setzero_si256();
        overlapping::<32, 16>(text, out, |src, dst| {
            let nibbles = nibbles(load32::<0, _>(src));
            values = _mm256_or_si256(values, nibbles);
            let pairs = _mm256_maddubs_epi16(nibbles, pair);
            let bytes = _mm_packus_epi16(_mm256_castsi256_si128(pairs), _mm256_extracti128_si256::<1>(pairs));
            store16::<0, _>(dst, bytes);
        });
        // Each lane of the OR is at most 15, so it has no bit of `0xf0`.
        _mm256_testz_si256(values, _mm256_set1_epi8(0xf0_u8.cast_signed())) == 1
    }

    /// The 64-bit words 0, 2, 1, and 3, which move the 8-byte halves of the 128-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn swap_middle(value: __m256i) -> __m256i {
        _mm256_permutevar8x32_epi32(value, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7))
    }

    // Each entry that a wider kernel takes runs inline in it.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(in super::super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        if bytes.len() < 32 {
            return super::ssse3::encode(class, bytes, out);
        }
        let digits = _mm256_broadcastsi128_si256(load16::<0, _>(&DIGITS));
        let nibble = _mm256_set1_epi8(0x0f);
        let block = |src: &[u8; 32], dst: &mut [u8; 64]| {
            // Each 128-bit lane gets the bytes of one half of the output.
            let src = swap_middle(load32::<0, _>(src));
            let hi = _mm256_shuffle_epi8(digits, _mm256_and_si256(_mm256_srli_epi16(src, 4), nibble));
            let lo = _mm256_shuffle_epi8(digits, _mm256_and_si256(src, nibble));
            store32::<0, _>(dst, _mm256_unpacklo_epi8(hi, lo));
            store32::<32, _>(dst, _mm256_unpackhi_epi8(hi, lo));
        };
        // Up to 64 bytes, one block at each end, with no loop.
        if bytes.len() <= 64 {
            ends(bytes, out, block);
        } else {
            overlapping(bytes, out, block);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(in super::super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
        match text.len() {
            0..32 => return super::ssse3::decode(class, text, out),
            32..64 => return decode_32(text, out),
            _ => {}
        }
        let mut values = _mm256_setzero_si256();
        let pair = _mm256_set1_epi16(PAIR);
        overlapping::<64, 32>(text, out, |src, dst| {
            let (first, second) = (nibbles(load32::<0, _>(src)), nibbles(load32::<32, _>(src)));
            values = _mm256_or_si256(values, _mm256_or_si256(first, second));
            let bytes = _mm256_packus_epi16(_mm256_maddubs_epi16(first, pair), _mm256_maddubs_epi16(second, pair));
            store32::<0, _>(dst, swap_middle(bytes));
        });
        // Each lane of the OR is at most 15, so it has no bit of `0xf0`.
        _mm256_testz_si256(values, _mm256_set1_epi8(0xf0_u8.cast_signed())) == 1
    }
}

/// The AVX-512BW decoder runs one 64-byte vector with masked loads and stores below 16 bytes of
/// result, the 32-byte vectors of the AVX2 kernel from 16 to 31 bytes, half blocks of one 64-byte
/// vector of text with no mask up to 64 bytes, and blocks of two vectors of text from 64 bytes.
/// The encoder runs one masked vector below 32 bytes, one vector at each end up to 64 bytes, blocks
/// of 64 bytes from 65 bytes, and 32-byte vectors from `WIDE_MIN`.
pub(super) mod avx512bw {
    use super::*;

    /// The shortest input that the encoder runs in 32-byte vectors. The loop of a longer input waits
    /// for the L2 cache, and 64-byte vectors lower the clock: at 65536 bytes, 32-byte vectors took
    /// 0.98 times as long as 64-byte vectors (0.97 native). At 16384 bytes, 64-byte vectors took
    /// 0.84 times as long as 32-byte vectors (0.86 native), so the switch is at 64 KiB (C109 probe 3).
    const WIDE_MIN: usize = 64 * 1024;

    /// The nibbles of 64 characters, and the mask of the lanes that are digits.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn nibbles(chars: __m512i) -> (__m512i, u64) {
        let digit = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(chars, _mm512_set1_epi8(b'0'.cast_signed())),
            _mm512_set1_epi8(10),
        );
        let letter = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(chars, _mm512_set1_epi8(b'a'.cast_signed())),
            _mm512_set1_epi8(6),
        );
        let value = _mm512_sub_epi8(chars, _mm512_set1_epi8(b'0'.cast_signed()));
        (
            _mm512_mask_sub_epi8(value, letter, value, _mm512_set1_epi8(LETTER_STEP)),
            _kor_mask64(digit, letter),
        )
    }

    /// The digit of each nibble, in each 128-bit lane.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn digits() -> __m512i {
        _mm512_broadcast_i32x4(load16::<0, _>(&DIGITS))
    }

    /// The 64 characters of 32 bytes: each byte in a 16-bit lane, its high nibble in the low byte,
    /// so that the digits come in the order of the text.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn encode_half(digits: __m512i, bytes: __m256i) -> __m512i {
        let src = _mm512_cvtepu8_epi16(bytes);
        let nibbles = _mm512_or_si512(
            _mm512_srli_epi16(src, 4),
            _mm512_slli_epi16(_mm512_and_si512(src, _mm512_set1_epi16(0x0f)), 8),
        );
        _mm512_shuffle_epi8(digits, nibbles)
    }

    /// The 32 bytes of 64 characters, and the mask of the lanes that are digits. `pmaddubsw` joins
    /// each pair in a 16-bit lane, and `vpmovwb` narrows the lanes to the bytes.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn decode_half(chars: __m512i) -> (__m256i, u64) {
        let (nibbles, valid) = nibbles(chars);
        (
            _mm512_cvtepi16_epi8(_mm512_maddubs_epi16(nibbles, _mm512_set1_epi16(PAIR))),
            valid,
        )
    }

    // A build with the feature fixes the kernel, and then the kernel runs inline in the caller.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        let digits = digits();
        // Below 32 bytes, one masked vector. From 32 to 64 bytes, one vector at each end, with no
        // loop. A path with 32-byte vectors took 1.04 to 1.14 times as long from 32 to 64 bytes,
        // although it kept the clock of the CPU higher (C109). The function stays small, so that a
        // build with the feature runs it inline.
        if bytes.len() < 32 {
            let src = _mm512_castsi512_si256(load_masked(bytes, _mm512_setzero_si512()));
            return store_masked(out, encode_half(digits, src));
        }
        if bytes.len() <= 64 {
            return ends::<32, 64>(bytes, out, |src, dst| {
                store64::<0, _>(dst, encode_half(digits, load32::<0, _>(src)));
            });
        }
        if bytes.len() >= WIDE_MIN {
            return encode_wide(class, bytes, out);
        }
        let nibble = _mm512_set1_epi8(0x0f);
        // Each 128-bit lane gets 8 bytes of the first half and 8 bytes of the second half.
        let spread = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        overlapping::<64, 128>(bytes, out, |src, dst| {
            let src = _mm512_permutexvar_epi32(spread, load64::<0, _>(src));
            let hi = _mm512_shuffle_epi8(digits, _mm512_and_si512(_mm512_srli_epi16(src, 4), nibble));
            let lo = _mm512_shuffle_epi8(digits, _mm512_and_si512(src, nibble));
            store64::<0, _>(dst, _mm512_unpacklo_epi8(hi, lo));
            store64::<64, _>(dst, _mm512_unpackhi_epi8(hi, lo));
        });
    }

    /// Encodes an input of at least `WIDE_MIN` bytes on the AVX2 code, out of line: the call costs
    /// little beside its work, and it keeps [`encode`] small enough to run inline in a build with
    /// the feature.
    #[inline(never)]
    #[target_feature(enable = "avx512bw")]
    fn encode_wide(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        super::avx2::encode(class, bytes, out);
    }

    /// Decodes one vector of text and gives whether each character is a digit.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode32(src: &[u8; 64], dst: &mut [u8; 32]) -> bool {
        let (bytes, valid) = decode_half(load64::<0, _>(src));
        store32::<0, _>(dst, bytes);
        valid == u64::MAX
    }

    /// Decodes two vectors of text and gives whether each character is a digit.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode64(src: &[u8; 128], dst: &mut [u8; 64]) -> bool {
        decode_pair(src, dst) == u64::MAX
    }

    /// Decodes two vectors of text and gives the mask of lanes valid in both.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn decode_pair(src: &[u8; 128], dst: &mut [u8; 64]) -> u64 {
        let pair = _mm512_set1_epi16(PAIR);
        let order = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        let (first, first_valid) = nibbles(load64::<0, _>(src));
        let (second, second_valid) = nibbles(load64::<64, _>(src));
        let bytes = _mm512_packus_epi16(_mm512_maddubs_epi16(first, pair), _mm512_maddubs_epi16(second, pair));
        store64::<0, _>(dst, _mm512_permutexvar_epi32(order, bytes));
        _kand_mask64(first_valid, second_valid)
    }

    /// Decodes a result shorter than one half block or at least two whole blocks.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode(text: &[u8], out: &mut [u8]) -> bool {
        if (32..64).contains(&text.len()) {
            return super::avx2::decode_32(text, out);
        }
        if text.len() < 64 {
            // '0', which is zero, is in place of the characters after the end.
            let (bytes, valid) = decode_half(load_masked(text, _mm512_set1_epi8(b'0'.cast_signed())));
            store_masked(out, _mm512_castsi256_si512(bytes));
            return valid == u64::MAX;
        }
        let mut valid = u64::MAX;
        overlapping::<128, 64>(text, out, |src, dst| {
            valid = _kand_mask64(valid, decode_pair(src, dst));
        });
        valid == u64::MAX
    }
}
