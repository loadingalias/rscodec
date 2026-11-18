//! The x86_64 kernels: SSSE3, AVX2, and AVX-512BW. `pshufb` maps each nibble to its digit, and
//! compares classify each character. Each decoder ANDs the valid lanes of each block into one
//! vector or mask, and tests it once after the last block.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::x86_64::*;

use crate::{
    ct::Class,
    simd::{
        blocks, whole_blocks,
        x86::{load_masked, load16, load32, load64, store_masked, store16, store32, store64},
    },
};

/// The digit of each nibble.
const DIGITS: [u8; 16] = *b"0123456789abcdef";

/// The multipliers of `pmaddubsw` that join the high and the low nibble of each pair: `hi * 16 + lo`.
const PAIR: i16 = 0x0110;

/// The step from `c - '0'` to the nibble of a letter `a`–`f`.
const LETTER_STEP: i8 = 39;

pub(super) mod ssse3 {
    use super::*;

    /// The nibbles of 16 characters, and the lanes that are digits, which are all ones.
    #[inline]
    #[target_feature(enable = "ssse3")]
    fn nibbles(chars: __m128i) -> (__m128i, __m128i) {
        // `c - '0' - 128` is below -118 only for `0`–`9`, and `c - 'a' - 128` is below -122 only for
        // `a`–`f`. Thus signed compares give the ranges.
        let digit = _mm_cmpgt_epi8(
            _mm_set1_epi8(-118),
            _mm_sub_epi8(chars, _mm_set1_epi8(0xb0_u8.cast_signed())),
        );
        let letter = _mm_cmpgt_epi8(
            _mm_set1_epi8(-122),
            _mm_sub_epi8(chars, _mm_set1_epi8(0xe1_u8.cast_signed())),
        );
        let value = _mm_sub_epi8(
            _mm_sub_epi8(chars, _mm_set1_epi8(b'0'.cast_signed())),
            _mm_and_si128(letter, _mm_set1_epi8(LETTER_STEP)),
        );
        (value, _mm_or_si128(digit, letter))
    }

    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        let digits = load16::<0, _>(&DIGITS);
        let nibble = _mm_set1_epi8(0x0f);
        blocks::<16, 32>(
            class,
            bytes,
            out,
            16,
            0,
            |len| len * 2,
            |src, dst| {
                let src = load16::<0, _>(src);
                let hi = _mm_shuffle_epi8(digits, _mm_and_si128(_mm_srli_epi16(src, 4), nibble));
                let lo = _mm_shuffle_epi8(digits, _mm_and_si128(src, nibble));
                store16::<0, _>(dst, _mm_unpacklo_epi8(hi, lo));
                store16::<16, _>(dst, _mm_unpackhi_epi8(hi, lo));
            },
        );
    }

    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
        let mut valid = _mm_set1_epi8(-1);
        blocks::<32, 16>(
            class,
            text,
            out,
            32,
            b'0',
            |len| len / 2,
            |src, dst| {
                let (first, first_valid) = nibbles(load16::<0, _>(src));
                let (second, second_valid) = nibbles(load16::<16, _>(src));
                valid = _mm_and_si128(valid, _mm_and_si128(first_valid, second_valid));
                let pair = _mm_set1_epi16(PAIR);
                store16::<0, _>(
                    dst,
                    _mm_packus_epi16(_mm_maddubs_epi16(first, pair), _mm_maddubs_epi16(second, pair)),
                );
            },
        );
        _mm_movemask_epi8(valid) == 0xffff
    }
}

pub(super) mod avx2 {
    use super::*;

    /// The nibbles of 32 characters, and the lanes that are digits. As [`super::ssse3`].
    #[inline]
    #[target_feature(enable = "avx2")]
    fn nibbles(chars: __m256i) -> (__m256i, __m256i) {
        let digit = _mm256_cmpgt_epi8(
            _mm256_set1_epi8(-118),
            _mm256_sub_epi8(chars, _mm256_set1_epi8(0xb0_u8.cast_signed())),
        );
        let letter = _mm256_cmpgt_epi8(
            _mm256_set1_epi8(-122),
            _mm256_sub_epi8(chars, _mm256_set1_epi8(0xe1_u8.cast_signed())),
        );
        let value = _mm256_sub_epi8(
            _mm256_sub_epi8(chars, _mm256_set1_epi8(b'0'.cast_signed())),
            _mm256_and_si256(letter, _mm256_set1_epi8(LETTER_STEP)),
        );
        (value, _mm256_or_si256(digit, letter))
    }

    /// The 64-bit words 0, 2, 1, and 3, which move the 8-byte halves of the 128-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn swap_middle(value: __m256i) -> __m256i {
        _mm256_permutevar8x32_epi32(value, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7))
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
        let digits = _mm256_broadcastsi128_si256(load16::<0, _>(&DIGITS));
        let nibble = _mm256_set1_epi8(0x0f);
        let (read, written) = whole_blocks::<32, 64>(
            bytes,
            out,
            32,
            |len| len * 2,
            |src, dst| {
                // Each 128-bit lane gets the bytes of one half of the output.
                let src = swap_middle(load32::<0, _>(src));
                let hi = _mm256_shuffle_epi8(digits, _mm256_and_si256(_mm256_srli_epi16(src, 4), nibble));
                let lo = _mm256_shuffle_epi8(digits, _mm256_and_si256(src, nibble));
                store32::<0, _>(dst, _mm256_unpacklo_epi8(hi, lo));
                store32::<32, _>(dst, _mm256_unpackhi_epi8(hi, lo));
            },
        );
        // The rest, in vectors of 16 bytes.
        super::ssse3::encode(class, &bytes[read..], &mut out[written..]);
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
        let mut valid = _mm256_set1_epi8(-1);
        let (read, written) = whole_blocks::<64, 32>(
            text,
            out,
            64,
            |len| len / 2,
            |src, dst| {
                let (first, first_valid) = nibbles(load32::<0, _>(src));
                let (second, second_valid) = nibbles(load32::<32, _>(src));
                valid = _mm256_and_si256(valid, _mm256_and_si256(first_valid, second_valid));
                let pair = _mm256_set1_epi16(PAIR);
                let bytes = _mm256_packus_epi16(_mm256_maddubs_epi16(first, pair), _mm256_maddubs_epi16(second, pair));
                store32::<0, _>(dst, swap_middle(bytes));
            },
        );
        // The rest, in vectors of 16 bytes.
        let rest_valid = super::ssse3::decode(class, &text[read..], &mut out[written..]);
        (_mm256_movemask_epi8(valid) == -1) & rest_valid
    }
}

/// The AVX-512BW kernels run whole blocks of 64 bytes, then one block with masked loads and stores
/// for the rest.
pub(super) mod avx512bw {
    use super::*;

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
            digit | letter,
        )
    }

    /// The bytes of 64 bytes for the two halves of the output, in the order of `unpack`.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn encode_block(digits: __m512i, src: __m512i) -> (__m512i, __m512i) {
        // Each 128-bit lane gets 8 bytes of the first half and 8 bytes of the second half.
        let spread = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        let src = _mm512_permutexvar_epi32(spread, src);
        let nibble = _mm512_set1_epi8(0x0f);
        let hi = _mm512_shuffle_epi8(digits, _mm512_and_si512(_mm512_srli_epi16(src, 4), nibble));
        let lo = _mm512_shuffle_epi8(digits, _mm512_and_si512(src, nibble));
        (_mm512_unpacklo_epi8(hi, lo), _mm512_unpackhi_epi8(hi, lo))
    }

    /// The 64 bytes of 128 characters, and the mask of the valid lanes of each half.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn decode_block(first: __m512i, second: __m512i) -> (__m512i, u64) {
        let (first, first_valid) = nibbles(first);
        let (second, second_valid) = nibbles(second);
        let pair = _mm512_set1_epi16(PAIR);
        let bytes = _mm512_packus_epi16(_mm512_maddubs_epi16(first, pair), _mm512_maddubs_epi16(second, pair));
        let order = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        (_mm512_permutexvar_epi32(order, bytes), first_valid & second_valid)
    }

    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn encode(bytes: &[u8], out: &mut [u8]) {
        let digits = _mm512_broadcast_i32x4(load16::<0, _>(&DIGITS));
        let (chunks, rest) = bytes.as_chunks::<64>();
        let (texts, tail) = out.split_at_mut(chunks.len() * 128);
        for (chunk, text) in chunks.iter().zip(texts.as_chunks_mut::<128>().0) {
            let (first, second) = encode_block(digits, load64::<0, _>(chunk));
            store64::<0, _>(text, first);
            store64::<64, _>(text, second);
        }
        if !rest.is_empty() {
            let (first, second) = encode_block(digits, load_masked(rest, _mm512_setzero_si512()));
            let (tail_first, tail_second) = tail.split_at_mut(tail.len().min(64));
            store_masked(tail_first, first);
            store_masked(tail_second, second);
        }
    }

    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode(text: &[u8], out: &mut [u8]) -> bool {
        let mut valid = u64::MAX;
        let (chunks, rest) = text.as_chunks::<128>();
        let (bytes, tail) = out.split_at_mut(chunks.len() * 64);
        for (chunk, bytes) in chunks.iter().zip(bytes.as_chunks_mut::<64>().0) {
            let (block, block_valid) = decode_block(load64::<0, _>(chunk), load64::<64, _>(chunk));
            valid &= block_valid;
            store64::<0, _>(bytes, block);
        }
        if !rest.is_empty() {
            // '0', which is zero, in place of the characters after the end.
            let zeros = _mm512_set1_epi8(b'0'.cast_signed());
            let (first, second) = rest.split_at(rest.len().min(64));
            let (block, block_valid) = decode_block(load_masked(first, zeros), load_masked(second, zeros));
            valid &= block_valid;
            store_masked(tail, block);
        }
        valid == u64::MAX
    }
}
