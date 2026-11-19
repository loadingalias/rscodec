//! The x86_64 kernels: SSSE3, AVX2, AVX-512BW, and AVX-512 VBMI.
//!
//! The SSSE3, AVX2, and AVX-512BW kernels use the method of Muła and Lemire (2018): two nibble
//! lookups with `pshufb` classify each character, and `pmaddubsw` and `pmaddwd` pack the bits. The
//! VBMI kernels use the method of Muła and Lemire (2020), with `vpermb`, `vpermi2b`, and
//! `vpmultishiftqb`. Each decoder ORs the classes of each block into one vector, and tests it once
//! after the last block.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::x86_64::*;

use super::{Profile, Tables, bytes_len, symbols_len};
use crate::{
    ct::Class,
    simd::{
        blocks, whole_blocks,
        x86::{load_masked, load16, load32, load64, store_masked, store16, store32, store64},
    },
};

/// The bytes `b1 b0 b2 b1` of each three input bytes, in each 32-bit lane, for the encoder.
const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

/// The three bytes of each 32-bit lane after the decoder packs it, first byte first. The last four
/// lanes are zero.
const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80];

/// The multipliers that pack four sextets into 24 bits: `pmaddubsw` gives `a << 6 | b` and
/// `c << 6 | d`, and `pmaddwd` joins them.
const PAIRS: i32 = 0x0140_0140;
const WORDS: i32 = 0x0001_1000;

/// The masks and the multipliers that move the four sextets of `b1 b0 b2 b1` into the four bytes
/// of the lane (Muła and Lemire, 2018).
const HIGH_SEXTETS: i32 = 0x0fc0_fc00;
const HIGH_SHIFTS: i32 = 0x0400_0040;
const LOW_SEXTETS: i32 = 0x003f_03f0;
const LOW_SHIFTS: i32 = 0x0100_0010;

pub(super) mod ssse3 {
    use core::arch::x86_64::*;

    use super::{
        Class, DECODE_ORDER, ENCODE_ORDER, HIGH_SEXTETS, HIGH_SHIFTS, LOW_SEXTETS, LOW_SHIFTS, PAIRS, Profile, Tables,
        WORDS, blocks, bytes_len, load16, store16, symbols_len,
    };

    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
        let tables = &Tables::of(profile).nibbles;
        let shift = load16::<0, _>(&tables.shift);
        let order = load16::<0, _>(&ENCODE_ORDER);
        blocks::<16, 16>(class, bytes, out, 12, 0, symbols_len, |src, dst| {
            let input = _mm_shuffle_epi8(load16::<0, _>(src), order);
            let high = _mm_mulhi_epu16(
                _mm_and_si128(input, _mm_set1_epi32(HIGH_SEXTETS)),
                _mm_set1_epi32(HIGH_SHIFTS),
            );
            let low = _mm_mullo_epi16(
                _mm_and_si128(input, _mm_set1_epi32(LOW_SEXTETS)),
                _mm_set1_epi32(LOW_SHIFTS),
            );
            let sextets = _mm_or_si128(high, low);
            // The class: 0 for a-z, 1 to 10 for 0-9, 11 and 12 for the last two, and 13 for A-Z.
            let class = _mm_or_si128(
                _mm_subs_epu8(sextets, _mm_set1_epi8(51)),
                _mm_and_si128(_mm_cmpgt_epi8(_mm_set1_epi8(26), sextets), _mm_set1_epi8(13)),
            );
            store16::<0, _>(dst, _mm_add_epi8(sextets, _mm_shuffle_epi8(shift, class)));
        });
    }

    #[target_feature(enable = "ssse3")]
    pub(in super::super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let tables = &Tables::of(profile).nibbles;
        let (lo_classes, hi_classes) = (load16::<0, _>(&tables.lo), load16::<0, _>(&tables.hi));
        let roll = load16::<0, _>(&tables.roll);
        let special = _mm_set1_epi8(tables.special.cast_signed());
        let special_index = _mm_set1_epi8(tables.special_index.cast_signed());
        let order = load16::<0, _>(&DECODE_ORDER);
        let nibble = _mm_set1_epi8(0x0f);
        let mut invalid = _mm_setzero_si128();
        blocks::<16, 16>(class, text, out, 16, b'A', bytes_len, |src, dst| {
            let src = load16::<0, _>(src);
            let hi = _mm_and_si128(_mm_srli_epi32(src, 4), nibble);
            let lo = _mm_and_si128(src, nibble);
            let classes = _mm_and_si128(_mm_shuffle_epi8(lo_classes, lo), _mm_shuffle_epi8(hi_classes, hi));
            invalid = _mm_or_si128(invalid, classes);
            let index = _mm_add_epi8(hi, _mm_and_si128(_mm_cmpeq_epi8(src, special), special_index));
            let sextets = _mm_add_epi8(src, _mm_shuffle_epi8(roll, index));
            let words = _mm_madd_epi16(_mm_maddubs_epi16(sextets, _mm_set1_epi32(PAIRS)), _mm_set1_epi32(WORDS));
            store16::<0, _>(dst, _mm_shuffle_epi8(words, order));
        });
        _mm_movemask_epi8(_mm_cmpeq_epi8(invalid, _mm_setzero_si128())) == 0xffff
    }
}

pub(super) mod avx2 {
    use core::arch::x86_64::*;

    use super::{
        Class, DECODE_ORDER, ENCODE_ORDER, HIGH_SEXTETS, HIGH_SHIFTS, LOW_SEXTETS, LOW_SHIFTS, PAIRS, Profile, Tables,
        WORDS, bytes_len, load16, load32, ssse3, store32, symbols_len, whole_blocks,
    };

    /// A table of 16 bytes in each 128-bit lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn each_lane(table: &[u8; 16]) -> __m256i {
        _mm256_broadcastsi128_si256(load16::<0, _>(table))
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
        let shift = each_lane(&Tables::of(profile).nibbles.shift);
        let order = each_lane(&ENCODE_ORDER);
        // The 12 bytes of each 128-bit lane: the 32-bit words 0 to 2 and 3 to 5.
        let spread = _mm256_setr_epi32(0, 1, 2, 2, 3, 4, 5, 5);
        let (read, written) = whole_blocks::<32, 32>(bytes, out, 24, symbols_len, |src, dst| {
            let input = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(load32::<0, _>(src), spread), order);
            let high = _mm256_mulhi_epu16(
                _mm256_and_si256(input, _mm256_set1_epi32(HIGH_SEXTETS)),
                _mm256_set1_epi32(HIGH_SHIFTS),
            );
            let low = _mm256_mullo_epi16(
                _mm256_and_si256(input, _mm256_set1_epi32(LOW_SEXTETS)),
                _mm256_set1_epi32(LOW_SHIFTS),
            );
            let sextets = _mm256_or_si256(high, low);
            let class = _mm256_or_si256(
                _mm256_subs_epu8(sextets, _mm256_set1_epi8(51)),
                _mm256_and_si256(_mm256_cmpgt_epi8(_mm256_set1_epi8(26), sextets), _mm256_set1_epi8(13)),
            );
            store32::<0, _>(dst, _mm256_add_epi8(sextets, _mm256_shuffle_epi8(shift, class)));
        });
        // The rest, in vectors of 16 bytes.
        ssse3::encode(class, profile, &bytes[read..], &mut out[written..]);
    }

    #[target_feature(enable = "avx2")]
    pub(in super::super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let tables = &Tables::of(profile).nibbles;
        let (lo_classes, hi_classes) = (each_lane(&tables.lo), each_lane(&tables.hi));
        let roll = each_lane(&tables.roll);
        let special = _mm256_set1_epi8(tables.special.cast_signed());
        let special_index = _mm256_set1_epi8(tables.special_index.cast_signed());
        let order = each_lane(&DECODE_ORDER);
        // The 12 bytes of each 128-bit lane, together.
        let pack = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
        let nibble = _mm256_set1_epi8(0x0f);
        let mut invalid = _mm256_setzero_si256();
        let (read, written) = whole_blocks::<32, 32>(text, out, 32, bytes_len, |src, dst| {
            let src = load32::<0, _>(src);
            let hi = _mm256_and_si256(_mm256_srli_epi32(src, 4), nibble);
            let lo = _mm256_and_si256(src, nibble);
            let classes = _mm256_and_si256(_mm256_shuffle_epi8(lo_classes, lo), _mm256_shuffle_epi8(hi_classes, hi));
            invalid = _mm256_or_si256(invalid, classes);
            let index = _mm256_add_epi8(hi, _mm256_and_si256(_mm256_cmpeq_epi8(src, special), special_index));
            let sextets = _mm256_add_epi8(src, _mm256_shuffle_epi8(roll, index));
            let words = _mm256_madd_epi16(
                _mm256_maddubs_epi16(sextets, _mm256_set1_epi32(PAIRS)),
                _mm256_set1_epi32(WORDS),
            );
            store32::<0, _>(
                dst,
                _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(words, order), pack),
            );
        });
        // The rest, in vectors of 16 bytes.
        let rest_valid = ssse3::decode(class, profile, &text[read..], &mut out[written..]);
        (_mm256_movemask_epi8(_mm256_cmpeq_epi8(invalid, _mm256_setzero_si256())) == -1) & rest_valid
    }
}

/// The AVX-512BW kernels: the method of the AVX2 kernels on 64-byte vectors, with the shifts
/// `vpsrlvw` and `vpsllvw` in place of the multiplies. They use only instructions of the DOIT list
/// of Intel.
pub(super) mod avx512bw {
    use core::arch::x86_64::*;

    use super::{
        DECODE_ORDER, ENCODE_ORDER, HIGH_SEXTETS, LOW_SEXTETS, PAIRS, Profile, Tables, WORDS, decode_64, decode_rest,
        encode_64, encode_rest, load16,
    };

    /// A table of 16 bytes in each 128-bit lane.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn each_lane(table: &[u8; 16]) -> __m512i {
        _mm512_broadcast_i32x4(load16::<0, _>(table))
    }

    /// The block of the encoder of `profile`: the 64 characters of the first 48 bytes of a vector.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn encoder(profile: Profile) -> impl FnMut(__m512i) -> __m512i {
        let shift = each_lane(&Tables::of(profile).nibbles.shift);
        let order = each_lane(&ENCODE_ORDER);
        // The 12 bytes of each 128-bit lane: the 32-bit words 0 to 2, 3 to 5, 6 to 8, and 9 to 11.
        let spread = _mm512_setr_epi32(0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11);
        move |src| {
            let input = _mm512_shuffle_epi8(_mm512_permutexvar_epi32(spread, src), order);
            // The shifts of each 16-bit lane in place of the multiplies of the AVX2 kernel.
            let high = _mm512_srlv_epi16(
                _mm512_and_si512(input, _mm512_set1_epi32(HIGH_SEXTETS)),
                _mm512_set1_epi32(0x0006_000a),
            );
            let low = _mm512_sllv_epi16(
                _mm512_and_si512(input, _mm512_set1_epi32(LOW_SEXTETS)),
                _mm512_set1_epi32(0x0008_0004),
            );
            let sextets = _mm512_or_si512(high, low);
            let below_26 = _mm512_cmpgt_epi8_mask(_mm512_set1_epi8(26), sextets);
            let class = _mm512_mask_blend_epi8(
                below_26,
                _mm512_subs_epu8(sextets, _mm512_set1_epi8(51)),
                _mm512_set1_epi8(13),
            );
            _mm512_add_epi8(sextets, _mm512_shuffle_epi8(shift, class))
        }
    }

    /// The block of the decoder of `profile`: the 48 bytes of a vector of characters. It ORs the
    /// classes of the characters into `invalid`, which is not zero for a byte outside the alphabet.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn decoder(profile: Profile, invalid: &mut __m512i) -> impl FnMut(__m512i) -> __m512i {
        let tables = &Tables::of(profile).nibbles;
        let (lo_classes, hi_classes) = (each_lane(&tables.lo), each_lane(&tables.hi));
        let roll = each_lane(&tables.roll);
        let special = _mm512_set1_epi8(tables.special.cast_signed());
        let special_index = _mm512_set1_epi8(tables.special_index.cast_signed());
        let order = each_lane(&DECODE_ORDER);
        // The 12 bytes of each 128-bit lane, together.
        let pack = _mm512_setr_epi32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
        let nibble = _mm512_set1_epi8(0x0f);
        move |src| {
            let hi = _mm512_and_si512(_mm512_srli_epi32(src, 4), nibble);
            let lo = _mm512_and_si512(src, nibble);
            let classes = _mm512_and_si512(_mm512_shuffle_epi8(lo_classes, lo), _mm512_shuffle_epi8(hi_classes, hi));
            *invalid = _mm512_or_si512(*invalid, classes);
            let index = _mm512_add_epi8(
                hi,
                _mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(src, special), special_index),
            );
            let sextets = _mm512_add_epi8(src, _mm512_shuffle_epi8(roll, index));
            let words = _mm512_madd_epi16(
                _mm512_maddubs_epi16(sextets, _mm512_set1_epi32(PAIRS)),
                _mm512_set1_epi32(WORDS),
            );
            _mm512_permutexvar_epi32(pack, _mm512_shuffle_epi8(words, order))
        }
    }

    // A build with the feature fixes the kernel, and then the kernel runs inline in the caller.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn encode(profile: Profile, bytes: &[u8], out: &mut [u8]) {
        encode_64(bytes, out, encoder(profile));
    }

    /// Encodes at most `ONE_MAX` bytes, with no loop.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn encode_one(profile: Profile, bytes: &[u8], out: &mut [u8]) {
        encode_rest(bytes, out, encoder(profile));
    }

    // A build with the feature fixes the kernel, and then the kernel runs inline in the caller.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode(profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let mut invalid = _mm512_setzero_si512();
        decode_64(text, out, decoder(profile, &mut invalid));
        _mm512_test_epi8_mask(invalid, invalid) == 0
    }

    /// Decodes a text of at most `ONE_MAX` bytes of result, with no loop.
    #[cfg_attr(target_feature = "avx512bw", inline)]
    #[target_feature(enable = "avx512bw")]
    pub(in super::super) fn decode_one(profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let mut invalid = _mm512_setzero_si512();
        decode_rest(text, out, decoder(profile, &mut invalid));
        _mm512_test_epi8_mask(invalid, invalid) == 0
    }
}

/// The AVX-512 VBMI kernels: byte permutes in place of the nibble lookups and the shuffles.
pub(super) mod avx512vbmi {
    use core::arch::x86_64::*;

    use super::{PAIRS, Profile, Tables, WORDS, decode_64, decode_rest, encode_64, encode_rest, load64};

    /// The block of the encoder of `profile`: the 64 characters of the first 48 bytes of a vector.
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    fn encoder(profile: Profile) -> impl FnMut(__m512i) -> __m512i {
        let alphabet = load64::<0, _>(&Tables::of(profile).alphabet);
        // The bytes `b1 b0 b2 b1` of each three input bytes, in each 32-bit lane.
        let order = _mm512_setr_epi32(
            0x0102_0001,
            0x0405_0304,
            0x0708_0607,
            0x0a0b_090a,
            0x0d0e_0c0d,
            0x1011_0f10,
            0x1314_1213,
            0x1617_1516,
            0x191a_1819,
            0x1c1d_1b1c,
            0x1f20_1e1f,
            0x2223_2122,
            0x2526_2425,
            0x2829_2728,
            0x2b2c_2a2b,
            0x2e2f_2d2e,
        );
        // The bit offset of each sextet in each 64-bit lane.
        let shifts = _mm512_set1_epi64(0x3036_242a_1016_040a);
        move |src| {
            let sextets = _mm512_multishift_epi64_epi8(shifts, _mm512_permutexvar_epi8(order, src));
            _mm512_permutexvar_epi8(sextets, alphabet)
        }
    }

    /// The block of the decoder of `profile`: the 48 bytes of a vector of characters. It ORs each
    /// character and its sextet into `invalid`: bit 7 is set for a byte outside the alphabet.
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    fn decoder(profile: Profile, invalid: &mut __m512i) -> impl FnMut(__m512i) -> __m512i {
        let [low, high] = &Tables::of(profile).sextets;
        let (low, high) = (load64::<0, _>(low), load64::<0, _>(high));
        // The three bytes of each 32-bit lane, first byte first, together.
        let pack = _mm512_setr_epi32(
            0x0600_0102,
            0x090a_0405,
            0x0c0d_0e08,
            0x1610_1112,
            0x191a_1415,
            0x1c1d_1e18,
            0x2620_2122,
            0x292a_2425,
            0x2c2d_2e28,
            0x3630_3132,
            0x393a_3435,
            0x3c3d_3e38,
            0,
            0,
            0,
            0,
        );
        move |src| {
            let sextets = _mm512_permutex2var_epi8(low, src, high);
            *invalid = _mm512_or_si512(*invalid, _mm512_or_si512(src, sextets));
            let words = _mm512_madd_epi16(
                _mm512_maddubs_epi16(sextets, _mm512_set1_epi32(PAIRS)),
                _mm512_set1_epi32(WORDS),
            );
            _mm512_permutexvar_epi8(pack, words)
        }
    }

    // A build with the features fixes the kernel, and then the kernel runs inline in the caller.
    #[cfg_attr(all(target_feature = "avx512bw", target_feature = "avx512vbmi"), inline)]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    pub(in super::super) fn encode(profile: Profile, bytes: &[u8], out: &mut [u8]) {
        encode_64(bytes, out, encoder(profile));
    }

    /// Encodes at most `ONE_MAX` bytes, with no loop.
    #[cfg_attr(all(target_feature = "avx512bw", target_feature = "avx512vbmi"), inline)]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    pub(in super::super) fn encode_one(profile: Profile, bytes: &[u8], out: &mut [u8]) {
        encode_rest(bytes, out, encoder(profile));
    }

    // A build with the features fixes the kernel, and then the kernel runs inline in the caller.
    #[cfg_attr(all(target_feature = "avx512bw", target_feature = "avx512vbmi"), inline)]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    pub(in super::super) fn decode(profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let mut invalid = _mm512_setzero_si512();
        decode_64(text, out, decoder(profile, &mut invalid));
        _mm512_movepi8_mask(invalid) == 0
    }

    /// Decodes a text of at most `ONE_MAX` bytes of result, with no loop.
    #[cfg_attr(all(target_feature = "avx512bw", target_feature = "avx512vbmi"), inline)]
    #[target_feature(enable = "avx512bw,avx512vbmi")]
    pub(in super::super) fn decode_one(profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        let mut invalid = _mm512_setzero_si512();
        decode_rest(text, out, decoder(profile, &mut invalid));
        _mm512_movepi8_mask(invalid) == 0
    }
}

/// Encodes in blocks of 48 bytes with `block`, which gives the 64 characters of the first 48
/// bytes of a vector. A masked load of 48 lanes, with a constant mask, reads each block, so the
/// loop has no branch on the place of a block. Each step of the loop runs two blocks.
#[inline]
#[target_feature(enable = "avx512bw")]
fn encode_64(bytes: &[u8], out: &mut [u8], mut block: impl FnMut(__m512i) -> __m512i) {
    let zero = _mm512_setzero_si512();
    let (pairs, rest) = bytes.as_chunks::<96>();
    let (texts, out) = out.split_at_mut(pairs.len() * 128);
    for (pair, text) in pairs.iter().zip(texts.as_chunks_mut::<128>().0) {
        for (chunk, text) in pair.as_chunks::<48>().0.iter().zip(text.as_chunks_mut::<64>().0) {
            store64::<0, _>(text, block(load_masked(chunk, zero)));
        }
    }
    let (chunks, rest) = rest.as_chunks::<48>();
    let (texts, tail) = out.split_at_mut(chunks.len() * 64);
    for (chunk, text) in chunks.iter().zip(texts.as_chunks_mut::<64>().0) {
        store64::<0, _>(text, block(load_masked(chunk, zero)));
    }
    if !rest.is_empty() {
        encode_rest(rest, tail, block);
    }
}

/// The most bytes of one block of 48 bytes: the entries of one block take at most 64 characters.
pub(super) const ONE_MAX: usize = 48;

/// Encodes at most `ONE_MAX` bytes in one block with `block`, with masked loads and stores.
#[inline]
#[target_feature(enable = "avx512bw")]
fn encode_rest(bytes: &[u8], out: &mut [u8], mut block: impl FnMut(__m512i) -> __m512i) {
    store_masked(out, block(load_masked(bytes, _mm512_setzero_si512())));
}

/// Decodes a text of at most `ONE_MAX` bytes of result in one block with `block`, with masked
/// loads and stores. 'A', which is zero, is in place of the characters after the end.
#[inline]
#[target_feature(enable = "avx512bw")]
fn decode_rest(text: &[u8], out: &mut [u8], mut block: impl FnMut(__m512i) -> __m512i) {
    store_masked(out, block(load_masked(text, _mm512_set1_epi8(b'A'.cast_signed()))));
}

/// Decodes in blocks of 64 characters with `block`, which gives the 48 bytes of a vector of
/// characters. Each step of the loop runs four blocks, and three full stores write their 192 bytes:
/// two-source permutes of 64-bit words join the blocks. A masked store of 48 lanes, with a constant
/// mask, writes each of the last blocks, so the loops have no branch on the place of a block.
#[inline]
#[target_feature(enable = "avx512bw")]
fn decode_64(text: &[u8], out: &mut [u8], mut block: impl FnMut(__m512i) -> __m512i) {
    // The six words of each block in turn: words 0 to 7 of the first source, then 8 to 15 of the
    // second.
    let first = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 8, 9);
    let second = _mm512_setr_epi64(2, 3, 4, 5, 8, 9, 10, 11);
    let third = _mm512_setr_epi64(4, 5, 8, 9, 10, 11, 12, 13);
    let (quads, rest) = text.as_chunks::<256>();
    let (outs, out) = out.split_at_mut(quads.len() * 192);
    for (quad, bytes) in quads.iter().zip(outs.as_chunks_mut::<192>().0) {
        let (a, b) = (block(load64::<0, _>(quad)), block(load64::<64, _>(quad)));
        let (c, d) = (block(load64::<128, _>(quad)), block(load64::<192, _>(quad)));
        store64::<0, _>(bytes, _mm512_permutex2var_epi64(a, first, b));
        store64::<64, _>(bytes, _mm512_permutex2var_epi64(b, second, c));
        store64::<128, _>(bytes, _mm512_permutex2var_epi64(c, third, d));
    }
    let (chunks, rest) = rest.as_chunks::<64>();
    let (outs, tail) = out.split_at_mut(chunks.len() * 48);
    for (chunk, bytes) in chunks.iter().zip(outs.as_chunks_mut::<48>().0) {
        store_masked(bytes, block(load64::<0, _>(chunk)));
    }
    if !rest.is_empty() {
        decode_rest(rest, tail, block);
    }
}
