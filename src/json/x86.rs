use core::arch::x86_64::*;

use super::{
    block::{self, Block, Carry, Classes},
    simd,
};
use crate::vector::x86::{load_masked, load16, load32};

#[inline(never)]
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, PCLMULQDQ.
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
pub(super) fn avx2(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
    let halves = match bytes.first_chunk() {
        Some(bytes) => [load32::<0, 64>(bytes), load32::<32, 64>(bytes)],
        None => {
            let words = block::tail(bytes).map(u64::cast_signed);
            [0, 4].map(|at| _mm256_set_epi64x(words[at + 3], words[at + 2], words[at + 1], words[at]))
        }
    };
    let table = |table: &[u8; 16]| _mm256_broadcastsi128_si256(load16::<0, 16>(table));
    let (whitespace, structural) = (table(&simd::WHITESPACE), table(&simd::STRUCTURAL));
    let set = |byte: u8| _mm256_set1_epi8(byte.cast_signed());
    let control = bits(halves, |half| {
        _mm256_cmpeq_epi8(_mm256_max_epu8(half, set(0x1f)), set(0x1f))
    });
    let curly = bits(halves, |half| {
        _mm256_cmpeq_epi8(_mm256_shuffle_epi8(structural, half), _mm256_or_si256(half, set(0x20)))
    });
    let classes = Classes {
        structural: curly & !control,
        whitespace: bits(halves, |half| {
            _mm256_cmpeq_epi8(_mm256_shuffle_epi8(whitespace, half), half)
        }),
        quote: bits(halves, |half| _mm256_cmpeq_epi8(half, set(b'"'))),
        backslash: bits(halves, |half| _mm256_cmpeq_epi8(half, set(b'\\'))),
        control,
        utf8_error: utf8_error_avx2(halves, before),
    };
    let prefix_xor = |bits: u64| {
        let product = _mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(bits.cast_signed()), _mm_set1_epi8(-1));
        _mm_cvtsi128_si64(product).cast_unsigned()
    };
    simd::scan(classes, carry, prefix_xor)
}

#[inline(never)]
/// # Safety
/// The CPU must support AVX512BW, AVX512VL, AVX512VBMI2, PCLMULQDQ, BMI1, BMI2.
#[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
pub(super) fn avx512(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
    let input = load_masked(bytes, _mm512_set1_epi8(b' '.cast_signed()));
    let table = |table: &[u8; 16]| _mm512_broadcast_i32x4(load16::<0, 16>(table));
    let set = |byte: u8| _mm512_set1_epi8(byte.cast_signed());
    let control = _mm512_cmplt_epu8_mask(input, set(0x20));
    let (quote, backslash) = (
        _mm512_cmpeq_epi8_mask(input, set(b'"')),
        _mm512_cmpeq_epi8_mask(input, set(b'\\')),
    );
    if carry.in_plain_string()
        && control | quote | backslash | _mm512_movepi8_mask(input) == 0
        && !block::ends_incomplete(before)
    {
        return Block { tokens: 0, carry };
    }
    let curly = _mm512_cmpeq_epi8_mask(
        _mm512_shuffle_epi8(table(&simd::STRUCTURAL), input),
        _mm512_or_si512(input, set(0x20)),
    );
    let classes = Classes {
        structural: curly & !control,
        whitespace: _mm512_cmpeq_epi8_mask(_mm512_shuffle_epi8(table(&simd::WHITESPACE), input), input),
        quote,
        backslash,
        control,
        utf8_error: utf8_error_avx512(input, before),
    };
    let prefix_xor = |bits: u64| {
        let product = _mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(bits.cast_signed()), _mm_set1_epi8(-1));
        _mm_cvtsi128_si64(product).cast_unsigned()
    };
    simd::scan(classes, carry, prefix_xor)
}

#[inline]
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, PCLMULQDQ.
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn bits(halves: [__m256i; 2], test: impl Fn(__m256i) -> __m256i) -> u64 {
    let [low, high] = halves.map(|half| _mm256_movemask_epi8(test(half)).cast_unsigned());
    u64::from(low) | u64::from(high) << 32
}

#[inline]
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, PCLMULQDQ.
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn utf8_error_avx2(halves: [__m256i; 2], before: u32) -> bool {
    let [low, high] = halves;
    if _mm256_movemask_epi8(_mm256_or_si256(low, high)) == 0 {
        return block::ends_incomplete(before);
    }
    let tables = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH]
        .map(|table| _mm256_broadcastsi128_si256(load16::<0, 16>(&table)));
    let previous = _mm256_set_epi32(before.cast_signed(), 0, 0, 0, 0, 0, 0, 0);
    let error = _mm256_or_si256(check_avx2(low, previous, tables), check_avx2(high, low, tables));
    _mm256_testz_si256(error, error) == 0
}

#[inline]
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, PCLMULQDQ.
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn check_avx2(input: __m256i, previous: __m256i, [first_high, first_low, second_high]: [__m256i; 3]) -> __m256i {
    let before = _mm256_permute2x128_si256::<0x21>(previous, input);
    let (first, second, third) = (
        _mm256_alignr_epi8::<13>(input, before),
        _mm256_alignr_epi8::<14>(input, before),
        _mm256_alignr_epi8::<15>(input, before),
    );
    let nibble = _mm256_set1_epi8(0x0f);
    let high = |bytes: __m256i| _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), nibble);
    let pair = _mm256_and_si256(
        _mm256_and_si256(
            _mm256_shuffle_epi8(first_high, high(third)),
            _mm256_shuffle_epi8(first_low, _mm256_and_si256(third, nibble)),
        ),
        _mm256_shuffle_epi8(second_high, high(input)),
    );
    let continues = _mm256_or_si256(
        _mm256_subs_epu8(second, _mm256_set1_epi8(0x60)),
        _mm256_subs_epu8(first, _mm256_set1_epi8(0x70)),
    );
    _mm256_xor_si256(_mm256_and_si256(continues, _mm256_set1_epi8(i8::MIN)), pair)
}

#[inline]
/// # Safety
/// The CPU must support AVX512BW, AVX512VL, AVX512VBMI2, PCLMULQDQ, BMI1, BMI2.
#[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
fn utf8_error_avx512(input: __m512i, before: u32) -> bool {
    if _mm512_movepi8_mask(input) == 0 {
        return block::ends_incomplete(before);
    }
    let [first_high, first_low, second_high] = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH]
        .map(|table| _mm512_broadcast_i32x4(load16::<0, 16>(&table)));
    let previous = _mm512_set_epi32(before.cast_signed(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let before = _mm512_alignr_epi64::<6>(input, previous);
    let (first, second, third) = (
        _mm512_alignr_epi8::<13>(input, before),
        _mm512_alignr_epi8::<14>(input, before),
        _mm512_alignr_epi8::<15>(input, before),
    );
    let nibble = _mm512_set1_epi8(0x0f);
    let high = |bytes: __m512i| _mm512_and_si512(_mm512_srli_epi16::<4>(bytes), nibble);
    let pair = _mm512_and_si512(
        _mm512_and_si512(
            _mm512_shuffle_epi8(first_high, high(third)),
            _mm512_shuffle_epi8(first_low, _mm512_and_si512(third, nibble)),
        ),
        _mm512_shuffle_epi8(second_high, high(input)),
    );
    let continues = _mm512_or_si512(
        _mm512_subs_epu8(second, _mm512_set1_epi8(0x60)),
        _mm512_subs_epu8(first, _mm512_set1_epi8(0x70)),
    );
    let error = _mm512_xor_si512(_mm512_and_si512(continues, _mm512_set1_epi8(i8::MIN)), pair);
    _mm512_test_epi8_mask(error, error) != 0
}

#[expect(unsafe_code, reason = "each x86_64 CPU has SSE2")]
pub(super) mod escape {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_min_epu8, _mm_movemask_epi8, _mm_or_si128, _mm_set_epi64x, _mm_set1_epi8,
    };

    pub(in super::super) const BITS: u32 = 1;

    #[inline(always)]
    pub(in super::super) fn escapes(block: &[u8; 16]) -> u64 {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `scan`.
        unsafe { scan(crate::vector::x86::load16::<0, 16>(block)) }
    }

    #[inline(always)]
    pub(in super::super) fn escapes_of_words(low: u64, high: u64) -> u64 {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `_mm_set_epi64x` and `scan`.
        unsafe { scan(_mm_set_epi64x(high.cast_signed(), low.cast_signed())) }
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    fn scan(bytes: __m128i) -> u64 {
        let quote = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b'"'.cast_signed()));
        let backslash = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b'\\'.cast_signed()));
        let control = _mm_cmpeq_epi8(_mm_min_epu8(bytes, _mm_set1_epi8(0x1f)), bytes);
        u64::from(_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(quote, backslash), control)).cast_unsigned())
    }
}

#[expect(unsafe_code, reason = "each x86_64 CPU has SSE2")]
pub(super) mod digits {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi64, _mm_cmpgt_epi8, _mm_cvtsi128_si64, _mm_movemask_epi8, _mm_mul_epu32, _mm_mulhi_epu16,
        _mm_mullo_epi16, _mm_or_si128, _mm_set_epi64x, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi64x,
        _mm_setzero_si128, _mm_shuffle_epi32, _mm_slli_epi16, _mm_slli_epi32, _mm_srli_epi16, _mm_srli_epi64,
        _mm_sub_epi16, _mm_unpackhi_epi64,
    };

    use super::super::float::insert;

    #[inline(always)]
    pub(in super::super) fn sixteen(value: u64) -> (u64, u64) {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `split` and `words`.
        unsafe { words(split(value)) }
    }

    #[inline(always)]
    pub(in super::super) fn mantissa(digits: u64, extra: u8, whole: usize, dest: &mut [u8]) -> usize {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `text`, `store`, and `words`.
        let (text, count) = unsafe { text(digits, extra) };
        // SAFETY: as above.
        unsafe { store(dest, 1, text) };
        match whole {
            1..8 => {
                // SAFETY: as above.
                let (first, _) = unsafe { words(text) };
                dest[..8].copy_from_slice(&insert(first, whole, b'.').0.to_le_bytes());
            }
            8..16 => {
                // SAFETY: as above.
                let (first, second) = unsafe { words(text) };
                dest[..8].copy_from_slice(&first.to_le_bytes());
                dest[8..16].copy_from_slice(&insert(second, whole - 8, b'.').0.to_le_bytes());
            }
            _ => {
                // SAFETY: as above.
                unsafe { store(dest, 0, text) };
                dest[16] = b'.';
            }
        }
        dest[17] = b'0' + extra;
        count
    }

    #[inline(always)]
    pub(in super::super) fn plain(digits: u64, extra: u8, at: usize, dest: &mut [u8]) -> usize {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `text` and `store`.
        let (text, count) = unsafe { text(digits, extra) };
        // SAFETY: as above.
        unsafe { store(dest, at, text) };
        dest[at + 16] = b'0' + extra;
        count
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    fn text(digits: u64, extra: u8) -> (__m128i, usize) {
        let values = split(digits);
        let nonzero = _mm_movemask_epi8(_mm_cmpgt_epi8(values, _mm_setzero_si128())).cast_unsigned();
        let count = 32 - (nonzero | u32::from(extra != 0) << 16).leading_zeros() as usize;
        (_mm_or_si128(values, _mm_set1_epi8(b'0'.cast_signed())), count)
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    fn words(bytes: __m128i) -> (u64, u64) {
        (
            _mm_cvtsi128_si64(bytes).cast_unsigned(),
            _mm_cvtsi128_si64(_mm_unpackhi_epi64(bytes, bytes)).cast_unsigned(),
        )
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    fn store(dest: &mut [u8], at: usize, bytes: __m128i) {
        if let Some(dest) = dest.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<16>) {
            crate::vector::x86::store16::<0, 16>(dest, bytes);
        }
    }

    /// Each step splits each lane into its quotient and its remainder by a power of 10, the
    /// quotient in the low half of the lane. A multiply that keeps the high bits gives each
    /// quotient: `x / 10^4` is `(x × 109_951_163) >> 40` for `x < 10^8`, `x / 100` is
    /// `(x × 5_243) >> 19` for `x < 10^4`, and `x / 10` is `(x × 6_554) >> 16` for `x < 100`:
    /// the digit values of `value`, under 10^16, the first in byte 0.
    #[inline(always)]
    /// # Safety
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    fn split(value: u64) -> __m128i {
        let (high, low) = (value / 100_000_000, value % 100_000_000);
        let halves = _mm_set_epi64x(low.cast_signed(), high.cast_signed());
        let quotients = _mm_srli_epi64::<40>(_mm_mul_epu32(halves, _mm_set1_epi64x(109_951_163)));
        let fourths = _mm_add_epi64(halves, _mm_mul_epu32(quotients, _mm_set1_epi64x((1 << 32) - 10_000)));
        let fourths = _mm_shuffle_epi32::<0b10_11_00_01>(fourths);
        let hundreds = _mm_srli_epi16::<3>(_mm_mulhi_epu16(fourths, _mm_set1_epi16(5_243)));
        let remainders = _mm_sub_epi16(fourths, _mm_mullo_epi16(hundreds, _mm_set1_epi16(100)));
        let pairs = _mm_or_si128(hundreds, _mm_slli_epi32::<16>(remainders));
        let tens = _mm_mulhi_epu16(pairs, _mm_set1_epi16(6_554));
        _mm_sub_epi16(_mm_slli_epi16::<8>(pairs), _mm_mullo_epi16(tens, _mm_set1_epi16(2_559)))
    }
}
