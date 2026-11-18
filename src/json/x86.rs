//! The x86_64 kernels, AVX2 and AVX-512. Nibble tables classify the bytes and check UTF-8, and a
//! carry-less multiply builds the string mask.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::x86_64::*;

use super::{
    block::{self, Block, Carry, Classes},
    simd,
};

/// The AVX2 kernel.
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
pub(super) fn avx2(bytes: &[u8], before: u32, carry: Carry) -> Block {
    let halves = match bytes.first_chunk() {
        Some(bytes) => load::halves(bytes),
        None => {
            let words = block::tail(bytes).map(u64::cast_signed);
            [0, 4].map(|at| _mm256_set_epi64x(words[at + 3], words[at + 2], words[at + 1], words[at]))
        }
    };
    let table = |table: &[u8; 16]| _mm256_broadcastsi128_si256(load::table(table));
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
    // The XOR of bits `0..=i` in each bit `i`: the carry-less product with all ones.
    let prefix_xor = |bits: u64| {
        let product = _mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(bits.cast_signed()), _mm_set1_epi8(-1));
        _mm_cvtsi128_si64(product).cast_unsigned()
    };
    simd::scan(classes, carry, prefix_xor)
}

/// The AVX-512 kernel.
#[inline]
#[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
pub(super) fn avx512(bytes: &[u8], before: u32, carry: Carry) -> Block {
    let input = load::masked(bytes);
    let table = |table: &[u8; 16]| _mm512_broadcast_i32x4(load::table(table));
    let set = |byte: u8| _mm512_set1_epi8(byte.cast_signed());
    let control = _mm512_cmplt_epu8_mask(input, set(0x20));
    let curly = _mm512_cmpeq_epi8_mask(
        _mm512_shuffle_epi8(table(&simd::STRUCTURAL), input),
        _mm512_or_si512(input, set(0x20)),
    );
    let classes = Classes {
        structural: curly & !control,
        whitespace: _mm512_cmpeq_epi8_mask(_mm512_shuffle_epi8(table(&simd::WHITESPACE), input), input),
        quote: _mm512_cmpeq_epi8_mask(input, set(b'"')),
        backslash: _mm512_cmpeq_epi8_mask(input, set(b'\\')),
        control,
        utf8_error: utf8_error_avx512(input, before),
    };
    // The XOR of bits `0..=i` in each bit `i`: the carry-less product with all ones.
    let prefix_xor = |bits: u64| {
        let product = _mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(bits.cast_signed()), _mm_set1_epi8(-1));
        _mm_cvtsi128_si64(product).cast_unsigned()
    };
    simd::scan(classes, carry, prefix_xor)
}

/// The bit of each byte of `halves` that `test` sets to all ones.
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn bits(halves: [__m256i; 2], test: impl Fn(__m256i) -> __m256i) -> u64 {
    let [low, high] = halves.map(|half| _mm256_movemask_epi8(test(half)).cast_unsigned());
    u64::from(low) | u64::from(high) << 32
}

/// The UTF-8 check of [`block::utf8_error_at`] on each byte of `halves`.
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn utf8_error_avx2(halves: [__m256i; 2], before: u32) -> bool {
    let [low, high] = halves;
    if _mm256_movemask_epi8(_mm256_or_si256(low, high)) == 0 {
        return block::ends_incomplete(before);
    }
    let tables = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH]
        .map(|table| _mm256_broadcastsi128_si256(load::table(&table)));
    let previous = _mm256_set_epi32(before.cast_signed(), 0, 0, 0, 0, 0, 0, 0);
    let error = _mm256_or_si256(check_avx2(low, previous, tables), check_avx2(high, low, tables));
    _mm256_testz_si256(error, error) == 0
}

/// A byte that is not zero at each byte of `input` that ends an error, after the bytes of
/// `previous`.
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
fn check_avx2(input: __m256i, previous: __m256i, [first_high, first_low, second_high]: [__m256i; 3]) -> __m256i {
    // The 16 bytes before each 16-byte lane of `input`.
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
    // Bit 7 is set where `second` is at least 0xe0 or `first` is at least 0xf0.
    let continues = _mm256_or_si256(
        _mm256_subs_epu8(second, _mm256_set1_epi8(0x60)),
        _mm256_subs_epu8(first, _mm256_set1_epi8(0x70)),
    );
    _mm256_xor_si256(_mm256_and_si256(continues, _mm256_set1_epi8(i8::MIN)), pair)
}

/// The UTF-8 check of [`block::utf8_error_at`] on each byte of `input`.
#[inline]
#[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
fn utf8_error_avx512(input: __m512i, before: u32) -> bool {
    if _mm512_movepi8_mask(input) == 0 {
        return block::ends_incomplete(before);
    }
    let [first_high, first_low, second_high] = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH]
        .map(|table| _mm512_broadcast_i32x4(load::table(&table)));
    let previous = _mm512_set_epi32(before.cast_signed(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // The 16 bytes before each 16-byte lane of `input`.
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

/// The loads of the kernels.
#[expect(unsafe_code, reason = "each load reads only the bytes of the array that it takes")]
mod load {
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm512_mask_loadu_epi8, _mm512_set1_epi8,
    };

    /// The 16 bytes of `table`.
    #[inline(always)]
    pub(super) fn table(table: &[u8; 16]) -> __m128i {
        // SAFETY: the load reads the 16 bytes of `table`, and it needs no alignment. Each x86_64 CPU
        // has SSE2.
        unsafe { _mm_loadu_si128(table.as_ptr().cast()) }
    }

    /// The two halves of `bytes`.
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
    pub(super) fn halves(bytes: &[u8; 64]) -> [__m256i; 2] {
        let [low, high] = [0, 32].map(|at| {
            // SAFETY: the load reads the 32 bytes at `at`, which are in `bytes`, and it needs no
            // alignment.
            unsafe { _mm256_loadu_si256(bytes.as_ptr().add(at).cast()) }
        });
        [low, high]
    }

    /// The first 64 bytes of `bytes`, with spaces after them when `bytes` has fewer.
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
    pub(super) fn masked(bytes: &[u8]) -> __m512i {
        // The first `len` lanes, at most 64: `64 - len` is at most 64, and a shift of 64 gives 0.
        let len = u32::try_from(bytes.len().min(64)).unwrap_or(64);
        let lanes = u64::MAX.unbounded_shr(64 - len);
        // SAFETY: the mask holds only the lanes of bytes of `bytes`, and a lane outside the mask
        // reads no memory. The load needs no alignment.
        unsafe { _mm512_mask_loadu_epi8(_mm512_set1_epi8(b' '.cast_signed()), lanes, bytes.as_ptr().cast()) }
    }
}
