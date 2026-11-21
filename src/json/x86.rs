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
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2,pclmulqdq")]
pub(super) fn avx2(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
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
#[inline(never)]
#[target_feature(enable = "avx512bw,avx512vl,avx512vbmi2,pclmulqdq,bmi1,bmi2")]
pub(super) fn avx512(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
    let input = load::masked(bytes);
    let table = |table: &[u8; 16]| _mm512_broadcast_i32x4(load::table(table));
    let set = |byte: u8| _mm512_set1_epi8(byte.cast_signed());
    let control = _mm512_cmplt_epu8_mask(input, set(0x20));
    let (quote, backslash) = (
        _mm512_cmpeq_epi8_mask(input, set(b'"')),
        _mm512_cmpeq_epi8_mask(input, set(b'\\')),
    );
    // A block in a string with no quote, `\`, byte under 0x20, or byte of 0x80 or more stays in the
    // string and has no token, as the full scan gives. Its UTF-8 is valid if no sequence before it
    // needs more bytes. A long string, as in a JWKS, is mostly such blocks.
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

/// The scan of the writer for the bytes that a JSON string escapes, with SSE2, which each x86_64
/// CPU has, so the build fixes the kernel.
#[expect(unsafe_code, reason = "each x86_64 CPU has SSE2")]
pub(super) mod escape {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_min_epu8, _mm_movemask_epi8, _mm_or_si128, _mm_set_epi64x, _mm_set1_epi8,
    };

    /// The bits of a mask for each byte of [`escapes`].
    pub(in super::super) const BITS: u32 = 1;

    /// A mask with [`BITS`] bits for each byte of `block` that a JSON string escapes: `"`, `\\`,
    /// or a byte under 0x20, the first byte in the low bits.
    #[inline(always)]
    pub(in super::super) fn escapes(block: &[u8; 16]) -> u64 {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `scan`.
        unsafe { scan(super::load::table(block)) }
    }

    /// [`escapes`] of the 16 bytes of the little-endian words `low` and `high`, with no load.
    #[inline(always)]
    pub(in super::super) fn escapes_of_words(low: u64, high: u64) -> u64 {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `_mm_set_epi64x` and `scan`.
        unsafe { scan(_mm_set_epi64x(high.cast_signed(), low.cast_signed())) }
    }

    #[inline(always)]
    #[target_feature(enable = "sse2")]
    fn scan(bytes: __m128i) -> u64 {
        let quote = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b'"'.cast_signed()));
        let backslash = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b'\\'.cast_signed()));
        // A byte is under 0x20 exactly when its unsigned minimum with 0x1f is the byte.
        let control = _mm_cmpeq_epi8(_mm_min_epu8(bytes, _mm_set1_epi8(0x1f)), bytes);
        u64::from(_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(quote, backslash), control)).cast_unsigned())
    }
}

/// The digits of the printers, with SSE2, which each x86_64 CPU has, so the build fixes the kernel.
#[expect(unsafe_code, reason = "each x86_64 CPU has SSE2")]
pub(super) mod digits {
    use core::arch::x86_64::{
        _mm_add_epi64, _mm_cvtsi128_si64, _mm_mul_epu32, _mm_mulhi_epu16, _mm_mullo_epi16, _mm_or_si128,
        _mm_set_epi64x, _mm_set1_epi16, _mm_set1_epi64x, _mm_shuffle_epi32, _mm_slli_epi16, _mm_slli_epi32,
        _mm_srli_epi16, _mm_srli_epi64, _mm_sub_epi16, _mm_unpackhi_epi64,
    };

    /// [`super::super::integer::eight_digits`] of the two halves of `value`, under 10^16, high
    /// first.
    #[inline(always)]
    pub(in super::super) fn sixteen(value: u64) -> (u64, u64) {
        // SAFETY: each x86_64 CPU has SSE2, the feature of `split`.
        unsafe { split(value) }
    }

    /// Each step splits each lane into its quotient and its remainder by a power of 10, the
    /// quotient in the low half of the lane. A multiply that keeps the high bits gives each
    /// quotient: `x / 10^4` is `(x × 109_951_163) >> 40` for `x < 10^8`, `x / 100` is
    /// `(x × 5_243) >> 19` for `x < 10^4`, and `x / 10` is `(x × 6_554) >> 16` for `x < 100`.
    #[inline(always)]
    #[target_feature(enable = "sse2")]
    fn split(value: u64) -> (u64, u64) {
        let (high, low) = (value / 100_000_000, value % 100_000_000);
        let halves = _mm_set_epi64x(low.cast_signed(), high.cast_signed());
        let quotients = _mm_srli_epi64::<40>(_mm_mul_epu32(halves, _mm_set1_epi64x(109_951_163)));
        // `x + q × (2^32 − 10^4)` is the remainder in the low half and the quotient in the high
        // half, and the shuffle swaps the halves.
        let fourths = _mm_add_epi64(halves, _mm_mul_epu32(quotients, _mm_set1_epi64x((1 << 32) - 10_000)));
        let fourths = _mm_shuffle_epi32::<0b10_11_00_01>(fourths);
        // The high 16 bits of each 32-bit lane are 0, so their products are 0.
        let hundreds = _mm_srli_epi16::<3>(_mm_mulhi_epu16(fourths, _mm_set1_epi16(5_243)));
        let remainders = _mm_sub_epi16(fourths, _mm_mullo_epi16(hundreds, _mm_set1_epi16(100)));
        let pairs = _mm_or_si128(hundreds, _mm_slli_epi32::<16>(remainders));
        // `t | (x − 10t) << 8` is `(x << 8) − t × (10 × 2^8 − 1)` for the tens `t` of `x`.
        let tens = _mm_mulhi_epu16(pairs, _mm_set1_epi16(6_554));
        let digits = _mm_sub_epi16(_mm_slli_epi16::<8>(pairs), _mm_mullo_epi16(tens, _mm_set1_epi16(2_559)));
        (
            _mm_cvtsi128_si64(digits).cast_unsigned(),
            _mm_cvtsi128_si64(_mm_unpackhi_epi64(digits, digits)).cast_unsigned(),
        )
    }
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
