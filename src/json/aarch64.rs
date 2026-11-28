use core::arch::aarch64::vmull_p64;

#[inline]
/// # Safety
/// The CPU must support NEON, AES.
#[target_feature(enable = "neon,aes")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the low 64 bits of the product are the result"
)]
fn pmull_prefix_xor(bits: u64) -> u64 {
    vmull_p64(bits, u64::MAX) as u64
}

#[cfg(target_feature = "neon")]
#[expect(
    unsafe_code,
    reason = "the build has Neon, the feature of each function that it calls"
)]
pub(super) mod escape {
    use core::arch::aarch64::{
        uint8x16_t, vceqq_u8, vcltq_u8, vcombine_u64, vcreate_u64, vdupq_n_u8, vget_lane_u64, vorrq_u8,
        vreinterpret_u64_u8, vreinterpretq_u8_u64, vreinterpretq_u16_u8, vshrn_n_u16,
    };

    use crate::vector::aarch64::load16;

    pub(in super::super) const BITS: u32 = 4;

    #[inline(always)]
    pub(in super::super) fn escapes(block: &[u8; 16]) -> u64 {
        // SAFETY: the build has Neon, the feature of `load16` and `scan`.
        unsafe { scan(load16::<0, 16>(block)) }
    }

    #[inline(always)]
    pub(in super::super) fn escapes_of_words(low: u64, high: u64) -> u64 {
        // SAFETY: the build has Neon, the feature of `scan`.
        unsafe { scan(vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(low), vcreate_u64(high)))) }
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    fn scan(bytes: uint8x16_t) -> u64 {
        let quote = vceqq_u8(bytes, vdupq_n_u8(b'"'));
        let backslash = vceqq_u8(bytes, vdupq_n_u8(b'\\'));
        let control = vcltq_u8(bytes, vdupq_n_u8(0x20));
        let hit = vorrq_u8(vorrq_u8(quote, backslash), control);
        // Four bits for each byte: the narrowing shift keeps the middle byte of each 16-bit lane.
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(hit))))
    }
}

#[cfg(target_feature = "neon")]
#[expect(
    unsafe_code,
    reason = "the build has Neon, the feature of each function that it calls"
)]
pub(super) mod digits {
    use core::arch::aarch64::{
        uint8x16_t, vcreate_u32, vdupq_n_u8, vget_lane_u64, vgetq_lane_u64, vmla_n_u32, vmlaq_n_u16, vmlaq_n_u32,
        vorrq_u8, vqdmulh_n_s32, vqdmulhq_n_s16, vqdmulhq_n_s32, vqtbl1q_u8, vreinterpret_s32_u32,
        vreinterpret_u16_u32, vreinterpret_u32_s32, vreinterpret_u64_u8, vreinterpretq_s16_u16, vreinterpretq_s32_u32,
        vreinterpretq_u8_u16, vreinterpretq_u16_s16, vreinterpretq_u16_u8, vreinterpretq_u16_u32,
        vreinterpretq_u32_s32, vreinterpretq_u64_u8, vrev64q_u8, vshll_n_u16, vshr_n_u32, vshrn_n_u16, vtstq_u8,
    };

    use crate::vector::aarch64::{load16, store16};

    static POINTS: [[u8; 16]; 16] = {
        let mut table = [[0; 16]; 16];
        let mut whole: u8 = 1;
        while whole <= 16 {
            let mut at: u8 = 0;
            while at < 16 {
                table[whole as usize - 1][at as usize] = if at < whole {
                    at
                } else if at == whole {
                    0xff
                } else {
                    at - 1
                };
                at += 1;
            }
            whole += 1;
        }
        table
    };

    #[inline(always)]
    pub(in super::super) fn sixteen(value: u64) -> (u64, u64) {
        // SAFETY: the build has Neon, the feature of `split` and of each intrinsic here.
        unsafe {
            let words = vreinterpretq_u64_u8(split(value));
            (vgetq_lane_u64::<0>(words), vgetq_lane_u64::<1>(words))
        }
    }

    #[inline(always)]
    pub(in super::super) fn mantissa(digits: u64, extra: u8, whole: usize, dest: &mut [u8]) -> usize {
        // SAFETY: the build has Neon, the feature of `text`, `load16`, `store`, and `vqtbl1q_u8`.
        let count = unsafe {
            let (text, count) = text(digits, extra);
            store(dest, 1, text);
            store(dest, 0, vqtbl1q_u8(text, load16::<0, 16>(&POINTS[whole - 1])));
            count
        };
        dest[whole] = b'.';
        dest[17] = b'0' + extra;
        count
    }

    #[inline(always)]
    pub(in super::super) fn plain(digits: u64, extra: u8, at: usize, dest: &mut [u8]) -> usize {
        // SAFETY: the build has Neon, the feature of `text` and `store`.
        let count = unsafe {
            let (text, count) = text(digits, extra);
            store(dest, at, text);
            count
        };
        dest[at + 16] = b'0' + extra;
        count
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    fn text(digits: u64, extra: u8) -> (uint8x16_t, usize) {
        let values = split(digits);
        let nonzero = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(vtstq_u8(
            values, values,
        )))));
        let count = 17 - ((nonzero >> 4 | u64::from(extra != 0) << 63).leading_zeros() / 4) as usize;
        (vorrq_u8(values, vdupq_n_u8(b'0')), count)
    }

    #[inline(always)]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    fn store(dest: &mut [u8], at: usize, bytes: uint8x16_t) {
        if let Some(dest) = dest.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<16>) {
            store16::<0, 16>(dest, bytes);
        }
    }

    /// Each step splits each lane into its quotient and its remainder by a power of 10, the
    /// remainder in the low half of the lane. A doubling multiply that keeps the high half gives
    /// each quotient: `x / 10^4` is `(2x × 109_951_163) >> 41` for `x < 10^8`, `x / 100` is
    /// `(2x × 21_474_837) >> 32` for `x < 10^4`, and `x / 10` is `(2x × 3_277) >> 16` for
    /// `x < 100`. Thus each half has its digits from the last, and a reverse of each half gives them
    /// from the first: the digit values of `value`, under 10^16, the first in lane 0.
    #[inline(always)]
    /// # Safety
    /// The CPU must support Neon.
    #[target_feature(enable = "neon")]
    fn split(value: u64) -> uint8x16_t {
        let (high, low) = (value / 100_000_000, value % 100_000_000);
        let halves = vcreate_u32(low << 32 | high);
        let fourths = vshr_n_u32::<9>(vreinterpret_u32_s32(vqdmulh_n_s32(
            vreinterpret_s32_u32(halves),
            109_951_163,
        )));
        let fourths = vmla_n_u32(halves, fourths, (1 << 16) - 10_000);
        let fourths = vshll_n_u16::<0>(vreinterpret_u16_u32(fourths));
        let pairs = vreinterpretq_u32_s32(vqdmulhq_n_s32(vreinterpretq_s32_u32(fourths), 21_474_837));
        let pairs = vreinterpretq_u16_u32(vmlaq_n_u32(fourths, pairs, (1 << 16) - 100));
        let tens = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(pairs), 3_277));
        vrev64q_u8(vreinterpretq_u8_u16(vmlaq_n_u16(pairs, tens, (1 << 8) - 10)))
    }
}

macro_rules! kernel {
    ($name:ident, $features:literal, $prefix_xor:expr) => {
        pub(super) mod $name {
            use core::arch::aarch64::*;

            use super::super::{
                block::{self, Block, Carry, Classes},
                simd,
            };
            use crate::vector::aarch64::{load16, table64};

            #[inline(never)]
            #[doc = concat!("# Safety\nThe CPU must support ", $features, ".")]
            #[target_feature(enable = $features)]
            pub(in super::super) fn block(input: &[u8], start: usize, carry: Carry) -> Block {
                let (bytes, before) = (&input[start..], block::before(input, start));
                simd::scan(classify(bytes, before), carry, $prefix_xor)
            }

            #[inline]
            #[doc = concat!("# Safety\nThe CPU must support ", $features, ".")]
            #[target_feature(enable = $features)]
            fn classify(bytes: &[u8], before: u32) -> Classes {
                let quarters = match bytes.first_chunk() {
                    Some(bytes) => {
                        let uint8x16x4_t(a, b, c, d) = table64::<0, 64>(bytes);
                        [a, b, c, d]
                    }
                    None => {
                        let words = block::tail(bytes);
                        core::array::from_fn(|index| {
                            let (low, high) = (vcreate_u64(words[2 * index]), vcreate_u64(words[2 * index + 1]));
                            vreinterpretq_u8_u64(vcombine_u64(low, high))
                        })
                    }
                };
                let (whitespace, structural) = (
                    load16::<0, 16>(&simd::WHITESPACE),
                    load16::<0, 16>(&simd::STRUCTURAL),
                );
                let nibble = vdupq_n_u8(0x0f);
                let control = bits(quarters.map(|quarter| vcltq_u8(quarter, vdupq_n_u8(0x20))));
                let curly = bits(quarters.map(|quarter| {
                    let entry = vqtbl1q_u8(structural, vandq_u8(quarter, nibble));
                    vceqq_u8(entry, vorrq_u8(quarter, vdupq_n_u8(0x20)))
                }));
                Classes {
                    structural: curly & !control,
                    whitespace: bits(
                        quarters.map(|quarter| vceqq_u8(vqtbl1q_u8(whitespace, vandq_u8(quarter, nibble)), quarter)),
                    ),
                    quote: bits(quarters.map(|quarter| vceqq_u8(quarter, vdupq_n_u8(b'"')))),
                    backslash: bits(quarters.map(|quarter| vceqq_u8(quarter, vdupq_n_u8(b'\\')))),
                    control,
                    utf8_error: utf8_error(quarters, before),
                }
            }

            #[inline]
            #[doc = concat!("# Safety\nThe CPU must support ", $features, ".")]
            #[target_feature(enable = $features)]
            fn bits(lanes: [uint8x16_t; 4]) -> u64 {
                const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
                let weights = load16::<0, 16>(&WEIGHTS);
                let [a, b, c, d] = lanes.map(|lane| vandq_u8(lane, weights));
                let sums = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
                vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(sums, sums)))
            }

            #[inline]
            #[doc = concat!("# Safety\nThe CPU must support ", $features, ".")]
            #[target_feature(enable = $features)]
            fn utf8_error(quarters: [uint8x16_t; 4], before: u32) -> bool {
                let [a, b, c, d] = quarters;
                if vmaxvq_u8(vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d))) < 0x80 {
                    return block::ends_incomplete(before);
                }
                let tables =
                    [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH].map(|table| load16::<0, 16>(&table));
                let previous = vreinterpretq_u8_u32(vsetq_lane_u32::<3>(before, vdupq_n_u32(0)));
                let error = vorrq_u8(
                    vorrq_u8(check(a, previous, tables), check(b, a, tables)),
                    vorrq_u8(check(c, b, tables), check(d, c, tables)),
                );
                vmaxvq_u8(error) != 0
            }

            #[inline]
            #[doc = concat!("# Safety\nThe CPU must support ", $features, ".")]
            #[target_feature(enable = $features)]
            fn check(
                input: uint8x16_t,
                previous: uint8x16_t,
                [first_high, first_low, second_high]: [uint8x16_t; 3],
            ) -> uint8x16_t {
                let (first, second, third) = (
                    vextq_u8::<13>(previous, input),
                    vextq_u8::<14>(previous, input),
                    vextq_u8::<15>(previous, input),
                );
                let pair = vandq_u8(
                    vandq_u8(
                        vqtbl1q_u8(first_high, vshrq_n_u8::<4>(third)),
                        vqtbl1q_u8(first_low, vandq_u8(third, vdupq_n_u8(0x0f))),
                    ),
                    vqtbl1q_u8(second_high, vshrq_n_u8::<4>(input)),
                );
                let continues = vorrq_u8(
                    vqsubq_u8(second, vdupq_n_u8(0x60)),
                    vqsubq_u8(first, vdupq_n_u8(0x70)),
                );
                veorq_u8(vandq_u8(continues, vdupq_n_u8(0x80)), pair)
            }
        }
    };
}

kernel!(pmull, "neon,aes", |bits| super::pmull_prefix_xor(bits));

kernel!(shift, "neon", block::prefix_xor);
