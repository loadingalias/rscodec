//! The Neon kernels. Nibble tables classify the bytes and check UTF-8. A carry-less multiply
//! (PMULL) builds the string mask where the CPU has it, and six shifts where it does not.
//!
//! The macro `kernel` makes the module of each kernel from one source, so each kernel has its own
//! copy of each helper, with the target features of the kernel. The compiler does not inline a
//! helper of this size that has two callers, and a call for each block costs time.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::aarch64::vmull_p64;

/// The XOR of bits `0..=i` of `bits` in each bit `i`: the carry-less product with all ones.
#[inline]
#[target_feature(enable = "neon,aes")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the low 64 bits of the product are the result"
)]
fn pmull_prefix_xor(bits: u64) -> u64 {
    vmull_p64(bits, u64::MAX) as u64
}

/// The scan of the writer for the bytes that a JSON string escapes. Neon is in the base of each
/// aarch64 target that has it, so the build fixes the kernel.
#[cfg(target_feature = "neon")]
#[expect(
    unsafe_code,
    reason = "the build has Neon, and the load reads only the bytes of the array that it takes"
)]
pub(super) mod escape {
    use core::arch::aarch64::{
        uint8x16_t, vceqq_u8, vcltq_u8, vcombine_u64, vcreate_u64, vdupq_n_u8, vget_lane_u64, vld1q_u8, vorrq_u8,
        vreinterpret_u64_u8, vreinterpretq_u8_u64, vreinterpretq_u16_u8, vshrn_n_u16,
    };

    /// The bits of a mask for each byte of [`escapes`].
    pub(in super::super) const BITS: u32 = 4;

    /// A mask with [`BITS`] bits for each byte of `block` that a JSON string escapes: `"`, `\\`,
    /// or a byte under 0x20, the first byte in the low bits.
    #[inline(always)]
    pub(in super::super) fn escapes(block: &[u8; 16]) -> u64 {
        // SAFETY: the build has Neon, the feature of `load` and `scan`.
        unsafe { scan(load(block)) }
    }

    /// [`escapes`] of the 16 bytes of the little-endian words `low` and `high`, with no load.
    #[inline(always)]
    pub(in super::super) fn escapes_of_words(low: u64, high: u64) -> u64 {
        // SAFETY: the build has Neon, the feature of `scan`.
        unsafe { scan(vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(low), vcreate_u64(high)))) }
    }

    #[inline(always)]
    #[target_feature(enable = "neon")]
    fn load(block: &[u8; 16]) -> uint8x16_t {
        // SAFETY: the load reads the 16 bytes of `block`, and it needs no alignment.
        unsafe { vld1q_u8(block.as_ptr()) }
    }

    #[inline(always)]
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

/// The digits of the printers. Neon is in the base of each aarch64 target that has it, so the build
/// fixes the kernel.
#[cfg(target_feature = "neon")]
#[expect(
    unsafe_code,
    reason = "the build has Neon, and each load and store touches only the bytes of the array that it takes"
)]
pub(super) mod digits {
    use core::arch::aarch64::{
        uint8x16_t, vcreate_u32, vdupq_n_u8, vget_lane_u64, vgetq_lane_u64, vld1q_u8, vmla_n_u32, vmlaq_n_u16,
        vmlaq_n_u32, vorrq_u8, vqdmulh_n_s32, vqdmulhq_n_s16, vqdmulhq_n_s32, vqtbl1q_u8, vreinterpret_s32_u32,
        vreinterpret_u16_u32, vreinterpret_u32_s32, vreinterpret_u64_u8, vreinterpretq_s16_u16, vreinterpretq_s32_u32,
        vreinterpretq_u8_u16, vreinterpretq_u16_s16, vreinterpretq_u16_u8, vreinterpretq_u16_u32,
        vreinterpretq_u32_s32, vreinterpretq_u64_u8, vrev64q_u8, vshll_n_u16, vshr_n_u32, vshrn_n_u16, vst1q_u8,
        vtstq_u8,
    };

    /// For each `whole` from 1 to 16, at index `whole − 1`, the bytes of the lookup of [`mantissa`]:
    /// each digit before the point, a 0 for the point, and each other digit one place higher.
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

    /// [`super::super::integer::eight_digits`] of the two halves of `value`, under 10^16, high
    /// first.
    #[inline(always)]
    pub(in super::super) fn sixteen(value: u64) -> (u64, u64) {
        // SAFETY: the build has Neon, the feature of `split` and of each intrinsic here.
        unsafe {
            let words = vreinterpretq_u64_u8(split(value));
            (vgetq_lane_u64::<0>(words), vgetq_lane_u64::<1>(words))
        }
    }

    /// [`super::super::float::mantissa`]. The digits stay in their register: one lookup puts the
    /// point in, and two stores write them, so no digit goes through a general register.
    #[inline(always)]
    pub(in super::super) fn mantissa(digits: u64, extra: u8, whole: usize, dest: &mut [u8]) -> usize {
        // SAFETY: the build has Neon, the feature of `text`, `load`, `store`, and `vqtbl1q_u8`.
        let count = unsafe {
            let (text, count) = text(digits, extra);
            store(dest, 1, text);
            store(dest, 0, vqtbl1q_u8(text, load(&POINTS[whole - 1])));
            count
        };
        // The lookup writes 0 in the place of the point, and it cannot reach the 17th byte, which
        // takes the point when `whole` is 16.
        dest[whole] = b'.';
        dest[17] = b'0' + extra;
        count
    }

    /// [`super::super::float::plain`].
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

    /// The 16 digits of `digits`, from 10^15 up and under 10^16, as text, the first in lane 0,
    /// and their count without the zeros after the last one, or 17 when `extra` is not 0.
    #[inline(always)]
    #[target_feature(enable = "neon")]
    fn text(digits: u64, extra: u8) -> (uint8x16_t, usize) {
        let values = split(digits);
        // Four bits for each digit that is not 0: the narrowing shift keeps the middle byte of each
        // 16-bit lane.
        let nonzero = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(vtstq_u8(
            values, values,
        )))));
        // The count with no branch, because it varies from value to value: a 17th digit sets the
        // top bit, and without one, the bits of the first digit go and each other digit is one
        // digit lower, which a first digit alone leaves 0.
        let count = 17 - ((nonzero >> 4 | u64::from(extra != 0) << 63).leading_zeros() / 4) as usize;
        (vorrq_u8(values, vdupq_n_u8(b'0')), count)
    }

    #[inline(always)]
    #[target_feature(enable = "neon")]
    fn load(bytes: &[u8; 16]) -> uint8x16_t {
        // SAFETY: the load reads the 16 bytes of `bytes`, and it needs no alignment.
        unsafe { vld1q_u8(bytes.as_ptr()) }
    }

    /// Stores `bytes` into `dest` from `at`. The offsets of the callers leave 16 bytes, so the
    /// store always runs.
    #[inline(always)]
    #[target_feature(enable = "neon")]
    fn store(dest: &mut [u8], at: usize, bytes: uint8x16_t) {
        if let Some(dest) = dest.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<16>) {
            // SAFETY: the store writes the 16 bytes of `dest`, and it needs no alignment.
            unsafe { vst1q_u8(dest.as_mut_ptr(), bytes) }
        }
    }

    /// Each step splits each lane into its quotient and its remainder by a power of 10, the
    /// remainder in the low half of the lane. A doubling multiply that keeps the high half gives
    /// each quotient: `x / 10^4` is `(2x × 109_951_163) >> 41` for `x < 10^8`, `x / 100` is
    /// `(2x × 21_474_837) >> 32` for `x < 10^4`, and `x / 10` is `(2x × 3_277) >> 16` for
    /// `x < 100`. Thus each half has its digits from the last, and a reverse of each half gives them
    /// from the first: the digit values of `value`, under 10^16, the first in lane 0.
    #[inline(always)]
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

/// Makes the module `$name` of a kernel with the target features `$features`. `$prefix_xor` gives
/// the XOR of bits `0..=i` of its argument in each bit `i`.
macro_rules! kernel {
    ($(#[$doc:meta])* $name:ident, $features:literal, $prefix_xor:expr) => {
        $(#[$doc])*
        pub(super) mod $name {
            use core::arch::aarch64::*;

            use super::super::{
                block::{self, Block, Carry, Classes},
                simd,
            };

            /// The kernel.
            #[inline(never)]
            #[target_feature(enable = $features)]
            pub(in super::super) fn block(input: &[u8], start: usize, carry: Carry) -> Block {
                let (bytes, before) = (&input[start..], block::before(input, start));
                simd::scan(classify(bytes, before), carry, $prefix_xor)
            }

            #[inline]
            #[target_feature(enable = $features)]
            fn classify(bytes: &[u8], before: u32) -> Classes {
                let quarters = match bytes.first_chunk() {
                    Some(bytes) => load::quarters(bytes),
                    None => {
                        let words = block::tail(bytes);
                        core::array::from_fn(|index| {
                            let (low, high) = (vcreate_u64(words[2 * index]), vcreate_u64(words[2 * index + 1]));
                            vreinterpretq_u8_u64(vcombine_u64(low, high))
                        })
                    }
                };
                let (whitespace, structural) = (load::table(&simd::WHITESPACE), load::table(&simd::STRUCTURAL));
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

            /// The bit of each byte of `lanes` that is all ones. Each byte keeps the bit of its
            /// place in a group of 8, and three pairwise adds sum each group into one byte.
            #[inline]
            #[target_feature(enable = $features)]
            fn bits(lanes: [uint8x16_t; 4]) -> u64 {
                const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
                let weights = load::table(&WEIGHTS);
                let [a, b, c, d] = lanes.map(|lane| vandq_u8(lane, weights));
                let sums = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
                vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(sums, sums)))
            }

            /// The UTF-8 check of [`block::utf8_error_at`] on each byte of `quarters`.
            #[inline]
            #[target_feature(enable = $features)]
            fn utf8_error(quarters: [uint8x16_t; 4], before: u32) -> bool {
                let [a, b, c, d] = quarters;
                if vmaxvq_u8(vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d))) < 0x80 {
                    return block::ends_incomplete(before);
                }
                let tables = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH].map(|table| load::table(&table));
                let previous = vreinterpretq_u8_u32(vsetq_lane_u32::<3>(before, vdupq_n_u32(0)));
                let error = vorrq_u8(
                    vorrq_u8(check(a, previous, tables), check(b, a, tables)),
                    vorrq_u8(check(c, b, tables), check(d, c, tables)),
                );
                vmaxvq_u8(error) != 0
            }

            /// A byte that is not zero at each byte of `input` that ends an error, after the bytes
            /// of `previous`.
            #[inline]
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
                // Bit 7 is set where `second` is at least 0xe0 or `first` is at least 0xf0.
                let continues = vorrq_u8(vqsubq_u8(second, vdupq_n_u8(0x60)), vqsubq_u8(first, vdupq_n_u8(0x70)));
                veorq_u8(vandq_u8(continues, vdupq_n_u8(0x80)), pair)
            }

            /// The loads of the kernel.
            #[expect(unsafe_code, reason = "each load reads only the bytes of the array that it takes")]
            mod load {
                use core::arch::aarch64::{uint8x16_t, uint8x16x4_t, vld1q_u8, vld1q_u8_x4};

                /// The 16 bytes of `table`.
                #[inline]
                #[target_feature(enable = $features)]
                pub(super) fn table(table: &[u8; 16]) -> uint8x16_t {
                    // SAFETY: the load reads the 16 bytes of `table`, and it needs no alignment.
                    unsafe { vld1q_u8(table.as_ptr()) }
                }

                /// The four quarters of `bytes`.
                #[inline]
                #[target_feature(enable = $features)]
                pub(super) fn quarters(bytes: &[u8; 64]) -> [uint8x16_t; 4] {
                    // SAFETY: the load reads the 64 bytes of `bytes`, and it needs no alignment.
                    let uint8x16x4_t(a, b, c, d) = unsafe { vld1q_u8_x4(bytes.as_ptr()) };
                    [a, b, c, d]
                }
            }
        }
    };
}

kernel!(
    /// The Neon kernel with PMULL.
    pmull,
    "neon,aes",
    |bits| super::pmull_prefix_xor(bits)
);

kernel!(
    /// The Neon kernel without PMULL.
    shift,
    "neon",
    block::prefix_xor
);
