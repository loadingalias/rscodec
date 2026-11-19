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
            #[inline(always)]
            #[target_feature(enable = $features)]
            pub(in super::super) fn block(bytes: &[u8], before: u32, carry: Carry) -> Block {
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
