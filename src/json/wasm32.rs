use core::arch::wasm32::*;

use super::{
    block::{self, Block, Carry, Classes},
    simd,
};
use crate::vector::wasm32::load16;

#[inline(never)]
pub(super) fn simd128(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
    let quarters = match bytes.first_chunk() {
        Some(bytes) => [
            load16::<0, 64>(bytes),
            load16::<16, 64>(bytes),
            load16::<32, 64>(bytes),
            load16::<48, 64>(bytes),
        ],
        None => {
            let words = block::tail(bytes);
            core::array::from_fn(|index| u64x2(words[2 * index], words[2 * index + 1]))
        }
    };
    let (whitespace, structural) = (load16::<0, 16>(&simd::WHITESPACE), load16::<0, 16>(&simd::STRUCTURAL));
    let nibble = u8x16_splat(0x0f);
    let control = bits(quarters.map(|quarter| u8x16_lt(quarter, u8x16_splat(0x20))));
    let curly = bits(quarters.map(|quarter| {
        let entry = u8x16_swizzle(structural, v128_and(quarter, nibble));
        u8x16_eq(entry, v128_or(quarter, u8x16_splat(0x20)))
    }));
    let classes = Classes {
        structural: curly & !control,
        whitespace: bits(
            quarters.map(|quarter| u8x16_eq(u8x16_swizzle(whitespace, v128_and(quarter, nibble)), quarter)),
        ),
        quote: bits(quarters.map(|quarter| u8x16_eq(quarter, u8x16_splat(b'"')))),
        backslash: bits(quarters.map(|quarter| u8x16_eq(quarter, u8x16_splat(b'\\')))),
        control,
        utf8_error: utf8_error(quarters, before),
    };
    simd::scan(classes, carry, block::prefix_xor)
}

#[inline]
fn bits(lanes: [v128; 4]) -> u64 {
    let [a, b, c, d] = lanes.map(|lane| u64::from(u8x16_bitmask(lane)));
    a | b << 16 | c << 32 | d << 48
}

#[inline]
fn utf8_error(quarters: [v128; 4], before: u32) -> bool {
    let [a, b, c, d] = quarters;
    if u8x16_bitmask(v128_or(v128_or(a, b), v128_or(c, d))) == 0 {
        return block::ends_incomplete(before);
    }
    let tables = [block::FIRST_HIGH, block::FIRST_LOW, block::SECOND_HIGH].map(|table| load16::<0, 16>(&table));
    let previous = u32x4(0, 0, 0, before);
    let error = v128_or(
        v128_or(check(a, previous, tables), check(b, a, tables)),
        v128_or(check(c, b, tables), check(d, c, tables)),
    );
    v128_any_true(error)
}

#[inline]
fn check(input: v128, previous: v128, [first_high, first_low, second_high]: [v128; 3]) -> v128 {
    let first = i8x16_shuffle::<13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28>(previous, input);
    let second = i8x16_shuffle::<14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29>(previous, input);
    let third = i8x16_shuffle::<15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(previous, input);
    let pair = v128_and(
        v128_and(
            u8x16_swizzle(first_high, u8x16_shr(third, 4)),
            u8x16_swizzle(first_low, v128_and(third, u8x16_splat(0x0f))),
        ),
        u8x16_swizzle(second_high, u8x16_shr(input, 4)),
    );
    let continues = v128_or(
        u8x16_sub_sat(second, u8x16_splat(0x60)),
        u8x16_sub_sat(first, u8x16_splat(0x70)),
    );
    v128_xor(v128_and(continues, u8x16_splat(0x80)), pair)
}

pub(super) const BITS: u32 = 1;

#[inline(always)]
pub(super) fn escapes(block: &[u8; 16]) -> u64 {
    scan(load16::<0, 16>(block))
}

#[inline(always)]
pub(super) fn escapes_of_words(low: u64, high: u64) -> u64 {
    scan(u64x2(low, high))
}

#[inline(always)]
fn scan(bytes: v128) -> u64 {
    let quote = u8x16_eq(bytes, u8x16_splat(b'"'));
    let backslash = u8x16_eq(bytes, u8x16_splat(b'\\'));
    let control = u8x16_lt(bytes, u8x16_splat(0x20));
    u64::from(u8x16_bitmask(v128_or(v128_or(quote, backslash), control)))
}
