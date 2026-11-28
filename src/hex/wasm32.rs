use core::arch::wasm32::*;

use super::encode_steps;
use crate::{
    ct::Class,
    simd::{overlapping, short_blocks},
    vector::wasm32::{load16, store_low, store16},
};

const DIGITS: [u8; 16] = *b"0123456789abcdef";

#[inline]
fn nibbles(chars: v128) -> (v128, v128) {
    let value = i8x16_sub(chars, u8x16_splat(b'0'));
    let digit = u8x16_lt(value, u8x16_splat(10));
    let letter = u8x16_lt(i8x16_sub(chars, u8x16_splat(b'a')), u8x16_splat(6));
    (
        i8x16_sub(value, v128_and(letter, u8x16_splat(39))),
        v128_or(digit, letter),
    )
}

#[inline]
fn join(nibbles: v128) -> v128 {
    v128_or(i16x8_shl(nibbles, 4), u16x8_shr(nibbles, 8))
}

#[inline]
fn evens(first: v128, second: v128) -> v128 {
    u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(first, second)
}

pub(super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
    let digits = load16::<0, _>(&DIGITS);
    let block = |src: &[u8; 16], dst: &mut [u8; 32]| {
        let src = load16::<0, _>(src);
        let hi = i8x16_swizzle(digits, u8x16_shr(src, 4));
        let lo = i8x16_swizzle(digits, v128_and(src, u8x16_splat(0x0f)));
        store16::<0, _>(
            dst,
            u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(hi, lo),
        );
        store16::<16, _>(
            dst,
            u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(hi, lo),
        );
    };
    encode_steps(class, bytes, out, block, |bytes, out| overlapping(bytes, out, block));
}

pub(super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
    let mut valid = u8x16_splat(0xff);
    let mut block = |src: &[u8; 32], dst: &mut [u8; 16]| {
        let (first, first_valid) = nibbles(load16::<0, _>(src));
        let (second, second_valid) = nibbles(load16::<16, _>(src));
        valid = v128_and(valid, v128_and(first_valid, second_valid));
        store16::<0, _>(dst, evens(join(first), join(second)));
    };
    if text.len() >= 32 {
        overlapping(text, out, block);
    } else if text.len() >= 16 {
        overlapping::<16, 8>(text, out, |src, dst| {
            let (chars, chars_valid) = nibbles(load16::<0, _>(src));
            valid = v128_and(valid, chars_valid);
            store_low(dst, evens(join(chars), join(chars)));
        });
    } else {
        short_blocks(class, text, out, 32, b'0', |len| len / 2, &mut block);
    }
    u8x16_all_true(valid)
}
