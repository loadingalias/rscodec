//! The WebAssembly SIMD kernels: the method of the SSSE3 kernels, with `i8x16.swizzle` for
//! `pshufb`, shifts for the multiplies of the encoder, and `i32x4.dot_i16x8_s` for `pmaddwd`.
//! WebAssembly has no timing contract, so the kernels keep only the rule of the code: no branch
//! and no table load on the data.

use core::arch::wasm32::*;

use super::{Profile, Tables, bytes_len, symbols_len};
use crate::{
    ct::Class,
    simd::{
        blocks,
        wasm32::{load16, store16},
    },
};

/// The bytes `b1 b0 b2 b1` of each three input bytes, in each 32-bit lane, for the encoder.
const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

/// The three bytes of each 32-bit lane after the decoder packs it, first byte first.
const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xff, 0xff, 0xff, 0xff];

pub(super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let shift = load16::<0, _>(&Tables::of(profile).nibbles.shift);
    let order = load16::<0, _>(&ENCODE_ORDER);
    blocks::<16, 16>(class, bytes, out, 12, 0, symbols_len, |src, dst| {
        let input = i8x16_swizzle(load16::<0, _>(src), order);
        // In each 32-bit lane, the low 16 bits hold the first and the second sextet at bits 10
        // and 4, and the high 16 bits hold the third and the fourth at bits 6 and 0.
        let first = u16x8_shr(v128_and(input, u32x4_splat(0x0000_fc00)), 10);
        let third = u16x8_shr(v128_and(input, u32x4_splat(0x0fc0_0000)), 6);
        let second_fourth = i16x8_mul(v128_and(input, u32x4_splat(0x003f_03f0)), u32x4_splat(0x0100_0010));
        let sextets = v128_or(v128_or(first, third), second_fourth);
        let class = v128_or(
            u8x16_sub_sat(sextets, u8x16_splat(51)),
            v128_and(i8x16_gt(i8x16_splat(26), sextets), u8x16_splat(13)),
        );
        store16::<0, _>(dst, i8x16_add(sextets, i8x16_swizzle(shift, class)));
    });
}

pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let tables = &Tables::of(profile).nibbles;
    let (lo_classes, hi_classes) = (load16::<0, _>(&tables.lo), load16::<0, _>(&tables.hi));
    let roll = load16::<0, _>(&tables.roll);
    let special = u8x16_splat(tables.special);
    let special_index = u8x16_splat(tables.special_index);
    let order = load16::<0, _>(&DECODE_ORDER);
    let mut invalid = u8x16_splat(0);
    blocks::<16, 16>(class, text, out, 16, b'A', bytes_len, |src, dst| {
        let src = load16::<0, _>(src);
        let hi = u8x16_shr(src, 4);
        let lo = v128_and(src, u8x16_splat(0x0f));
        invalid = v128_or(
            invalid,
            v128_and(i8x16_swizzle(lo_classes, lo), i8x16_swizzle(hi_classes, hi)),
        );
        let index = i8x16_add(hi, v128_and(i8x16_eq(src, special), special_index));
        let sextets = i8x16_add(src, i8x16_swizzle(roll, index));
        // Each 16-bit lane: `a << 6 | b`. Each 32-bit lane: the 24 bits of the four sextets.
        let pairs = v128_or(
            v128_and(i16x8_shl(sextets, 6), u16x8_splat(0x0fc0)),
            u16x8_shr(sextets, 8),
        );
        let words = i32x4_dot_i16x8(pairs, u32x4_splat(0x0001_1000));
        store16::<0, _>(dst, i8x16_swizzle(words, order));
    });
    !v128_any_true(invalid)
}
