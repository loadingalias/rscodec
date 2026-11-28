use core::arch::wasm32::*;

use super::{Profile, Tables, decode_steps, encode_steps};
use crate::{
    ct::Class,
    simd::{byte_at, from_end, short_blocks, to_end},
    vector::wasm32::{load16, store16},
};

const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xff, 0xff, 0xff, 0xff];

pub(super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let shift = load16::<0, _>(&Tables::of(profile).nibbles.shift);
    let order = load16::<0, _>(&ENCODE_ORDER);
    let block = |src: &[u8; 16], skip: u8, dst: &mut [u8; 16]| {
        // `swizzle` gives zero to a lane whose index is past the vector.
        let input = i8x16_swizzle(load16::<0, _>(src), i8x16_add(order, u8x16_splat(skip)));
        let first = u16x8_shr(v128_and(input, u32x4_splat(0x0000_fc00)), 10);
        let third = u16x8_shr(v128_and(input, u32x4_splat(0x0fc0_0000)), 6);
        let second_fourth = i16x8_mul(v128_and(input, u32x4_splat(0x003f_03f0)), u32x4_splat(0x0100_0010));
        let sextets = v128_or(v128_or(first, third), second_fourth);
        let class = v128_or(
            u8x16_sub_sat(sextets, u8x16_splat(51)),
            v128_and(i8x16_gt(i8x16_splat(26), sextets), u8x16_splat(13)),
        );
        store16::<0, _>(dst, i8x16_add(sextets, i8x16_swizzle(shift, class)));
    };
    match bytes.as_rchunks().1 {
        [.., last] => encode_steps(class, bytes, last, out, to_end, block),
        [] => {
            short_blocks::<16, 16>(
                class,
                bytes,
                out,
                12,
                0,
                |step| step / 3 * 4,
                |src, dst| block(src, 0, dst),
            );
        }
    }
}

pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let tables = &Tables::of(profile).nibbles;
    let (lo_classes, hi_classes) = (load16::<0, _>(&tables.lo), load16::<0, _>(&tables.hi));
    let roll = load16::<0, _>(&tables.roll);
    let special = u8x16_splat(tables.special);
    let special_index = u8x16_splat(tables.special_index);
    let order = load16::<0, _>(&DECODE_ORDER);
    let mut invalid = u8x16_splat(0);
    let trailing = decode_steps(class, text, out, (from_end, byte_at), |src, dst| {
        let src = load16::<0, _>(src);
        let hi = u8x16_shr(src, 4);
        let lo = v128_and(src, u8x16_splat(0x0f));
        invalid = v128_or(
            invalid,
            v128_and(i8x16_swizzle(lo_classes, lo), i8x16_swizzle(hi_classes, hi)),
        );
        let index = i8x16_add(hi, v128_and(i8x16_eq(src, special), special_index));
        let sextets = i8x16_add(src, i8x16_swizzle(roll, index));
        let pairs = v128_or(
            v128_and(i16x8_shl(sextets, 6), u16x8_splat(0x0fc0)),
            u16x8_shr(sextets, 8),
        );
        let words = i32x4_dot_i16x8(pairs, u32x4_splat(0x0001_1000));
        store16::<0, _>(dst, i8x16_swizzle(words, order));
    });
    !v128_any_true(invalid) & (trailing == 0)
}
