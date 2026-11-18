//! The Neon kernels: `tbl` maps each nibble to its digit, `st2` joins the two characters of each
//! byte, and `uzp1` and `uzp2` split them. The decoder ANDs the valid lanes of each block into one
//! vector, and tests it once after the last block.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::aarch64::*;

use crate::{
    ct::Class,
    simd::{
        aarch64::{load16, store_pairs, store16},
        blocks,
    },
};

/// The digit of each nibble.
const DIGITS: [u8; 16] = *b"0123456789abcdef";

/// The nibbles of 16 characters, and the lanes that are digits, which are all ones.
#[inline]
#[target_feature(enable = "neon")]
fn nibbles(chars: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    let value = vsubq_u8(chars, vdupq_n_u8(b'0'));
    let digit = vcltq_u8(value, vdupq_n_u8(10));
    let letter = vcltq_u8(vsubq_u8(chars, vdupq_n_u8(b'a')), vdupq_n_u8(6));
    // The step from `c - '0'` to the nibble of a letter.
    (
        vsubq_u8(value, vandq_u8(letter, vdupq_n_u8(39))),
        vorrq_u8(digit, letter),
    )
}

#[target_feature(enable = "neon")]
pub(super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
    let digits = load16::<0, _>(&DIGITS);
    blocks::<16, 32>(
        class,
        bytes,
        out,
        16,
        0,
        |len| len * 2,
        |src, dst| {
            let src = load16::<0, _>(src);
            let hi = vqtbl1q_u8(digits, vshrq_n_u8::<4>(src));
            let lo = vqtbl1q_u8(digits, vandq_u8(src, vdupq_n_u8(0x0f)));
            store_pairs(dst, uint8x16x2_t(hi, lo));
        },
    );
}

#[target_feature(enable = "neon")]
pub(super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
    let mut valid = vdupq_n_u8(0xff);
    blocks::<32, 16>(
        class,
        text,
        out,
        32,
        b'0',
        |len| len / 2,
        |src, dst| {
            let (first, second) = (load16::<0, _>(src), load16::<16, _>(src));
            let (hi, lo) = (vuzp1q_u8(first, second), vuzp2q_u8(first, second));
            let ((hi, hi_valid), (lo, lo_valid)) = (nibbles(hi), nibbles(lo));
            valid = vandq_u8(valid, vandq_u8(hi_valid, lo_valid));
            store16::<0, _>(dst, vorrq_u8(vshlq_n_u8::<4>(hi), lo));
        },
    );
    vminvq_u8(valid) == 0xff
}
