//! The Neon kernels: `tbl` maps each nibble to its digit, and `zip1` and `zip2` put the two
//! characters of each byte in order. The decoder splits the characters with `uzp1` and `uzp2`, and
//! looks up the flag of each in a table of the 64 bytes from `0` (`tbl`): the nibble with bit 7
//! set, or zero for a byte that is not a digit. It ANDs the flags of each block into one vector,
//! and tests bit 7 once after the last block. An input of up to `SHORT_MAX` bytes runs inline in
//! the caller, and a longer one runs in a function of its own.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::aarch64::*;

use super::encode_steps;
use crate::{
    ct::Class,
    simd::{SHORT_MAX, overlapping, short_blocks},
    vector::aarch64::{load16, store16, table64},
};

/// The digit of each nibble.
const DIGITS: [u8; 16] = *b"0123456789abcdef";

/// The flag of each byte from `0`: the nibble with bit 7 set for a digit, and zero for another
/// byte.
const FLAGS: [u8; 64] = {
    let mut flags = [0; 64];
    let mut nibble: u8 = 0;
    while nibble < 16 {
        flags[(DIGITS[nibble as usize] - b'0') as usize] = 0x80 | nibble;
        nibble += 1;
    }
    flags
};

/// Encodes `bytes` into `out`. A short input runs inline in the caller.
#[inline(always)]
#[target_feature(enable = "neon")]
pub(super) fn encode(class: impl Class, bytes: &[u8], out: &mut [u8]) {
    let digits = load16::<0, _>(&DIGITS);
    encode_steps(
        class,
        bytes,
        out,
        |src, dst| encode_block(digits, src, dst),
        |bytes, out| encode_long(digits, bytes, out),
    );
}

/// Encodes a long input in blocks of 64 bytes.
#[inline(never)]
#[target_feature(enable = "neon")]
fn encode_long(digits: uint8x16_t, bytes: &[u8], out: &mut [u8]) {
    overlapping::<64, 128>(bytes, out, |src, dst| {
        for (src, dst) in src.as_chunks::<16>().0.iter().zip(dst.as_chunks_mut::<32>().0) {
            encode_block(digits, src, dst);
        }
    });
}

/// The 32 characters of 16 bytes.
#[inline]
#[target_feature(enable = "neon")]
fn encode_block(digits: uint8x16_t, src: &[u8; 16], dst: &mut [u8; 32]) {
    let src = load16::<0, _>(src);
    let hi = vqtbl1q_u8(digits, vshrq_n_u8::<4>(src));
    let lo = vqtbl1q_u8(digits, vandq_u8(src, vdupq_n_u8(0x0f)));
    store16::<0, _>(dst, vzip1q_u8(hi, lo));
    store16::<16, _>(dst, vzip2q_u8(hi, lo));
}

/// Decodes `text` into `out`, and gives whether each character is a digit. A short input runs
/// inline in the caller.
#[inline(always)]
#[target_feature(enable = "neon")]
pub(super) fn decode(class: impl Class, text: &[u8], out: &mut [u8]) -> bool {
    let flags = table64::<0, _>(&FLAGS);
    let mut valid = vdupq_n_u8(0xff);
    if out.len() > SHORT_MAX {
        valid = decode_long(flags, text, out);
    } else if text.len() >= 32 {
        overlapping::<32, 16>(text, out, |src, dst| {
            valid = vandq_u8(valid, decode_block(flags, src, dst));
        });
    } else if text.len() >= 16 {
        overlapping::<16, 8>(text, out, |src, dst| {
            let chars = load16::<0, _>(src);
            let (hi, lo) = (
                flag(flags, vuzp1q_u8(chars, chars)),
                flag(flags, vuzp2q_u8(chars, chars)),
            );
            valid = vandq_u8(valid, vandq_u8(hi, lo));
            let mut bytes = [0; 16];
            store16::<0, _>(&mut bytes, vsliq_n_u8::<4>(lo, hi));
            if let Some(low) = bytes.first_chunk() {
                *dst = *low;
            }
        });
    } else {
        short_blocks::<32, 16>(
            class,
            text,
            out,
            32,
            b'0',
            |len| len / 2,
            |src, dst| {
                valid = vandq_u8(valid, decode_block(flags, src, dst));
            },
        );
    }
    vminvq_u8(valid) >= 0x80
}

/// Decodes a long input in blocks of 128 characters, and gives the AND of the flags.
#[inline(never)]
#[target_feature(enable = "neon")]
fn decode_long(flags: uint8x16x4_t, text: &[u8], out: &mut [u8]) -> uint8x16_t {
    let mut valid = vdupq_n_u8(0xff);
    overlapping::<128, 64>(text, out, |src, dst| {
        for (src, dst) in src.as_chunks::<32>().0.iter().zip(dst.as_chunks_mut::<16>().0) {
            valid = vandq_u8(valid, decode_block(flags, src, dst));
        }
    });
    valid
}

/// The flag of each of 16 characters.
#[inline]
#[target_feature(enable = "neon")]
fn flag(flags: uint8x16x4_t, chars: uint8x16_t) -> uint8x16_t {
    vqtbl4q_u8(flags, vsubq_u8(chars, vdupq_n_u8(b'0')))
}

/// The 16 bytes of 32 characters, and the AND of their flags. `sli` keeps only the low nibble of
/// the second flag of each byte, so bit 7 leaves no trace in the bytes.
#[inline]
#[target_feature(enable = "neon")]
fn decode_block(flags: uint8x16x4_t, src: &[u8; 32], dst: &mut [u8; 16]) -> uint8x16_t {
    let (first, second) = (load16::<0, _>(src), load16::<16, _>(src));
    let (hi, lo) = (
        flag(flags, vuzp1q_u8(first, second)),
        flag(flags, vuzp2q_u8(first, second)),
    );
    store16::<0, _>(dst, vsliq_n_u8::<4>(lo, hi));
    vandq_u8(hi, lo)
}
