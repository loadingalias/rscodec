//! The Neon kernels. An input of up to `SHORT_MAX` bytes runs inline in the caller, in blocks of 16
//! characters. A longer input uses the interleaved loads (`ld3`, `ld4`) for whole blocks of 64
//! characters, and blocks of 16 characters for the rest. The encoder runs inline at each length,
//! and the decoder runs a longer input in a function of its own. The encoder looks
//! up each sextet in the alphabet (`tbl`), and `zip1` and `zip2` put the characters in order. The
//! decoder looks up the flag of each character in a table of the 80 bytes from `+` (`tbl` and
//! `tbx`): the sextet with bit 7 set, or zero for a byte outside the alphabet. It ANDs the flags of
//! each block into one vector, and tests bit 7 once after the last block.
//!
//! The kernels follow the pipes of Neoverse V2: `st4` and `st2` put most of their work on pipes V0
//! and V1, where `tbl` runs too, and `zip1`, `zip2`, and `st1` do not.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::aarch64::*;

use super::{Profile, Tables, decode_steps, encode_steps};
use crate::{
    ct::Class,
    simd::{SHORT_MAX, byte_at, from_end, short_blocks, to_end},
    vector::aarch64::{all_valid, load_quads, load_triples, load16, store_triples, store16, table64},
};

/// The bytes `b1 b0 b2 b1` of each three input bytes, in each 32-bit lane, for the encoder.
const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

/// The three bytes of each 32-bit lane after the decoder packs it, first byte first.
const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xff, 0xff, 0xff, 0xff];

/// The flag table of the decoder in vectors: its first 64 bytes, and its last 16.
struct Lookups {
    low: uint8x16x4_t,
    high: uint8x16_t,
}

#[inline]
#[target_feature(enable = "neon")]
fn lookups(tables: &Tables) -> Lookups {
    Lookups {
        low: table64::<0, _>(&tables.sextets_from_plus),
        high: load16::<64, _>(&tables.sextets_from_plus),
    }
}

/// The flags of 16 characters: the sextet with bit 7 set for a byte of the alphabet, and zero for
/// another byte. The table holds them. A byte below `+` or past the table has an index past both
/// lookups, which give zero.
#[inline]
#[target_feature(enable = "neon")]
fn flags(lookups: &Lookups, chars: uint8x16_t) -> uint8x16_t {
    let index = vsubq_u8(chars, vdupq_n_u8(b'+'));
    // `tbx` keeps the result of the first lookup in each lane whose index is past its table.
    vqtbx1q_u8(
        vqtbl4q_u8(lookups.low, index),
        lookups.high,
        vsubq_u8(index, vdupq_n_u8(64)),
    )
}

/// Encodes `bytes` into the characters `out`, with no padding, inline in the caller at each length.
///
/// A call in the encoder would put the constants of a loop of the caller in memory: a call keeps
/// only the low 64 bits of `v8` to `v15`. With a call for an input longer than `SHORT_MAX` or
/// shorter than one vector, base64url of 16 bytes took 1.14 times as long as base64-simd on
/// Graviton4, and inline 0.97. The code at each call site is about three times as large, and an
/// encode of 4 KiB or more took 1.00 to 1.14 times as long (C109).
#[inline(always)]
#[target_feature(enable = "neon")]
pub(super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    if bytes.len() <= SHORT_MAX {
        encode_12(class, profile, bytes, out);
    } else {
        encode_long(class, profile, bytes, out);
    }
}

/// Encodes a long input: two blocks of 48 bytes in each step, then one, with the interleaved
/// loads, then the rest, at least 16 bytes, in blocks of 12 bytes.
#[inline(always)]
#[target_feature(enable = "neon")]
fn encode_long(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let alphabet = table64::<0, _>(&Tables::of(profile).alphabet);
    let blocks = bytes.len().saturating_sub(16) / 48;
    let (head, rest) = bytes.split_at(48 * blocks);
    let (head_out, rest_out) = out.split_at_mut(64 * blocks);
    let (pairs, single) = head.as_chunks::<96>();
    let (pair_outs, single_out) = head_out.as_chunks_mut::<128>();
    for (src, dst) in pairs.iter().zip(pair_outs) {
        for (src, dst) in src.as_chunks::<48>().0.iter().zip(dst.as_chunks_mut::<64>().0) {
            encode_48(alphabet, src, dst);
        }
    }
    if let (Some(src), Some(dst)) = (single.first_chunk(), single_out.first_chunk_mut()) {
        encode_48(alphabet, src, dst);
    }
    encode_12(class, profile, rest, rest_out);
}

/// The 64 characters of 48 bytes. `zip1` and `zip2` interleave the four vectors of characters,
/// which `st4` would do on the pipes of `tbl`.
#[inline]
#[target_feature(enable = "neon")]
fn encode_48(alphabet: uint8x16x4_t, src: &[u8; 48], dst: &mut [u8; 64]) {
    let sextet = vdupq_n_u8(63);
    let uint8x16x3_t(a, b, c) = load_triples(src);
    let chars = |sextets| vqtbl4q_u8(alphabet, sextets);
    let first = chars(vshrq_n_u8::<2>(a));
    let second = chars(vandq_u8(vsriq_n_u8::<4>(vshlq_n_u8::<4>(a), b), sextet));
    let third = chars(vandq_u8(vsriq_n_u8::<6>(vshlq_n_u8::<2>(b), c), sextet));
    let fourth = chars(vandq_u8(c, sextet));
    // The first and second characters of each group in 16-bit lanes, and the third and fourth.
    let (low_pairs, high_pairs) = (vzip1q_u8(first, second), vzip2q_u8(first, second));
    let (low_ends, high_ends) = (vzip1q_u8(third, fourth), vzip2q_u8(third, fourth));
    let groups = |pairs: uint8x16_t, ends: uint8x16_t| {
        let (pairs, ends) = (vreinterpretq_u16_u8(pairs), vreinterpretq_u16_u8(ends));
        (
            vreinterpretq_u8_u16(vzip1q_u16(pairs, ends)),
            vreinterpretq_u8_u16(vzip2q_u16(pairs, ends)),
        )
    };
    let (groups_0, groups_4) = groups(low_pairs, low_ends);
    let (groups_8, groups_12) = groups(high_pairs, high_ends);
    store16::<0, _>(dst, groups_0);
    store16::<16, _>(dst, groups_4);
    store16::<32, _>(dst, groups_8);
    store16::<48, _>(dst, groups_12);
}

/// Encodes in blocks of 12 bytes. An input shorter than one vector runs on arrays on the stack.
#[inline(always)]
#[target_feature(enable = "neon")]
fn encode_12(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let block = block_12(profile);
    // A slice pattern gives the last 16 bytes with no test of their address for null.
    match bytes.as_rchunks().1 {
        [.., last] => encode_steps(class, bytes, last, out, |first, len| to_end(first, len), block),
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

/// The block of the encoder: the 16 characters of the 12 bytes from `skip` in `src`. One `tbl`
/// takes the bytes in the order of each group, and gives zero to a lane past the end of `src`.
#[inline(always)]
#[target_feature(enable = "neon")]
fn block_12(profile: Profile) -> impl Fn(&[u8; 16], u8, &mut [u8; 16]) {
    let alphabet = table64::<0, _>(&Tables::of(profile).alphabet);
    let order = load16::<0, _>(&ENCODE_ORDER);
    // In each 32-bit lane of `b1 b0 b2 b1`, the low 16 bits hold the first and the second sextet
    // at bits 10 and 4, and the high 16 bits hold the third and the fourth at bits 6 and 0. `ushl`
    // shifts each 16-bit lane by its own count, to the right for a negative count, and moves each
    // sextet to the low bits of its byte.
    let lanes = |pair: u32| vreinterpretq_u16_u32(vdupq_n_u32(pair));
    let (high, low) = (lanes(0x0fc0_fc00), lanes(0x003f_03f0));
    let counts = |pair: u32| vreinterpretq_s16_u32(vdupq_n_u32(pair));
    // -10 and -6, and 4 and 8.
    let (right, left) = (counts(0xfffa_fff6), counts(0x0008_0004));
    move |src, skip, dst| {
        let index = vaddq_u8(order, vdupq_n_u8(skip));
        let input = vreinterpretq_u16_u8(vqtbl1q_u8(load16::<0, _>(src), index));
        let sextets = vorrq_u16(
            vshlq_u16(vandq_u16(input, high), right),
            vshlq_u16(vandq_u16(input, low), left),
        );
        store16::<0, _>(dst, vqtbl4q_u8(alphabet, vreinterpretq_u8_u16(sextets)));
    }
}

/// Decodes the characters `text`, with no padding, into `out`, and gives whether each is in the
/// alphabet and the trailing bits are zero. A short input runs inline in the caller.
#[inline(always)]
#[target_feature(enable = "neon")]
pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    if out.len() <= SHORT_MAX {
        decode_16(class, profile, text, out)
    } else {
        decode_long(class, profile, text, out)
    }
}

/// Decodes a long input: blocks of 64 characters with the interleaved loads and stores, then the
/// rest, at least 16 characters, in blocks of 16 characters.
#[inline(never)]
#[target_feature(enable = "neon")]
fn decode_long(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let lookups = lookups(Tables::of(profile));
    // The AND of the flags: bit 7 is clear if a byte is outside the alphabet.
    let mut valid = vdupq_n_u8(0xff);
    let blocks = text.len().saturating_sub(16) / 64;
    let (head, rest) = text.split_at(64 * blocks);
    let (head_out, rest_out) = out.split_at_mut(48 * blocks);
    // One block for each step: in a native build, the schedule of two blocks for each step took
    // 1.10 times as long at 65536 bytes (C109 probe 3).
    for (src, dst) in head.as_chunks::<64>().0.iter().zip(head_out.as_chunks_mut::<48>().0) {
        valid = vandq_u8(valid, decode_64(&lookups, src, dst));
    }
    let rest_valid = decode_16(class, profile, rest, rest_out);
    all_valid(valid) & rest_valid
}

/// The 48 bytes of 64 characters, and the AND of their flags. Each `sli` keeps only bits of the
/// sextets, so bit 7 of each flag leaves no trace in the bytes.
#[inline]
#[target_feature(enable = "neon")]
fn decode_64(lookups: &Lookups, src: &[u8; 64], dst: &mut [u8; 48]) -> uint8x16_t {
    let uint8x16x4_t(a, b, c, d) = load_quads(src);
    let (a, b, c, d) = (
        flags(lookups, a),
        flags(lookups, b),
        flags(lookups, c),
        flags(lookups, d),
    );
    // `a << 2 | b >> 4`, `b << 4 | c >> 2`, and `c << 6 | d`, each in 8 bits.
    let first = vsliq_n_u8::<2>(vshrq_n_u8::<4>(b), a);
    let second = vsliq_n_u8::<4>(vshrq_n_u8::<2>(c), b);
    let third = vsliq_n_u8::<6>(d, c);
    store_triples(dst, uint8x16x3_t(first, second, third));
    vandq_u8(vandq_u8(a, b), vandq_u8(c, d))
}

/// Decodes in blocks of 16 characters, and gives whether each is in the alphabet and the trailing
/// bits are zero.
#[inline(always)]
#[target_feature(enable = "neon")]
fn decode_16(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let lookups = lookups(Tables::of(profile));
    let order = load16::<0, _>(&DECODE_ORDER);
    let mut valid = vdupq_n_u8(0xff);
    let ends = (
        |last: &[u8; 16], len, fill| from_end(last, len, fill),
        |value: &[u8; 16], at| byte_at(value, at),
    );
    let trailing = decode_steps(class, text, out, ends, |src, dst| {
        let flags = flags(&lookups, load16::<0, _>(src));
        valid = vandq_u8(valid, flags);
        // Each 16-bit lane: `a << 6 | b` in its low 12 bits. Each 32-bit lane: the 24 bits of the
        // four sextets in its low three bytes. Bit 7 of each flag goes only to the fourth byte,
        // which the order drops.
        let pairs = vreinterpretq_u16_u8(flags);
        let pairs = vsliq_n_u16::<6>(vshrq_n_u16::<8>(pairs), pairs);
        let words = vreinterpretq_u32_u16(pairs);
        let words = vsliq_n_u32::<12>(vshrq_n_u32::<16>(words), words);
        store16::<0, _>(dst, vqtbl1q_u8(vreinterpretq_u8_u32(words), order));
    });
    all_valid(valid) & (trailing == 0)
}
