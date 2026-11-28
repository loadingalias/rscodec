use core::arch::aarch64::*;

use super::{Profile, Tables, decode_steps, encode_steps};
use crate::{
    ct::Class,
    simd::{SHORT_MAX, byte_at, from_end, short_blocks, to_end},
    vector::aarch64::{all_valid, load_quads, load_triples, load16, store_triples, store16, table64},
};

const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xff, 0xff, 0xff, 0xff];

struct Lookups {
    low: uint8x16x4_t,
    high: uint8x16_t,
}

#[inline]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn lookups(tables: &Tables) -> Lookups {
    Lookups {
        low: table64::<0, _>(&tables.sextets_from_plus),
        high: load16::<64, _>(&tables.sextets_from_plus),
    }
}

#[inline]
/// # Safety
/// The CPU must support Neon.
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

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
pub(super) fn encode(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    if bytes.len() <= SHORT_MAX {
        encode_12(class, profile, bytes, out);
    } else {
        encode_long(class, profile, bytes, out);
    }
}

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
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
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn encode_48(alphabet: uint8x16x4_t, src: &[u8; 48], dst: &mut [u8; 64]) {
    let sextet = vdupq_n_u8(63);
    let uint8x16x3_t(a, b, c) = load_triples(src);
    let chars = |sextets| vqtbl4q_u8(alphabet, sextets);
    let first = chars(vshrq_n_u8::<2>(a));
    let second = chars(vandq_u8(vsriq_n_u8::<4>(vshlq_n_u8::<4>(a), b), sextet));
    let third = chars(vandq_u8(vsriq_n_u8::<6>(vshlq_n_u8::<2>(b), c), sextet));
    let fourth = chars(vandq_u8(c, sextet));
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

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn encode_12(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let block = block_12(profile);
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

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn block_12(profile: Profile) -> impl Fn(&[u8; 16], u8, &mut [u8; 16]) {
    let alphabet = table64::<0, _>(&Tables::of(profile).alphabet);
    let order = load16::<0, _>(&ENCODE_ORDER);
    let lanes = |pair: u32| vreinterpretq_u16_u32(vdupq_n_u32(pair));
    let (high, low) = (lanes(0x0fc0_fc00), lanes(0x003f_03f0));
    let counts = |pair: u32| vreinterpretq_s16_u32(vdupq_n_u32(pair));
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

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    if out.len() <= SHORT_MAX {
        decode_16(class, profile, text, out)
    } else {
        decode_long(class, profile, text, out)
    }
}

#[inline(never)]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn decode_long(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let lookups = lookups(Tables::of(profile));
    let mut valid = vdupq_n_u8(0xff);
    let blocks = text.len().saturating_sub(16) / 64;
    let (head, rest) = text.split_at(64 * blocks);
    let (head_out, rest_out) = out.split_at_mut(48 * blocks);
    // Two blocks per iteration took 10% longer at 64 KiB on Graviton4.
    for (src, dst) in head.as_chunks::<64>().0.iter().zip(head_out.as_chunks_mut::<48>().0) {
        valid = vandq_u8(valid, decode_64(&lookups, src, dst));
    }
    let rest_valid = decode_16(class, profile, rest, rest_out);
    all_valid(valid) & rest_valid
}

#[inline]
/// # Safety
/// The CPU must support Neon.
#[target_feature(enable = "neon")]
fn decode_64(lookups: &Lookups, src: &[u8; 64], dst: &mut [u8; 48]) -> uint8x16_t {
    let uint8x16x4_t(a, b, c, d) = load_quads(src);
    let (a, b, c, d) = (
        flags(lookups, a),
        flags(lookups, b),
        flags(lookups, c),
        flags(lookups, d),
    );
    let first = vsliq_n_u8::<2>(vshrq_n_u8::<4>(b), a);
    let second = vsliq_n_u8::<4>(vshrq_n_u8::<2>(c), b);
    let third = vsliq_n_u8::<6>(d, c);
    store_triples(dst, uint8x16x3_t(first, second, third));
    vandq_u8(vandq_u8(a, b), vandq_u8(c, d))
}

#[inline(always)]
/// # Safety
/// The CPU must support Neon.
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
        let pairs = vreinterpretq_u16_u8(flags);
        let pairs = vsliq_n_u16::<6>(vshrq_n_u16::<8>(pairs), pairs);
        let words = vreinterpretq_u32_u16(pairs);
        let words = vsliq_n_u32::<12>(vshrq_n_u32::<16>(words), words);
        store16::<0, _>(dst, vqtbl1q_u8(vreinterpretq_u8_u32(words), order));
    });
    all_valid(valid) & (trailing == 0)
}
