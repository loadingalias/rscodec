//! The Neon kernels. An input of up to `SHORT_MAX` bytes runs inline in the caller, in blocks of 16
//! characters. A longer input runs in a function of its own: whole blocks of 64 characters use the
//! interleaved loads and stores (`ld3`, `st4`, `ld4`, `st3`), and the rest uses blocks of 16
//! characters. The encoder looks up each sextet in the alphabet (`tbl`). The decoder looks up the
//! sextet of each character in a table of the 80 bytes from `+` (`tbl`), which sets bit 7 for a
//! byte outside the alphabet. It ORs the sextets of each block into one vector, and tests bit 7 once
//! after the last block.

#![expect(
    clippy::missing_safety_doc,
    reason = "each function is safe; only its target feature makes a call from code without the feature unsafe"
)]

use core::arch::aarch64::*;

use super::{Profile, Tables, bytes_len, symbols_len};
use crate::{
    ct::Class,
    simd::{
        SHORT_MAX,
        aarch64::{load_quads, load_triples, load16, store_quads, store_triples, store16, table64},
        blocks, whole_blocks,
    },
};

/// The bytes `b1 b0 b2 b1` of each three input bytes, in each 32-bit lane, for the encoder.
const ENCODE_ORDER: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];

/// The three bytes of each 32-bit lane after the decoder packs it, first byte first.
const DECODE_ORDER: [u8; 16] = [2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xff, 0xff, 0xff, 0xff];

/// The sextet table of the decoder in vectors: its first 64 bytes, and its last 16.
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
    let low = vqtbl4q_u8(lookups.low, index);
    let high = vqtbl1q_u8(lookups.high, vsubq_u8(index, vdupq_n_u8(64)));
    vorrq_u8(low, high)
}

/// The sextets of 16 characters, with bit 7 set for a byte outside the alphabet: the flags with
/// bit 7 flipped.
#[inline]
#[target_feature(enable = "neon")]
fn sextets(lookups: &Lookups, chars: uint8x16_t) -> uint8x16_t {
    veorq_u8(flags(lookups, chars), vdupq_n_u8(0x80))
}

/// Encodes `bytes` into the characters `out`, with no padding. A short input runs inline in the
/// caller.
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
/// loads and stores, then the rest in blocks of 12 bytes.
#[inline(never)]
#[target_feature(enable = "neon")]
fn encode_long(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let alphabet = table64::<0, _>(&Tables::of(profile).alphabet);
    let (read, written) = whole_blocks::<96, 128>(bytes, out, 96, symbols_len, |src, dst| {
        for (src, dst) in src.as_chunks::<48>().0.iter().zip(dst.as_chunks_mut::<64>().0) {
            encode_48(alphabet, src, dst);
        }
    });
    let (bytes, out) = (&bytes[read..], &mut out[written..]);
    let (read, written) = whole_blocks::<48, 64>(bytes, out, 48, symbols_len, |src, dst| {
        encode_48(alphabet, src, dst);
    });
    encode_12(class, profile, &bytes[read..], &mut out[written..]);
}

/// The 64 characters of 48 bytes.
#[inline]
#[target_feature(enable = "neon")]
fn encode_48(alphabet: uint8x16x4_t, src: &[u8; 48], dst: &mut [u8; 64]) {
    let sextet = vdupq_n_u8(63);
    let uint8x16x3_t(a, b, c) = load_triples(src);
    let first = vshrq_n_u8::<2>(a);
    let second = vandq_u8(vsriq_n_u8::<4>(vshlq_n_u8::<4>(a), b), sextet);
    let third = vandq_u8(vsriq_n_u8::<6>(vshlq_n_u8::<2>(b), c), sextet);
    let fourth = vandq_u8(c, sextet);
    let chars = |sextets| vqtbl4q_u8(alphabet, sextets);
    store_quads(
        dst,
        uint8x16x4_t(chars(first), chars(second), chars(third), chars(fourth)),
    );
}

/// Encodes in blocks of 12 bytes.
#[inline(always)]
#[target_feature(enable = "neon")]
fn encode_12(class: impl Class, profile: Profile, bytes: &[u8], out: &mut [u8]) {
    let alphabet = table64::<0, _>(&Tables::of(profile).alphabet);
    let order = load16::<0, _>(&ENCODE_ORDER);
    // In each 32-bit lane of `b1 b0 b2 b1`, the low 16 bits hold the first and the second sextet
    // at bits 10 and 4, and the high 16 bits hold the third and the fourth at bits 6 and 0. The
    // shifts move each sextet to the low bits of its byte.
    let lanes = |mask: u32| vreinterpretq_u16_u32(vdupq_n_u32(mask));
    let (first, second) = (lanes(0x0000_fc00), lanes(0x0000_03f0));
    let (third, fourth) = (lanes(0x0fc0_0000), lanes(0x003f_0000));
    blocks::<16, 16>(class, bytes, out, 12, 0, symbols_len, |src, dst| {
        let input = vreinterpretq_u16_u8(vqtbl1q_u8(load16::<0, _>(src), order));
        let sextets = vorrq_u16(
            vorrq_u16(
                vshrq_n_u16::<10>(vandq_u16(input, first)),
                vshlq_n_u16::<4>(vandq_u16(input, second)),
            ),
            vorrq_u16(
                vshrq_n_u16::<6>(vandq_u16(input, third)),
                vshlq_n_u16::<8>(vandq_u16(input, fourth)),
            ),
        );
        store16::<0, _>(dst, vqtbl4q_u8(alphabet, vreinterpretq_u8_u16(sextets)));
    });
}

/// Decodes the characters `text`, with no padding, into `out`, and gives whether each is in the
/// alphabet. A short input runs inline in the caller.
#[inline(always)]
#[target_feature(enable = "neon")]
pub(super) fn decode(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    if out.len() <= SHORT_MAX {
        decode_16(class, profile, text, out)
    } else {
        decode_long(class, profile, text, out)
    }
}

/// Decodes a long input: two blocks of 64 characters in each step, then one, with the interleaved
/// loads and stores, then the rest in blocks of 16 characters.
#[inline(never)]
#[target_feature(enable = "neon")]
fn decode_long(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let lookups = lookups(Tables::of(profile));
    // The AND of the flags: bit 7 is clear if a byte is outside the alphabet.
    let mut valid = vdupq_n_u8(0xff);
    let (read, written) = whole_blocks::<128, 96>(text, out, 128, bytes_len, |src, dst| {
        for (src, dst) in src.as_chunks::<64>().0.iter().zip(dst.as_chunks_mut::<48>().0) {
            valid = vandq_u8(valid, decode_64(&lookups, src, dst));
        }
    });
    let (text, out) = (&text[read..], &mut out[written..]);
    let (read, written) = whole_blocks::<64, 48>(text, out, 64, bytes_len, |src, dst| {
        valid = vandq_u8(valid, decode_64(&lookups, src, dst));
    });
    let rest_valid = decode_16(class, profile, &text[read..], &mut out[written..]);
    (vminvq_u8(valid) >= 0x80) & rest_valid
}

/// The 48 bytes of 64 characters, and the AND of their flags. Each bit select takes only bits of
/// the sextets, so bit 7 of each flag leaves no trace in the bytes.
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
    let join = |high: uint8x16_t, low: uint8x16_t, mask: u8| vbslq_u8(vdupq_n_u8(mask), high, low);
    let first = join(vshlq_n_u8::<2>(a), vshrq_n_u8::<4>(b), 0xfc);
    let second = join(vshlq_n_u8::<4>(b), vshrq_n_u8::<2>(c), 0xf0);
    let third = join(vshlq_n_u8::<6>(c), d, 0xc0);
    store_triples(dst, uint8x16x3_t(first, second, third));
    vandq_u8(vandq_u8(a, b), vandq_u8(c, d))
}

/// Decodes in blocks of 16 characters, and gives whether each is in the alphabet.
#[inline(always)]
#[target_feature(enable = "neon")]
fn decode_16(class: impl Class, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
    let lookups = lookups(Tables::of(profile));
    let order = load16::<0, _>(&DECODE_ORDER);
    let mut invalid = vdupq_n_u8(0);
    blocks::<16, 16>(class, text, out, 16, b'A', bytes_len, |src, dst| {
        let values = sextets(&lookups, load16::<0, _>(src));
        invalid = vorrq_u8(invalid, values);
        // Each 16-bit lane: `a << 6 | b`. Each 32-bit lane: the 24 bits of the four sextets.
        let pairs = vreinterpretq_u16_u8(values);
        let pairs = vsraq_n_u16::<8>(vandq_u16(vshlq_n_u16::<6>(pairs), vdupq_n_u16(0x0fc0)), pairs);
        let words = vreinterpretq_u32_u16(pairs);
        let words = vsraq_n_u32::<16>(vandq_u32(vshlq_n_u32::<12>(words), vdupq_n_u32(0x00ff_f000)), words);
        store16::<0, _>(dst, vqtbl1q_u8(vreinterpretq_u8_u32(words), order));
    });
    vminvq_u8(vmvnq_u8(invalid)) >= 0x80
}
