//! RFC 8785 uses lowercase hex escapes and the short escapes for backspace, tab, newline, form feed, and carriage return.

#[cfg(test)]
mod tests;

const BLOCK: usize = 16;

cfg_select! {
    target_arch = "x86_64" => {
        use super::x86::escape::{BITS, escapes, escapes_of_words};
    }
    all(target_arch = "aarch64", target_feature = "neon") => {
        use super::aarch64::escape::{BITS, escapes, escapes_of_words};
    }
    all(target_arch = "wasm32", target_feature = "simd128") => {
        use super::wasm32::{BITS, escapes, escapes_of_words};
    }
    _ => {
        const BITS: u32 = 1;

        #[inline(always)]
        fn escapes(block: &[u8; BLOCK]) -> u64 {
            let (words, _) = block.as_chunks::<8>();
            escapes_of_words(u64::from_le_bytes(words[0]), u64::from_le_bytes(words[1]))
        }

        #[inline(always)]
        fn escapes_of_words(low: u64, high: u64) -> u64 {
            const ONES: u64 = u64::from_ne_bytes([0x01; 8]);
            const HIGH: u64 = u64::from_ne_bytes([0x80; 8]);
            let zeros = |word: u64| !(((word & !HIGH) + !HIGH) | word) & HIGH;
            let pack = |bytes: u64| (bytes >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56;
            let escapes = |word: u64| {
                let (quote, backslash, high_bits) = (ONES * u64::from(b'"'), ONES * u64::from(b'\\'), ONES * 0xe0);
                pack(zeros(word ^ quote) | zeros(word ^ backslash) | zeros(word & high_bits))
            };
            escapes(low) | escapes(high) << 8
        }
    }
}

#[inline(always)]
const fn first(mask: u64) -> usize {
    (mask.trailing_zeros() / BITS) as usize
}

const SHORT: [u8; 32] = {
    let mut short = [0; 32];
    (short[0x08], short[0x09], short[0x0a], short[0x0c], short[0x0d]) = (b'b', b't', b'n', b'f', b'r');
    short
};

#[inline]
fn escape(byte: u8) -> ([u8; 8], usize) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    match SHORT.get(usize::from(byte)) {
        Some(0) => {
            let (high, low) = (HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 0x0f)]);
            ([b'\\', b'u', b'0', b'0', high, low, 0, 0], 6)
        }
        Some(&short) => ([b'\\', short, 0, 0, 0, 0, 0, 0], 2),
        None => ([b'\\', byte, 0, 0, 0, 0, 0, 0], 2),
    }
}

#[inline(always)]
fn tail(text: &[u8]) -> (u64, u64) {
    const SPACES: u64 = u64::from_ne_bytes([b' '; 8]);
    let len = text.len();
    let bits = |count: usize| u32::try_from(8 * count).unwrap_or(u32::MAX);
    if let (Some(first), Some(last)) = (text.first_chunk::<8>(), text.last_chunk::<8>()) {
        let last = u64::from_le_bytes(*last).unbounded_shr(bits(BLOCK - len));
        (u64::from_le_bytes(*first), last | SPACES.unbounded_shl(bits(len - 8)))
    } else if let (Some(first), Some(last)) = (text.first_chunk::<4>(), text.last_chunk::<4>()) {
        let (first, last) = (u32::from_le_bytes(*first), u32::from_le_bytes(*last));
        let low = u64::from(first) | u64::from(last) << bits(len - 4) | SPACES.unbounded_shl(bits(len));
        (low, SPACES)
    } else {
        let low = text.iter().rev().fold(0, |word, &byte| word << 8 | u64::from(byte));
        (low | SPACES.unbounded_shl(bits(len)), SPACES)
    }
}

#[inline(always)]
fn step(text: &[u8], out: &mut [u8], at: usize) -> (u64, usize) {
    let dest = out.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<BLOCK>);
    if let Some(block) = text.first_chunk::<BLOCK>() {
        if let Some(dest) = dest {
            *dest = *block;
        }
        return (escapes(block), BLOCK);
    }
    let (low, high) = tail(text);
    if let Some(dest) = dest {
        let (words, _) = dest.as_chunks_mut::<8>();
        (words[0], words[1]) = (low.to_le_bytes(), high.to_le_bytes());
    }
    (escapes_of_words(low, high), text.len())
}

#[inline(always)]
pub(super) fn write(out: &mut [u8], at: usize, text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.len() < BLOCK
        && let Some(dest) = out.get_mut(at..).and_then(<[u8]>::first_chunk_mut::<{ BLOCK + 1 }>)
    {
        let (low, high) = tail(bytes);
        if escapes_of_words(low, high) == 0 {
            dest[0] = b'"';
            let (words, _) = dest[1..].as_chunks_mut::<8>();
            (words[0], words[1]) = (low.to_le_bytes(), high.to_le_bytes());
            dest[1 + bytes.len()] = b'"';
            return Some(at + bytes.len() + 2);
        }
    }
    write_scanned(out, at, text)
}

fn write_scanned(out: &mut [u8], mut at: usize, text: &str) -> Option<usize> {
    *out.get_mut(at)? = b'"';
    at += 1;
    let mut rest = text.as_bytes();
    loop {
        let room = out.len().saturating_sub(at);
        let (mut copied, mut escaped) = (0, false);
        if rest.len() >= BLOCK && room >= BLOCK {
            let blocks = rest.len().min(room) / BLOCK;
            let (texts, _) = rest.as_chunks::<BLOCK>();
            let (dests, _) = out.get_mut(at..)?.as_chunks_mut::<BLOCK>();
            for (block, dest) in texts.get(..blocks)?.iter().zip(dests.get_mut(..blocks)?) {
                *dest = *block;
                let mask = escapes(block);
                if mask != 0 {
                    (copied, escaped) = (copied + first(mask), true);
                    break;
                }
                copied += BLOCK;
            }
        } else {
            let (mask, len) = step(rest, out, at);
            (copied, escaped) = if mask == 0 { (len, false) } else { (first(mask), true) };
            if room < BLOCK {
                out.get_mut(at..at + copied)?.copy_from_slice(&rest[..copied]);
            }
            if mask == 0 && len < BLOCK {
                at += copied;
                break;
            }
        }
        at += copied;
        rest = &rest[copied..];
        let Some((&byte, after)) = rest.split_first().filter(|_| escaped) else {
            continue;
        };
        let (form, form_len) = escape(byte);
        match out.get_mut(at..at + form.len()) {
            Some(dest) => dest.copy_from_slice(&form),
            None => out.get_mut(at..at + form_len)?.copy_from_slice(&form[..form_len]),
        }
        at += form_len;
        rest = after;
    }
    *out.get_mut(at)? = b'"';
    Some(at + 1)
}

/// The exact length of the JSON string that [`write`] writes for `text`, quotes included.
#[must_use]
pub fn string_len(text: &str) -> usize {
    let mut len = text.len().saturating_add(2);
    let mut rest = text.as_bytes();
    loop {
        let (mask, count) = step(rest, &mut [], 0);
        if mask == 0 {
            if count < BLOCK {
                return len;
            }
            rest = &rest[BLOCK..];
            continue;
        }
        let index = first(mask);
        len = len.saturating_add(escape(rest[index]).1 - 1);
        rest = &rest[index + 1..];
    }
}
