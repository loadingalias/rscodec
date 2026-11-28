use super::block::{self, Block, Carry, Classes, Escapes};

const ONES: u64 = u64::from_ne_bytes([0x01; 8]);
const HIGH: u64 = u64::from_ne_bytes([0x80; 8]);

#[inline(never)]
pub(super) fn block(input: &[u8], start: usize, carry: Carry) -> Block {
    let (bytes, before) = (&input[start..], block::before(input, start));
    let words = bytes.first_chunk::<64>().map_or_else(
        || block::tail(bytes),
        |bytes| core::array::from_fn(|index| u64::from_le_bytes(bytes.as_chunks::<8>().0[index])),
    );
    let classes = classify(words, before);
    classes.block(escapes(classes.backslash, carry.escaped()), carry, block::prefix_xor)
}

#[inline]
fn classify(words: [u64; 8], before: u32) -> Classes {
    let mut classes = Classes::default();
    for (index, word) in words.into_iter().enumerate() {
        let equal = |byte: u8| zeros(word ^ (ONES * u64::from(byte)));
        let curly = |byte: u8| zeros((word | (ONES * 0x20)) ^ (ONES * u64::from(byte)));
        let set = |mask: &mut u64, bytes: u64| *mask |= pack(bytes) << (8 * index);
        set(
            &mut classes.structural,
            curly(b'{') | curly(b'}') | equal(b':') | equal(b','),
        );
        set(
            &mut classes.whitespace,
            equal(b' ') | equal(b'\t') | equal(b'\n') | equal(b'\r'),
        );
        set(&mut classes.quote, equal(b'"'));
        set(&mut classes.backslash, equal(b'\\'));
        set(&mut classes.control, zeros(word & (ONES * 0xe0)));
    }
    classes.utf8_error = utf8_error(words, before);
    classes
}

#[inline]
const fn zeros(word: u64) -> u64 {
    !(((word & !HIGH) + !HIGH) | word) & HIGH
}

#[inline]
const fn pack(bytes: u64) -> u64 {
    (bytes >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

#[inline]
fn utf8_error(words: [u64; 8], before: u32) -> bool {
    if words.iter().fold(0, |any, word| any | word) & HIGH == 0 {
        return block::ends_incomplete(before);
    }
    let [_, mut first, mut second, mut third] = before.to_le_bytes();
    let mut error = false;
    for byte in words.into_iter().flat_map(u64::to_le_bytes) {
        error |= block::utf8_error_at(first, second, third, byte);
        (first, second, third) = (second, third, byte);
    }
    error
}

#[inline]
fn escapes(backslash: u64, escaped: u64) -> Escapes {
    let mut escapes = Escapes { escaped, starts: 0 };
    let mut rest = backslash & !escaped;
    while rest != 0 {
        let start = rest.isolate_lowest_one();
        escapes.starts |= start;
        escapes.escaped |= start << 1;
        rest &= !(start | start << 1);
    }
    escapes
}
