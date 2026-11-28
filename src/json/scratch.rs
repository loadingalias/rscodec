use core::cmp::Ordering;

use super::string;

/// Reader storage. Lend one slot per open container and per key of each open object.
/// An object with eight scalar members needs nine slots. Initialize each slot with [`Slot::EMPTY`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    fingerprint: u64,
    at: u32,
    len: u32,
}

impl Slot {
    /// A slot to fill an array of slots.
    pub const EMPTY: Self = Self {
        fingerprint: 0,
        at: 0,
        len: 0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Container {
    Root,
    Object,
    Array,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    pub(super) container: Container,
    pub(super) count: u32,
    pub(super) base: u32,
    pub(super) keys: u64,
}

impl Frame {
    pub(super) const ROOT: Self = Self {
        container: Container::Root,
        count: 0,
        base: 0,
        keys: 0,
    };

    #[inline(always)]
    pub(super) const fn save(self) -> Slot {
        Slot {
            fingerprint: self.keys | self.container as u64,
            at: self.base,
            len: self.count,
        }
    }

    #[inline(always)]
    pub(super) const fn load(slot: Slot) -> Self {
        let container = match slot.fingerprint & 3 {
            1 => Container::Object,
            2 => Container::Array,
            _ => Container::Root,
        };
        Self {
            container,
            count: slot.len,
            base: slot.at,
            keys: slot.fingerprint & !3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Key(Slot);

impl Key {
    #[inline(always)]
    pub(super) fn new(at: u32, raw: &[u8], len: u32, decoded: u32) -> Self {
        let sample = if len == decoded {
            sample(raw)
        } else {
            let decoded = decoded as usize;
            let mut words = [[0; 4]; 3];
            let starts = [0, (decoded / 2).saturating_sub(2), decoded.saturating_sub(4)];
            for (index, byte) in string::bytes(raw).enumerate() {
                for (word, start) in words.iter_mut().zip(starts) {
                    if let Some(slot) = index.checked_sub(start).and_then(|offset| word.get_mut(offset)) {
                        *slot = byte;
                    }
                }
            }
            if decoded < 4 {
                fold(&words[0][..decoded])
            } else {
                mix(words)
            }
        };
        Self(Slot {
            fingerprint: u64::from(decoded) << 32 | u64::from(sample),
            at,
            len,
        })
    }

    #[inline(always)]
    pub(super) const fn save(self) -> Slot {
        self.0
    }

    #[inline(always)]
    pub(super) const fn load(slot: Slot) -> Self {
        Self(slot)
    }

    #[inline(always)]
    fn raw(self, input: &[u8]) -> &[u8] {
        let start = self.0.at as usize + 1;
        &input[start..start + self.0.len as usize]
    }

    #[inline(always)]
    const fn plain(self) -> bool {
        self.0.fingerprint >> 32 == self.0.len as u64
    }

    #[inline(always)]
    pub(super) const fn bit(self) -> u64 {
        let mixed = self.0.fingerprint.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        1 << (2 + ((mixed as u128 * 62) >> 64) as u64)
    }

    #[inline(always)]
    pub(super) fn in_keys(self, keys: &[Slot], input: &[u8]) -> bool {
        keys.iter()
            .any(|slot| slot.fingerprint == self.0.fingerprint && Self(*slot).same_bytes(self, input))
    }

    #[cold]
    fn same_bytes(self, other: Self, input: &[u8]) -> bool {
        if self.plain() && other.plain() {
            self.raw(input) == other.raw(input)
        } else {
            string::bytes(self.raw(input)).eq(string::bytes(other.raw(input)))
        }
    }

    fn order(self, other: Self, input: &[u8]) -> Ordering {
        let bytes = if self.plain() && other.plain() {
            self.raw(input).cmp(other.raw(input))
        } else {
            string::bytes(self.raw(input)).cmp(string::bytes(other.raw(input)))
        };
        bytes.then(self.0.at.cmp(&other.0.at))
    }

    #[inline(always)]
    pub(super) const fn at(self) -> usize {
        self.0.at as usize
    }
}

#[inline(always)]
fn fold(bytes: &[u8]) -> u32 {
    match bytes {
        [] => 0,
        [first, ..] => {
            let len = bytes.len();
            u32::from(*first) << 16 | u32::from(bytes[len / 2]) << 8 | u32::from(bytes[len - 1])
        }
    }
}

#[inline(always)]
const fn mix([first, middle, last]: [[u8; 4]; 3]) -> u32 {
    u32::from_le_bytes(first) ^ u32::from_le_bytes(middle).rotate_left(11) ^ u32::from_le_bytes(last).rotate_left(22)
}

#[inline(always)]
fn sample(key: &[u8]) -> u32 {
    let len = key.len();
    let word = |start: usize| {
        key.get(start..)
            .and_then(<[u8]>::first_chunk)
            .copied()
            .unwrap_or_default()
    };
    match key.first_chunk() {
        Some(&first) => mix([first, word(len / 2 - 2), word(len - 4)]),
        None => fold(key),
    }
}

pub(super) fn first_duplicate(keys: &mut [Slot], input: &[u8]) -> Option<usize> {
    let mut seen = [0_u64; BUCKETS / 64];
    let mut collisions = 0;
    for (index, &slot) in keys.iter().enumerate() {
        let (word, bit) = bucket(slot.fingerprint);
        if seen[word] & bit != 0 {
            collisions += 1;
            if collisions > COLLISIONS_MAX {
                return sorted_first_duplicate(keys, input);
            }
            let key = Key::load(slot);
            if key.in_keys(&keys[..index], input) {
                return Some(key.at());
            }
        }
        seen[word] |= bit;
    }
    None
}

const BUCKETS: usize = 1024;

const COLLISIONS_MAX: usize = 8;

#[inline(always)]
fn bucket(fingerprint: u64) -> (usize, u64) {
    let mixed = fingerprint ^ fingerprint >> 32;
    let bucket =
        usize::try_from((mixed ^ mixed >> 10 ^ mixed >> 20) % BUCKETS as u64).expect("a bucket index fits usize");
    (bucket / 64, 1 << (bucket % 64))
}

fn sorted_first_duplicate(keys: &mut [Slot], input: &[u8]) -> Option<usize> {
    keys.sort_unstable_by_key(|slot| slot.fingerprint);
    let mut first: Option<usize> = None;
    for run in keys
        .chunk_by_mut(|a, b| a.fingerprint == b.fingerprint)
        .filter(|run| run.len() > 1)
    {
        run.sort_unstable_by(|a, b| Key::load(*a).order(Key::load(*b), input));
        for [a, b] in run.array_windows() {
            if Key::load(*a).same_bytes(Key::load(*b), input) {
                let at = Key::load(*b).at();
                first = Some(first.map_or(at, |first| first.min(at)));
            }
        }
    }
    first
}
