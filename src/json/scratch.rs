//! The scratch of the reader: one stack of slots in memory that the caller lends. A slot holds an
//! open container, or a key of an open object. The keys give the duplicate check, with no hash.

use core::cmp::Ordering;

use super::string;

/// Reader storage. Lend one slot per open container and per key of each open object.
/// An object with eight scalar members needs nine slots. Initialize each slot with [`Slot::EMPTY`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// A key: its length after unescape and a sample of its bytes after unescape. A container: its
    /// kind.
    fingerprint: u64,
    /// A key: the offset of its opening quote. A container: the index of its first key.
    at: u32,
    /// A key: its length in the input. A container: its members or elements so far.
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

/// The kind of an open container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Container {
    /// The document, outside each container.
    Root,
    Object,
    Array,
}

/// The state of an open container that the reader keeps while a nested container is open.
#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    pub(super) container: Container,
    /// The members of an object, or the elements of an array, so far.
    pub(super) count: u32,
    /// The index of the first key of an object in the scratch.
    pub(super) base: u32,
    /// The filter of the first 16 keys of an object, which compare with the earlier keys at once:
    /// the bit of [`Key::bit`] of each. Bits 0 and 1 are clear, so that the slot of a saved frame
    /// holds its container there.
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

/// A key of an open object.
///
/// Its fingerprint holds its length and a sample of its bytes after unescape, so two equal keys
/// have equal fingerprints. Most comparisons end at the fingerprint, with no load of the input.
#[derive(Clone, Copy, Debug)]
pub(super) struct Key(Slot);

impl Key {
    /// The key whose opening quote is at `at`, whose bytes between its quotes are `raw`, and whose
    /// length after unescape is `decoded`. Each escape is longer than its character, so the key has
    /// an escape exactly when `decoded` is less than the length of `raw`.
    #[inline(always)]
    pub(super) fn new(at: u32, raw: &[u8], len: u32, decoded: u32) -> Self {
        let sample = if len == decoded {
            sample(raw)
        } else {
            let decoded = decoded as usize;
            // The bytes at the places that `sample` reads, from one pass over the decoded key.
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

    /// The bytes of the key between its quotes.
    #[inline(always)]
    fn raw(self, input: &[u8]) -> &[u8] {
        let start = self.0.at as usize + 1;
        &input[start..start + self.0.len as usize]
    }

    /// Whether the key has no escape.
    #[inline(always)]
    const fn plain(self) -> bool {
        self.0.fingerprint >> 32 == self.0.len as u64
    }

    /// The bit of the key in the filter of [`Frame::keys`]: one of bits 2 to 63, from its
    /// fingerprint. Equal keys have the same bit. A multiply by an odd constant moves each bit of
    /// the fingerprint into the high bits, and the high word of a product with 62 maps them to
    /// `0..62`, with no division.
    #[inline(always)]
    pub(super) const fn bit(self) -> u64 {
        let mixed = self.0.fingerprint.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        1 << (2 + ((mixed as u128 * 62) >> 64) as u64)
    }

    /// Whether a key of `keys` equals this key after unescape. The loop compares the fingerprints
    /// in the slots, with no copy of a slot.
    #[inline(always)]
    pub(super) fn in_keys(self, keys: &[Slot], input: &[u8]) -> bool {
        keys.iter()
            .any(|slot| slot.fingerprint == self.0.fingerprint && Self(*slot).same_bytes(self, input))
    }

    /// Whether the bytes of the keys are equal after unescape, for keys of equal fingerprints.
    #[cold]
    fn same_bytes(self, other: Self, input: &[u8]) -> bool {
        if self.plain() && other.plain() {
            self.raw(input) == other.raw(input)
        } else {
            string::bytes(self.raw(input)).eq(string::bytes(other.raw(input)))
        }
    }

    /// An order of keys of equal fingerprints in which equal keys are next to each other, in the
    /// order of their offsets.
    fn order(self, other: Self, input: &[u8]) -> Ordering {
        let bytes = if self.plain() && other.plain() {
            self.raw(input).cmp(other.raw(input))
        } else {
            string::bytes(self.raw(input)).cmp(string::bytes(other.raw(input)))
        };
        bytes.then(self.0.at.cmp(&other.0.at))
    }

    /// The offset of the opening quote.
    #[inline(always)]
    pub(super) const fn at(self) -> usize {
        self.0.at as usize
    }
}

/// The first, the middle, and the last byte of a key of fewer than 4 bytes, in one word, with no
/// branch on the length.
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

/// The first 4, the middle 4, and the last 4 bytes of a key of 4 bytes or more, in one word. Keys
/// of one object often share a length and a first word, for example `profile_link_color` and
/// `profile_text_color`, and the other two words tell most of them apart.
#[inline(always)]
const fn mix([first, middle, last]: [[u8; 4]; 3]) -> u32 {
    u32::from_le_bytes(first) ^ u32::from_le_bytes(middle).rotate_left(11) ^ u32::from_le_bytes(last).rotate_left(22)
}

/// The sample of the bytes `key` for its fingerprint: [`fold`] of a key of fewer than 4 bytes, and
/// [`mix`] of a longer one.
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

/// The offset of the first key of `keys`, in the order of the input, that equals an earlier key.
/// The reader calls it when an object of more than 16 members ends.
///
/// A set of [`BUCKETS`] bits filters the keys: each key sets the bit of its fingerprint, and only a
/// key whose bit is set already compares with the earlier keys. Equal keys have equal
/// fingerprints, so the filter misses no duplicate. After [`COLLISIONS_MAX`] such keys, for example
/// in a hostile object, it sorts the keys instead. Thus each set of keys takes O(n log n)
/// comparisons at most, with no seed and no state from another document.
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

/// The bits of the filter of [`first_duplicate`]. Its keys rarely share a bit: 40 keys share one
/// with a chance of about 1 in 2.
const BUCKETS: usize = 1024;

/// The keys whose bit of the filter is set already, before [`first_duplicate`] sorts.
const COLLISIONS_MAX: usize = 8;

/// The word and the bit of `fingerprint` in the filter of [`first_duplicate`].
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bucket is under BUCKETS, which usize holds"
)]
fn bucket(fingerprint: u64) -> (usize, u64) {
    let mixed = fingerprint ^ fingerprint >> 32;
    let bucket = (mixed ^ mixed >> 10 ^ mixed >> 20) as usize % BUCKETS;
    (bucket / 64, 1 << (bucket % 64))
}

/// [`first_duplicate`] by a sort. The keys sort by their fingerprints, with no load of the input.
/// Then each run of equal fingerprints sorts by its bytes and offsets, so the second key of each
/// run of equal keys is its first duplicate.
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
