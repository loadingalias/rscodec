//! base64 ([RFC 4648, section 4]) with padding, and base64url ([RFC 4648, section 5]) without
//! padding.
//!
//! Each [`Profile`] decodes only the canonical text of its bytes, so each byte string has one text:
//! only the characters of its alphabet, no whitespace, the padding of the profile, and zero bits
//! after the last byte. The encoder writes that text.
//!
//! The source of each backend has no branch and no table load that depends on the bytes or the
//! characters, so that a secret, for example a key, can go through the codec. The time is to
//! depend only on the lengths, and for an error, on its offset. A check of the compiled code and a
//! timing test of this rule are not complete yet. An [`Error`] holds no byte of the input.
//!
//! ```
//! use rscodec::base64::Profile;
//!
//! let mut bytes = [0; 3];
//! assert_eq!(Profile::Url.decode(b"-_-_", &mut bytes)?, [0xfb, 0xff, 0xbf]);
//! let mut text = [0; 4];
//! assert_eq!(Profile::Url.encode(&bytes, &mut text)?, b"-_-_");
//! # Ok::<(), rscodec::base64::Error>(())
//! ```
//!
//! [RFC 4648, section 4]: https://www.rfc-editor.org/rfc/rfc4648#section-4
//! [RFC 4648, section 5]: https://www.rfc-editor.org/rfc/rfc4648#section-5

use core::fmt;

use crate::{Backend, cpu, ct};

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod portable;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The alphabet and the padding of a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// base64url without padding ([RFC 4648, section 5]): `A`–`Z`, `a`–`z`, `0`–`9`, `-`, and `_`.
    /// WebAuthn, JOSE, and OAuth use it.
    ///
    /// [RFC 4648, section 5]: https://www.rfc-editor.org/rfc/rfc4648#section-5
    Url,
    /// base64 with padding ([RFC 4648, section 4]): `A`–`Z`, `a`–`z`, `0`–`9`, `+`, and `/`, and
    /// `=` to make the length a multiple of 4.
    ///
    /// [RFC 4648, section 4]: https://www.rfc-editor.org/rfc/rfc4648#section-4
    Standard,
}

/// The cause of a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// No text of the profile has this length: 4n + 1 characters for [`Profile::Url`], and a length
    /// that is not a multiple of 4 for [`Profile::Standard`]. The offset is the length.
    Length,
    /// The byte at the offset is not a character of the alphabet. In [`Profile::Standard`], the
    /// padding `=` is a character only in the last two places.
    Symbol,
    /// The character at the offset, the last before the padding, has bits after the last byte, and
    /// they are not zero. Thus the text is not the canonical text of its bytes.
    TrailingBits,
    /// The output slice is too short. The offset is the length of the result, and the recovery is
    /// an output slice of that length.
    Output,
}

/// A refusal: its cause and the offset of the cause. It holds no byte of the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Error {
    /// The cause.
    pub kind: ErrorKind,
    /// The offset in the text, or for [`ErrorKind::Output`], the length of the result.
    pub offset: usize,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = self.offset;
        match self.kind {
            ErrorKind::Length => write!(f, "no base64 text has {offset} characters"),
            ErrorKind::Symbol => write!(f, "the base64 text has a byte outside its alphabet at offset {offset}"),
            ErrorKind::TrailingBits => write!(
                f,
                "the base64 text has nonzero bits after its last byte at offset {offset}"
            ),
            ErrorKind::Output => write!(f, "the output slice is shorter than the {offset} bytes of the result"),
        }
    }
}

impl core::error::Error for Error {}

impl Profile {
    /// The number of characters of the text of `len` bytes, or `None` if the number overflows
    /// `usize`.
    #[must_use]
    pub const fn encoded_len(self, len: usize) -> Option<usize> {
        let Some(full) = (len / 3).checked_mul(4) else {
            return None;
        };
        let rest = match (self, len % 3) {
            (_, 0) => 0,
            (Self::Url, rest) => rest + 1,
            (Self::Standard, _) => 4,
        };
        full.checked_add(rest)
    }

    /// The number of bytes that `text` decodes to. It reads only the length and the padding.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Length`] if no text of the profile has the length of `text`.
    pub const fn decoded_len(self, text: &[u8]) -> Result<usize, Error> {
        match self.symbols(text) {
            Ok(symbols) => Ok(bytes_len(symbols.len())),
            Err(error) => Err(error),
        }
    }

    /// Encodes `bytes` into the start of `out`, and gives the text, which is ASCII.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Output`] if `out` is shorter than [`Profile::encoded_len`].
    #[inline]
    pub fn encode<'o>(self, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_with(Backend::detect(), bytes, out)
    }

    /// [`Profile::encode`] on `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::encode`].
    #[inline]
    pub fn encode_with<'o>(self, backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(bytes, out, |bytes, text| {
            cpu::constant_time(|| kernel::encode(backend, self, bytes, text));
        })
    }

    /// Encodes `bytes` with `kernel`, which writes the characters before the padding.
    #[inline(always)]
    fn encode_by<'o>(
        self,
        bytes: &[u8],
        out: &'o mut [u8],
        kernel: impl FnOnce(&[u8], &mut [u8]),
    ) -> Result<&'o mut [u8], Error> {
        // A slice has at most `isize::MAX` bytes, so the length cannot overflow.
        let len = self.encoded_len(bytes.len()).unwrap_or(usize::MAX);
        let out = out.get_mut(..len).ok_or(Error {
            kind: ErrorKind::Output,
            offset: len,
        })?;
        let (text, padding) = out.split_at_mut(symbols_len(bytes.len()));
        // At most two bytes, with a store each: a fill of a length that the compiler does not know
        // is a call to `memset`.
        if let Some((last, rest)) = padding.split_last_mut() {
            *last = b'=';
            if let Some(first) = rest.first_mut() {
                *first = b'=';
            }
        }
        kernel(bytes, text);
        Ok(out)
    }

    /// Decodes the canonical text `text` into the start of `out`, and gives the bytes.
    ///
    /// The bytes of `out` after the result, and all of `out` after an error, are unspecified.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::Length`] if no text of the profile has the length of `text`.
    /// - [`ErrorKind::Output`] if `out` is shorter than [`Profile::decoded_len`].
    /// - [`ErrorKind::Symbol`] at the first byte of `text` that is not a character of the alphabet.
    /// - [`ErrorKind::TrailingBits`] if the last character before the padding has bits after the last
    ///   byte that are not zero.
    ///
    /// The decoder reads all of `text` before it refuses it for a symbol or the trailing bits.
    ///
    /// # Examples
    ///
    /// Decode a token into a buffer on the stack:
    ///
    /// ```
    /// use rscodec::base64::{Error, ErrorKind, Profile};
    ///
    /// // A WebAuthn challenge: 32 bytes in base64url without padding.
    /// let token = b"NI2kalLUY_EBIuNcW6yDtzD4K6bGemOZJL9NWqY_heQ";
    /// let mut buffer = [0; 32];
    /// let challenge = Profile::Url.decode(token, &mut buffer)?;
    /// assert_eq!(challenge[..4], [0x34, 0x8d, 0xa4, 0x6a]);
    ///
    /// // `/` is a character of base64, not of base64url, so the text refuses at the `/`.
    /// let standard = b"NI2kalLUY/EBIuNcW6yDtzD4K6bGemOZJL9NWqY/heQ";
    /// let refusal = Error { kind: ErrorKind::Symbol, offset: 9 };
    /// assert_eq!(Profile::Url.decode(standard, &mut buffer), Err(refusal));
    /// # Ok::<(), Error>(())
    /// ```
    #[inline]
    pub fn decode<'o>(self, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_with(Backend::detect(), text, out)
    }

    /// [`Profile::decode`] on `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::decode`].
    #[inline]
    pub fn decode_with<'o>(self, backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(text, out, |symbols, out| {
            cpu::constant_time(|| kernel::decode(backend, self, symbols, out))
        })
    }

    /// Decodes `text` with `kernel`, which decodes the characters before the padding and gives
    /// whether each is in the alphabet.
    #[inline(always)]
    fn decode_by<'o>(
        self,
        text: &[u8],
        out: &'o mut [u8],
        kernel: impl FnOnce(&[u8], &mut [u8]) -> bool,
    ) -> Result<&'o mut [u8], Error> {
        let symbols = self.symbols(text)?;
        let len = bytes_len(symbols.len());
        let out = out.get_mut(..len).ok_or(Error {
            kind: ErrorKind::Output,
            offset: len,
        })?;
        let valid = kernel(symbols, out);
        // The bits of the last character after the last byte: 4 bits after 4n + 2 characters, and
        // 2 bits after 4n + 3.
        let trailing_bits = match symbols.len() % 4 {
            2 => 0x0f,
            3 => 0x03,
            _ => 0,
        };
        let last = symbols.last().map_or(0, |&byte| portable::sextet(self, byte));
        // One branch for the whole text.
        if !(valid & (last & trailing_bits == 0)) {
            return Err(self.refusal(symbols));
        }
        Ok(out)
    }

    /// The characters of `text` before the padding.
    const fn symbols(self, text: &[u8]) -> Result<&[u8], Error> {
        let length = Error {
            kind: ErrorKind::Length,
            offset: text.len(),
        };
        match self {
            Self::Url if text.len() % 4 == 1 => Err(length),
            Self::Url => Ok(text),
            Self::Standard if !text.len().is_multiple_of(4) => Err(length),
            // The padding gives the length of the bytes, so the branch on it shows nothing more.
            Self::Standard => Ok(match text {
                [symbols @ .., b'=', b'='] | [symbols @ .., b'='] | symbols => symbols,
            }),
        }
    }

    /// The error of `symbols`, which have a byte outside the alphabet or nonzero trailing bits. Its
    /// time depends only on the length.
    fn refusal(self, symbols: &[u8]) -> Error {
        let first = ct::first_negative(symbols.iter().map(|&byte| portable::sextet(self, byte)));
        if first < symbols.len() {
            Error {
                kind: ErrorKind::Symbol,
                offset: first,
            }
        } else {
            Error {
                kind: ErrorKind::TrailingBits,
                offset: symbols.len().saturating_sub(1),
            }
        }
    }
}

/// The number of characters before the padding in the text of `len` bytes. A slice has at most
/// `isize::MAX` bytes, so the result cannot overflow.
const fn symbols_len(len: usize) -> usize {
    len / 3 * 4 + (len % 3 * 4).div_ceil(3)
}

/// The number of bytes of `len` characters with no padding.
const fn bytes_len(len: usize) -> usize {
    len / 4 * 3 + (len % 4).saturating_sub(1)
}

/// The calls of the kernel of each backend.
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    expect(
        unsafe_code,
        reason = "a `Backend` holds only an instruction set that the CPU has, so its target-feature kernels can run"
    )
)]
mod kernel {
    #[cfg(target_arch = "aarch64")]
    use super::aarch64;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use super::wasm32;
    #[cfg(target_arch = "x86_64")]
    use super::x86;
    use super::{Profile, portable};
    use crate::{Backend, Isa};

    /// Encodes `bytes` into the characters `out`, with no padding.
    #[inline]
    pub(super) fn encode(backend: Backend, profile: Profile, bytes: &[u8], out: &mut [u8]) {
        match backend.isa() {
            Isa::Portable => portable::encode(profile, bytes, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Ssse3 => unsafe { x86::ssse3::encode(profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx2 => unsafe { x86::avx2::encode(profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Bw => unsafe { x86::avx512bw::encode(profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Vbmi => unsafe { x86::avx512vbmi::encode(profile, bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Neon => unsafe { aarch64::encode(profile, bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => wasm32::encode(profile, bytes, out),
        }
    }

    /// Decodes the characters `text`, with no padding, into `out`, and gives whether each is in the
    /// alphabet.
    #[inline]
    pub(super) fn decode(backend: Backend, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        match backend.isa() {
            Isa::Portable => portable::decode(profile, text, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Ssse3 => unsafe { x86::ssse3::decode(profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx2 => unsafe { x86::avx2::decode(profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Bw => unsafe { x86::avx512bw::decode(profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Vbmi => unsafe { x86::avx512vbmi::decode(profile, text, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Neon => unsafe { aarch64::decode(profile, text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => wasm32::decode(profile, text, out),
        }
    }
}

/// The tables of the SIMD kernels of a profile.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
struct Tables {
    /// The tables of the nibble lookups of the x86_64 and WebAssembly kernels.
    #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
    nibbles: Nibbles,
    /// The character of each sextet.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    alphabet: [u8; 64],
    /// The sextet of each ASCII byte, or 0x80 for a byte outside the alphabet.
    #[cfg(target_arch = "x86_64")]
    sextets: [[u8; 64]; 2],
    /// The sextet with bit 7 set of each byte from `+`, the first byte of the two alphabets, to `z`,
    /// the last, or zero for a byte outside the alphabet.
    #[cfg(target_arch = "aarch64")]
    sextets_from_plus: [u8; 80],
}

/// The tables of the nibble lookups (Muła and Lemire, 2018).
#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
struct Nibbles {
    /// A byte `c` is outside the alphabet if `lo[c & 15] & hi[c >> 4]` is not zero. Each bit of `hi`
    /// is one class of high nibbles, and `lo` sets the bit of each class for which the low nibble is
    /// outside the alphabet.
    lo: [u8; 16],
    hi: [u8; 16],
    /// The step from a character to its sextet, at the high nibble of the character. The step of
    /// `special` is at its high nibble plus `special_index`.
    roll: [u8; 16],
    special: u8,
    special_index: u8,
    /// The step from a sextet to its character, at its class: 0 for `a`–`z`, 1 to 10 for
    /// `0`–`9`, 11 and 12 for the last two characters, and 13 for `A`–`Z`.
    shift: [u8; 16],
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
impl Tables {
    #[inline]
    const fn of(profile: Profile) -> &'static Self {
        // The steps are bytes: -65 is 0xbf, for example.
        const URL: Tables = Tables {
            #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
            nibbles: Nibbles {
                lo: [
                    0x0b, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x07, 0x37, 0x37, 0x35, 0x37, 0x27,
                ],
                hi: [
                    0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x08, 0x20, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                ],
                roll: [0, 0, 17, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0xe0, 0, 0, 0, 0, 0, 0, 0],
                special: b'_',
                special_index: 3,
                shift: [
                    71, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xef, 32, 65, 0, 0,
                ],
            },
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            alphabet: Tables::alphabet(Profile::Url),
            #[cfg(target_arch = "x86_64")]
            sextets: Tables::sextets(Profile::Url),
            #[cfg(target_arch = "aarch64")]
            sextets_from_plus: Tables::sextets_from_plus(Profile::Url),
        };
        const STANDARD: Tables = Tables {
            #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
            nibbles: Nibbles {
                lo: [
                    0x0b, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x07, 0x15, 0x17, 0x17, 0x17, 0x15,
                ],
                hi: [
                    0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x08, 0x10, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                ],
                roll: [0, 16, 19, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0],
                special: b'/',
                special_index: 0xff,
                shift: [
                    71, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xfc, 0xed, 0xf0, 65, 0, 0,
                ],
            },
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            alphabet: Tables::alphabet(Profile::Standard),
            #[cfg(target_arch = "x86_64")]
            sextets: Tables::sextets(Profile::Standard),
            #[cfg(target_arch = "aarch64")]
            sextets_from_plus: Tables::sextets_from_plus(Profile::Standard),
        };
        match profile {
            Profile::Url => &URL,
            Profile::Standard => &STANDARD,
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const fn alphabet(profile: Profile) -> [u8; 64] {
        let mut alphabet = [0; 64];
        let mut value: u8 = 0;
        while value < 64 {
            alphabet[value as usize] = portable::symbol(profile, value as i16);
            value += 1;
        }
        alphabet
    }

    #[cfg(target_arch = "aarch64")]
    const fn sextets_from_plus(profile: Profile) -> [u8; 80] {
        let mut sextets = [0; 80];
        let mut index: u8 = 0;
        while index < 80 {
            let sextet = portable::sextet(profile, b'+' + index);
            if sextet >= 0 {
                sextets[index as usize] = 0x80 | ct::low(sextet);
            }
            index += 1;
        }
        sextets
    }

    #[cfg(target_arch = "x86_64")]
    const fn sextets(profile: Profile) -> [[u8; 64]; 2] {
        let mut sextets = [[0x80; 64]; 2];
        let mut byte: u8 = 0;
        while byte < 128 {
            let sextet = portable::sextet(profile, byte);
            if sextet >= 0 {
                sextets[(byte / 64) as usize][(byte % 64) as usize] = ct::low(sextet);
            }
            byte += 1;
        }
        sextets
    }
}

/// The proofs of the portable backend, for `cargo kani`.
#[cfg(kani)]
mod proofs {
    use super::{Profile, portable};

    /// The largest input of the proofs: two blocks of three bytes.
    const BYTES_MAX: usize = 6;

    fn profile() -> Profile {
        if kani::any() { Profile::Url } else { Profile::Standard }
    }

    fn encode(profile: Profile, bytes: &[u8], out: &mut [u8]) -> usize {
        let kernel = |bytes: &[u8], text: &mut [u8]| portable::encode(profile, bytes, text);
        match profile.encode_by(bytes, out, kernel) {
            Ok(text) => text.len(),
            Err(_) => panic!("the output has room for the text"),
        }
    }

    /// Each input of up to `BYTES_MAX` bytes decodes from its text to itself.
    #[kani::proof]
    #[kani::unwind(9)]
    fn each_input_round_trips() {
        let profile = profile();
        let input: [u8; BYTES_MAX] = kani::any();
        let len = kani::any_where(|&len: &usize| len <= BYTES_MAX);
        let mut text = [0; 8];
        let text_len = encode(profile, &input[..len], &mut text);
        let mut bytes = [0; BYTES_MAX];
        let kernel = |text: &[u8], out: &mut [u8]| portable::decode(profile, text, out);
        let decoded = profile.decode_by(&text[..text_len], &mut bytes, kernel);
        assert!(decoded.is_ok_and(|decoded| *decoded == input[..len]));
    }

    /// Each text of up to 8 bytes that decodes is the text of its bytes, so no other text of those
    /// bytes decodes.
    #[kani::proof]
    #[kani::unwind(9)]
    fn each_text_that_decodes_is_canonical() {
        let profile = profile();
        let input: [u8; 8] = kani::any();
        let len = kani::any_where(|&len: &usize| len <= 8);
        let mut bytes = [0; BYTES_MAX];
        let kernel = |text: &[u8], out: &mut [u8]| portable::decode(profile, text, out);
        if let Ok(decoded) = profile.decode_by(&input[..len], &mut bytes, kernel) {
            let mut text = [0; 8];
            let text_len = encode(profile, decoded, &mut text);
            assert!(text[..text_len] == input[..len]);
        }
    }
}
