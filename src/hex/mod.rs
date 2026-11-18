//! Lowercase hex ([RFC 4648, section 8]): each byte is two characters of `0`–`9` and `a`–`f`.
//!
//! The decoder takes only lowercase text, so each byte string has one text. The source of each
//! backend has no branch and no table load that depends on the bytes or the characters, so that a
//! secret, for example a key, can go through the codec. The time is to depend only on the lengths,
//! and for an error, on its offset. A check of the compiled code and a timing test of this rule are
//! not complete yet. An [`Error`] holds no byte of the input.
//!
//! ```
//! use rscodec::hex;
//!
//! let mut bytes = [0; 2];
//! assert_eq!(hex::decode(b"c0de", &mut bytes)?, [0xc0, 0xde]);
//! let mut text = [0; 4];
//! assert_eq!(hex::encode(&bytes, &mut text)?, b"c0de");
//! # Ok::<(), rscodec::hex::Error>(())
//! ```
//!
//! [RFC 4648, section 8]: https://www.rfc-editor.org/rfc/rfc4648#section-8

use core::fmt;

use crate::{Backend, cpu, ct};

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod portable;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The cause of a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The text has an odd length. The offset is the length.
    Length,
    /// The byte at the offset is not `0`–`9` or `a`–`f`.
    Symbol,
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
            ErrorKind::Length => write!(f, "the hex text has an odd length, {offset}"),
            ErrorKind::Symbol => write!(
                f,
                "the hex text has a byte that is not a lowercase digit at offset {offset}"
            ),
            ErrorKind::Output => write!(f, "the output slice is shorter than the {offset} bytes of the result"),
        }
    }
}

impl core::error::Error for Error {}

/// The number of characters of the text of `len` bytes, or `None` if the number overflows `usize`.
#[must_use]
pub const fn encoded_len(len: usize) -> Option<usize> {
    len.checked_mul(2)
}

/// The number of bytes that `text` decodes to. It reads only the length.
///
/// # Errors
///
/// Returns [`ErrorKind::Length`] if `text` has an odd length.
pub const fn decoded_len(text: &[u8]) -> Result<usize, Error> {
    if text.len().is_multiple_of(2) {
        Ok(text.len() / 2)
    } else {
        Err(Error {
            kind: ErrorKind::Length,
            offset: text.len(),
        })
    }
}

/// Encodes `bytes` into the start of `out`, and gives the text, which is ASCII.
///
/// # Errors
///
/// Returns [`ErrorKind::Output`] if `out` is shorter than [`encoded_len`].
#[inline]
pub fn encode<'o>(bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    encode_with(Backend::detect(), bytes, out)
}

/// [`encode`] on `backend`.
///
/// # Errors
///
/// As [`encode`].
#[inline]
pub fn encode_with<'o>(backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    // A slice has at most `isize::MAX` bytes, so the length cannot overflow.
    let len = encoded_len(bytes.len()).unwrap_or(usize::MAX);
    let out = out.get_mut(..len).ok_or(Error {
        kind: ErrorKind::Output,
        offset: len,
    })?;
    cpu::constant_time(|| kernel::encode(backend, bytes, out));
    Ok(out)
}

/// Decodes the lowercase hex `text` into the start of `out`, and gives the bytes.
///
/// The bytes of `out` after the result, and all of `out` after an error, are unspecified.
///
/// # Errors
///
/// - [`ErrorKind::Length`] if `text` has an odd length.
/// - [`ErrorKind::Output`] if `out` is shorter than [`decoded_len`].
/// - [`ErrorKind::Symbol`] at the first byte of `text` that is not `0`–`9` or `a`–`f`. The decoder
///   reads all of `text` before it refuses it.
#[inline]
pub fn decode<'o>(text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    decode_with(Backend::detect(), text, out)
}

/// [`decode`] on `backend`.
///
/// # Errors
///
/// As [`decode`].
#[inline]
pub fn decode_with<'o>(backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let len = decoded_len(text)?;
    let out = out.get_mut(..len).ok_or(Error {
        kind: ErrorKind::Output,
        offset: len,
    })?;
    let valid = cpu::constant_time(|| kernel::decode(backend, text, out));
    // One branch for the whole text.
    if !valid {
        let offset = ct::first_negative(text.iter().map(|&byte| portable::nibble(byte)));
        return Err(Error {
            kind: ErrorKind::Symbol,
            offset,
        });
    }
    Ok(out)
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
    use super::portable;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use super::wasm32;
    #[cfg(target_arch = "x86_64")]
    use super::x86;
    use crate::{Backend, Isa};

    /// Encodes `bytes` into `out`, which has twice the length.
    #[inline]
    pub(super) fn encode(backend: Backend, bytes: &[u8], out: &mut [u8]) {
        match backend.isa() {
            Isa::Portable => portable::encode(bytes, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Ssse3 => unsafe { x86::ssse3::encode(bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx2 => unsafe { x86::avx2::encode(bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Bw | Isa::Avx512Vbmi => unsafe { x86::avx512bw::encode(bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Neon => unsafe { aarch64::encode(bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => wasm32::encode(bytes, out),
        }
    }

    /// Decodes `text` into `out`, which has half the length, and gives whether each character is a
    /// digit.
    #[inline]
    pub(super) fn decode(backend: Backend, text: &[u8], out: &mut [u8]) -> bool {
        match backend.isa() {
            Isa::Portable => portable::decode(text, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Ssse3 => unsafe { x86::ssse3::decode(text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx2 => unsafe { x86::avx2::decode(text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Avx512Bw | Isa::Avx512Vbmi => unsafe { x86::avx512bw::decode(text, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this instruction set exists only on a CPU that has it.
            Isa::Neon => unsafe { aarch64::decode(text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => wasm32::decode(text, out),
        }
    }
}

/// The proofs of the portable backend, for `cargo kani`.
#[cfg(kani)]
mod proofs {
    use super::{decoded_len, portable};

    /// The largest input of the proofs.
    const BYTES_MAX: usize = 4;

    /// Each input of up to `BYTES_MAX` bytes decodes from its text to itself, and each text that
    /// decodes is the text of its bytes.
    #[kani::proof]
    #[kani::unwind(9)]
    fn each_input_round_trips_and_each_text_is_canonical() {
        let input: [u8; BYTES_MAX] = kani::any();
        let len = kani::any_where(|&len: &usize| len <= BYTES_MAX);
        let mut text = [0; 2 * BYTES_MAX];
        portable::encode(&input[..len], &mut text[..2 * len]);
        let mut bytes = [0; BYTES_MAX];
        assert!(portable::decode(&text[..2 * len], &mut bytes[..len]));
        assert!(bytes[..len] == input[..len]);

        let chars: [u8; 2 * BYTES_MAX] = kani::any();
        let chars_len = kani::any_where(|&len: &usize| len <= 2 * BYTES_MAX);
        if let Ok(len) = decoded_len(&chars[..chars_len])
            && portable::decode(&chars[..chars_len], &mut bytes[..len])
        {
            portable::encode(&bytes[..len], &mut text[..chars_len]);
            assert!(text[..chars_len] == chars[..chars_len]);
        }
    }
}
