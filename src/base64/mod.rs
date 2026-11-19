//! base64 ([RFC 4648, section 4]) with padding, and base64url ([RFC 4648, section 5]) without
//! padding.
//!
//! Each [`Profile`] decodes only the canonical text of its bytes, so each byte string has one text:
//! only the characters of its alphabet, no whitespace, the padding of the profile, and zero bits
//! after the last byte. The encoder writes that text.
//!
//! The functions have two timing classes:
//!
//! - [`Profile::encode`] and [`Profile::decode`] are for public data, for example a digest or a
//!   signature. They run the fastest kernel of the CPU.
//! - [`Profile::encode_secret`] and [`Profile::decode_secret`] are for secret data, for example a
//!   key or a bearer token. They run only the kernels whose instructions have data independent
//!   timing (see [`Kernel`]). They take the proof [`Dit`] of a scope that holds `PSTATE.DIT`. After a
//!   refusal, `decode_secret` writes zero to each byte of the output that it wrote.
//!
//! The source of each kernel has no branch and no table load that depends on the bytes or the
//! characters. The time is to depend only on the lengths, and for an error, on its offset. A check
//! of the compiled code and a timing test of this rule are not complete yet. An [`Error`] holds no
//! byte of the input.
//!
//! ```
//! use rscodec::base64::Profile;
//!
//! // A signature is public.
//! let mut text = [0; 4];
//! assert_eq!(Profile::Url.encode(&[0xfb, 0xff, 0xbf], &mut text)?, b"-_-_");
//! // A key is secret.
//! let mut key = [0; 3];
//! rscodec::with_dit(|dit| Profile::Url.decode_secret(dit, b"-_-_", &mut key).map(|key| key.len()))??;
//! assert_eq!(key, [0xfb, 0xff, 0xbf]);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
//!
//! [RFC 4648, section 4]: https://www.rfc-editor.org/rfc/rfc4648#section-4
//! [RFC 4648, section 5]: https://www.rfc-editor.org/rfc/rfc4648#section-5

use core::fmt;

use crate::{
    Dit,
    cpu::{self, Caps, Kept, Timing, caps_static},
    ct::{self, Class, Public, Secret},
};

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

    /// Encodes `bytes` into the start of `out`, and gives the text, which is ASCII. It runs the
    /// fastest kernel of the CPU, for public bytes. Encode a secret with
    /// [`Profile::encode_secret`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Output`] if `out` is shorter than [`Profile::encoded_len`].
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn encode<'o>(self, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Public, Backend(Kernel::chosen(Timing::Public)), bytes, out)
    }

    /// [`Profile::encode`] for secret bytes, for example a key. It runs only the kernels whose
    /// instructions have data independent timing. The proof [`Dit`] shows that the scope of
    /// [`with_dit`](crate::with_dit) holds `PSTATE.DIT`, so the function itself writes no DIT.
    ///
    /// # Errors
    ///
    /// As [`Profile::encode`].
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn encode_secret<'o>(self, _: &Dit, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Secret, Backend(Kernel::chosen(Timing::Secret)), bytes, out)
    }

    /// [`Profile::encode`] on the kernel of `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::encode`].
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn encode_with<'o>(self, backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Public, backend, bytes, out)
    }

    /// Encodes `bytes` on the kernel of `backend` in `class`. The kernel writes the characters
    /// before the padding.
    #[inline(always)]
    fn encode_by<'o>(
        self,
        class: impl Class,
        backend: Backend,
        bytes: &[u8],
        out: &'o mut [u8],
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
        kernel::encode(class, backend, self, bytes, text);
        Ok(out)
    }

    /// Decodes the canonical text `text` into the start of `out`, and gives the bytes. It runs the
    /// fastest kernel of the CPU, for public text. Decode a secret with
    /// [`Profile::decode_secret`].
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
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn decode<'o>(self, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Public, Backend(Kernel::chosen(Timing::Public)), text, out)
    }

    /// [`Profile::decode`] for secret text, for example a bearer token. It runs only the kernels
    /// whose instructions have data independent timing, in the scope of the proof [`Dit`], as
    /// [`Profile::encode_secret`].
    ///
    /// After a refusal for a symbol or for the trailing bits, each byte that it wrote to `out` is
    /// zero: the first [`Profile::decoded_len`] bytes. After a refusal for the length or the
    /// output, it wrote no byte.
    ///
    /// # Errors
    ///
    /// As [`Profile::decode`].
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn decode_secret<'o>(self, _: &Dit, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Secret, Backend(Kernel::chosen(Timing::Secret)), text, out)
    }

    /// [`Profile::decode`] on the kernel of `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::decode`].
    // The short path runs inline in the caller: a call costs more than its work.
    #[inline(always)]
    pub fn decode_with<'o>(self, backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Public, backend, text, out)
    }

    /// Decodes `text` on the kernel of `backend` in `class`. The kernel decodes the characters
    /// before the padding and gives whether each is in the alphabet. In the secret class, a refusal
    /// of the characters writes zero to the output first.
    #[inline(always)]
    fn decode_by<'o>(
        self,
        class: impl Class,
        backend: Backend,
        text: &[u8],
        out: &'o mut [u8],
    ) -> Result<&'o mut [u8], Error> {
        let symbols = self.symbols(text)?;
        let len = bytes_len(symbols.len());
        let out = out.get_mut(..len).ok_or(Error {
            kind: ErrorKind::Output,
            offset: len,
        })?;
        let valid = kernel::decode(class, backend, self, symbols, out);
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
            class.wipe(out);
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

/// A kernel of the codec: its code for one instruction set. The variants depend on the target
/// architecture.
///
/// Each kernel has a timing class. Each kernel but `Kernel::Avx512Vbmi` of x86_64 uses only the
/// instructions that Intel lists with data operand independent timing (DOIT) and Arm lists with
/// data independent timing (DIT), so [`Profile::encode_secret`] and [`Profile::decode_secret`]
/// can run it. WebAssembly has no timing list, so its kernels keep only the rule of the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// Scalar arithmetic with no table. It runs on each CPU.
    Portable,
    /// SSSE3, with 16-byte vectors.
    #[cfg(target_arch = "x86_64")]
    Ssse3,
    /// AVX2, with 32-byte vectors.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512F and AVX-512BW, with 64-byte vectors. The secret functions run it where the CPU has
    /// AVX-512BW.
    #[cfg(target_arch = "x86_64")]
    Avx512Bw,
    /// AVX-512BW and AVX-512 VBMI, with byte permutes that the DOIT list of Intel does not hold.
    /// Only the public functions run it.
    #[cfg(target_arch = "x86_64")]
    Avx512Vbmi,
    /// Neon (Advanced SIMD), with 16-byte vectors.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// WebAssembly SIMD, with 16-byte vectors. It exists only with `-C target-feature=+simd128`,
    /// because WebAssembly has no run-time detection.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

impl Kernel {
    /// Each kernel of the target architecture, the fastest first.
    pub const ALL: &[Self] = &[
        #[cfg(target_arch = "x86_64")]
        Self::Avx512Vbmi,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512Bw,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        #[cfg(target_arch = "x86_64")]
        Self::Ssse3,
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Self::Simd128,
        Self::Portable,
    ];

    /// The CPU features that the kernel needs, and its timing class.
    const fn needs(self) -> (Caps, Timing) {
        #[cfg(target_arch = "x86_64")]
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI, SSSE3};
        match self {
            Self::Portable => (Caps::NONE, Timing::Secret),
            #[cfg(target_arch = "x86_64")]
            Self::Ssse3 => (SSSE3, Timing::Secret),
            // The AVX2 kernels give the rest of each input to the SSSE3 kernels.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => (AVX2.union(SSSE3), Timing::Secret),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bw => (AVX512F.union(AVX512BW), Timing::Secret),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Vbmi => (AVX512F.union(AVX512BW).union(AVX512VBMI), Timing::Public),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => (cpu::aarch64::NEON, Timing::Secret),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => (cpu::wasm32::SIMD128, Timing::Secret),
        }
    }

    /// The index in [`Kernel::ALL`] of the fastest kernel of `class` that a CPU with `caps` runs.
    /// The last kernel, the portable kernel, runs on each CPU and in each class.
    const fn fastest(caps: Caps, class: Timing) -> usize {
        let mut index = 0;
        while index + 1 < Self::ALL.len() {
            let (features, timing) = Self::ALL[index].needs();
            if caps.has(features) && class.runs(timing) {
                break;
            }
            index += 1;
        }
        index
    }

    /// The index of the kernel of `class` that the build fixes: if the build has no run-time
    /// detection, or if `caps_static()` holds the features of the fastest kernel of the class.
    const fn fixed(class: Timing) -> Option<usize> {
        let index = Self::fastest(caps_static(), class);
        if !cpu::DETECTS || index == Self::fastest(Caps::ALL, class) {
            Some(index)
        } else {
            None
        }
    }

    /// The kernel of the functions of `class` with no backend argument. If the build does not fix
    /// it, the first call chooses it, and each later call reads the choice.
    #[inline(always)]
    fn chosen(class: Timing) -> Self {
        const PUBLIC: Option<usize> = Kernel::fixed(Timing::Public);
        const SECRET: Option<usize> = Kernel::fixed(Timing::Secret);
        static KEPT: [Kept; 2] = [Kept::new(), Kept::new()];
        let (fixed, kept) = match class {
            Timing::Public => (PUBLIC, &KEPT[0]),
            Timing::Secret => (SECRET, &KEPT[1]),
        };
        let index = fixed.unwrap_or_else(|| kept.index(|| Self::fastest(cpu::caps(), class)));
        Self::ALL.get(index).copied().unwrap_or(Self::Portable)
    }
}

/// A kernel that this CPU runs. [`Profile::encode_with`] and [`Profile::decode_with`] run it, for
/// example to compare the kernels.
///
/// Only [`Backend::new`] makes a `Backend`, and it checks the CPU first. Thus a `Backend` exists
/// only for a kernel whose features the CPU has, and no safe call can run a kernel that the CPU
/// does not support.
///
/// ```
/// use rscodec::base64::{Backend, Kernel, Profile};
///
/// // Each CPU runs the portable kernel. `Backend::new` gives `None` for a kernel whose features
/// // the CPU does not have.
/// let portable = Backend::new(Kernel::Portable).expect("each CPU runs the portable kernel");
/// let mut text = [0; 4];
/// assert_eq!(Profile::Url.encode_with(portable, b"\xfb\xff\xbf", &mut text)?, b"-_-_");
/// # Ok::<(), rscodec::base64::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Backend(Kernel);

impl Backend {
    /// The backend of `kernel`, or `None` if this CPU does not have the features of `kernel`.
    #[inline]
    #[must_use]
    pub fn new(kernel: Kernel) -> Option<Self> {
        cpu::has(kernel.needs().0).then_some(Self(kernel))
    }
}

/// The calls of each kernel.
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    expect(
        unsafe_code,
        reason = "a `Backend` holds only a kernel whose features the CPU has, so its target-feature functions can run"
    )
)]
mod kernel {
    #[cfg(target_arch = "aarch64")]
    use super::aarch64;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use super::wasm32;
    #[cfg(target_arch = "x86_64")]
    use super::x86;
    use super::{Backend, Kernel, Profile, portable};
    use crate::ct::Class;

    /// Encodes `bytes` into the characters `out`, with no padding.
    #[inline(always)]
    pub(super) fn encode(class: impl Class, backend: Backend, profile: Profile, bytes: &[u8], out: &mut [u8]) {
        match backend.0 {
            Kernel::Portable => portable::encode(class, profile, bytes, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Ssse3 => unsafe { x86::ssse3::encode(class, profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx2 => unsafe { x86::avx2::encode(class, profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw if bytes.len() <= x86::ONE_MAX => unsafe {
                x86::avx512bw::encode_one(profile, bytes, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw => unsafe { x86::avx512bw::encode(profile, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi if bytes.len() <= x86::ONE_MAX => unsafe {
                x86::avx512vbmi::encode_one(profile, bytes, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi => unsafe { x86::avx512vbmi::encode(profile, bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::encode(class, profile, bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::encode(class, profile, bytes, out),
        }
    }

    /// Decodes the characters `text`, with no padding, into `out`, and gives whether each is in the
    /// alphabet.
    #[inline(always)]
    pub(super) fn decode(class: impl Class, backend: Backend, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
        match backend.0 {
            Kernel::Portable => portable::decode(class, profile, text, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Ssse3 => unsafe { x86::ssse3::decode(class, profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx2 => unsafe { x86::avx2::decode(class, profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw if out.len() <= x86::ONE_MAX => unsafe { x86::avx512bw::decode_one(profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw => unsafe { x86::avx512bw::decode(profile, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi if out.len() <= x86::ONE_MAX => unsafe {
                x86::avx512vbmi::decode_one(profile, text, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi => unsafe { x86::avx512vbmi::decode(profile, text, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::decode(class, profile, text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::decode(class, profile, text, out),
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

/// The kernel table of x86_64.
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::Kernel;
    use crate::cpu::{
        Caps, Timing,
        x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ, SSSE3},
    };

    /// The kernel of each timing class on a CPU with `features`.
    fn chosen(features: &[Caps]) -> [Kernel; 2] {
        let caps = features.iter().copied().fold(Caps::NONE, Caps::union);
        [Timing::Public, Timing::Secret].map(|class| Kernel::ALL[Kernel::fastest(caps, class)])
    }

    /// On the CPU of the C001 x86 host, a Xeon 6975P-C, the public class runs VBMI and the secret
    /// class runs AVX-512BW, because the VBMI permutes are not on the DOIT list of Intel. Without
    /// VBMI, both classes run AVX-512BW ([CODEC Base-N]).
    #[test]
    fn the_public_class_runs_vbmi_and_the_secret_class_runs_avx512bw() {
        let c001 = [
            SSSE3,
            AVX2,
            BMI1,
            BMI2,
            PCLMULQDQ,
            AVX512F,
            AVX512BW,
            AVX512VL,
            AVX512VBMI,
            AVX512VBMI2,
        ];
        assert_eq!(chosen(&c001), [Kernel::Avx512Vbmi, Kernel::Avx512Bw]);
        assert_eq!(chosen(&c001[..8]), [Kernel::Avx512Bw, Kernel::Avx512Bw]);
        assert_eq!(chosen(&c001[..5]), [Kernel::Avx2, Kernel::Avx2]);
        assert_eq!(chosen(&[]), [Kernel::Portable, Kernel::Portable]);
    }
}
