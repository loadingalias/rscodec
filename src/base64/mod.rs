//! Padded [base64](https://www.rfc-editor.org/rfc/rfc4648#section-4) and unpadded
//! [base64url](https://www.rfc-editor.org/rfc/rfc4648#section-5).
//!
//! Each [`Profile`] accepts only its alphabet, required padding, and zero trailing bits.
//! Whitespace is an error. [`Profile::encode`] and [`Profile::decode`] handle public data.
//! Use [`Profile::encode_secret`] and [`Profile::decode_secret`] for keys and bearer tokens.
//! Secret decode clears the result region after a symbol or trailing-bit error.
//! [Timing evidence and limits](https://github.com/loadingalias/rscodec/blob/main/docs/content/base64-and-hex.md#timing-evidence)
//!
//! ```
//! use rscodec::base64::Profile;
//!
//! let mut text = [0; 4];
//! assert_eq!(Profile::Url.encode(&[0xfb, 0xff, 0xbf], &mut text)?, b"-_-_");
//! # Ok::<(), rscodec::base64::Error>(())
//! ```

use core::fmt;

use crate::{
    Dit,
    cpu::{self, Caps, Timing},
    ct::{Class, Public, Secret},
};

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use crate::simd::{short_blocks, write16};

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
    kind: ErrorKind,
    offset: usize,
}

impl Error {
    /// The refusal of `kind` at `offset`.
    #[inline]
    #[must_use]
    pub const fn new(kind: ErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    /// The cause.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The offset in the text, or for [`ErrorKind::Output`], the length of the result.
    #[inline]
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
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
    #[inline]
    #[must_use]
    pub const fn encoded_len(self, len: usize) -> Option<usize> {
        match self.text_lens(len) {
            Some((_, len)) => Some(len),
            None => None,
        }
    }

    const fn text_lens(self, len: usize) -> Option<(usize, usize)> {
        let rest = len % 3;
        let Some(full) = (len / 3).checked_mul(4) else {
            return None;
        };
        let Some(symbols) = full.checked_add(rest + rest.div_ceil(2)) else {
            return None;
        };
        match self {
            Self::Url => Some((symbols, symbols)),
            Self::Standard if rest == 0 => Some((symbols, symbols)),
            Self::Standard => match full.checked_add(4) {
                Some(len) => Some((symbols, len)),
                None => None,
            },
        }
    }

    /// The number of bytes that `text` decodes to. It reads only the length and the padding.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Length`] if no text of the profile has the length of `text`.
    #[inline]
    pub const fn decoded_len(self, text: &[u8]) -> Result<usize, Error> {
        match self.symbols(text) {
            Ok(symbols) => Ok(bytes_len(symbols.len())),
            Err(error) => Err(error),
        }
    }

    /// Encode public bytes into the start of `out` and return the ASCII text. For keys, use [`Profile::encode_secret`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Output`] gives the required length if the buffer is too short.
    #[inline(always)]
    pub fn encode<'o>(self, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Public, Backend(Kernel::chosen(Timing::Public)), bytes, out)
    }

    /// Encode secret bytes inside [`with_dit`](crate::with_dit).
    ///
    /// # Errors
    ///
    /// As [`Profile::encode`].
    #[inline(always)]
    pub fn encode_secret<'o>(self, _: &Dit, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Secret, Backend(Kernel::chosen(Timing::Secret)), bytes, out)
    }

    /// [`Profile::encode`] on the kernel of `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::encode`].
    #[inline(always)]
    pub fn encode_with<'o>(self, backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.encode_by(Public, backend, bytes, out)
    }

    #[inline(always)]
    fn encode_by<'o>(
        self,
        class: impl Class,
        backend: Backend,
        bytes: &[u8],
        out: &'o mut [u8],
    ) -> Result<&'o mut [u8], Error> {
        // A slice has at most `isize::MAX` bytes, so the lengths cannot overflow.
        let (symbols, len) = self.text_lens(bytes.len()).unwrap_or((usize::MAX, usize::MAX));
        let out = out.get_mut(..len).ok_or_else(|| Error::new(ErrorKind::Output, len))?;
        let (text, padding) = out.split_at_mut_checked(symbols).unwrap_or_default();
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

    /// Decode public text into the start of `out` and return the decoded bytes. Use [`Profile::decode_secret`] for secrets.
    ///
    /// # Errors
    ///
    /// Length and capacity errors leave `out` untouched. A symbol error gives the first invalid byte offset.
    /// [`ErrorKind::TrailingBits`] identifies a final character with nonzero unused bits.
    /// After a character error, the result region is unspecified.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscodec::base64::{Error, ErrorKind, Profile};
    ///
    /// let mut out = [0; 3];
    /// assert_eq!(Profile::Url.decode(b"-_-_", &mut out)?, [0xfb, 0xff, 0xbf]);
    /// assert_eq!(Profile::Url.decode(b"+/+/", &mut out), Err(Error::new(ErrorKind::Symbol, 0)));
    /// # Ok::<(), Error>(())
    /// ```
    #[inline(always)]
    pub fn decode<'o>(self, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Public, Backend(Kernel::chosen(Timing::Public)), text, out)
    }

    /// Decode secret text inside [`with_dit`](crate::with_dit). A symbol or trailing-bit error clears the result region.
    /// Length and capacity errors leave `out` untouched.
    ///
    /// # Errors
    ///
    /// As [`Profile::decode`].
    #[inline(always)]
    pub fn decode_secret<'o>(self, _: &Dit, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Secret, Backend(Kernel::chosen(Timing::Secret)), text, out)
    }

    /// [`Profile::decode`] on the kernel of `backend`.
    ///
    /// # Errors
    ///
    /// As [`Profile::decode`].
    #[inline(always)]
    pub fn decode_with<'o>(self, backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
        self.decode_by(Public, backend, text, out)
    }

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
        let out = out.get_mut(..len).ok_or_else(|| Error::new(ErrorKind::Output, len))?;
        if !kernel::decode(class, backend, self, symbols, out) {
            return Err(self.refusal(class, symbols, out));
        }
        Ok(out)
    }

    const fn symbols(self, text: &[u8]) -> Result<&[u8], Error> {
        let length = Error::new(ErrorKind::Length, text.len());
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

    #[cold]
    #[inline(never)]
    fn refusal(self, class: impl Class, symbols: &[u8], out: &mut [u8]) -> Error {
        class.wipe(out);
        let first = portable::first_invalid(self, symbols);
        if first < symbols.len() {
            Error::new(ErrorKind::Symbol, first)
        } else {
            Error::new(ErrorKind::TrailingBits, symbols.len().saturating_sub(1))
        }
    }
}

const fn bytes_len(len: usize) -> usize {
    len / 4 * 3 + (len % 4).saturating_sub(1)
}

/// A base64 implementation available on this architecture.
/// AVX-512 VBMI handles public data only. Secret dispatch selects AVX-512BW on such CPUs.
/// Timing guarantees depend on the compiled binary and hardware mode.
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

    crate::cpu::choice!(Portable);
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
    #[cfg(target_arch = "x86_64")]
    use crate::cpu::Timing;
    use crate::ct::Class;

    #[inline(always)]
    pub(super) fn encode<C: Class>(class: C, backend: Backend, profile: Profile, bytes: &[u8], out: &mut [u8]) {
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
            Kernel::Avx512Vbmi if C::TIMING.runs(Timing::Public) && bytes.len() <= x86::ONE_MAX => unsafe {
                x86::avx512vbmi::encode_one(profile, bytes, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi if C::TIMING.runs(Timing::Public) => unsafe {
                x86::avx512vbmi::encode(profile, bytes, out)
            },
            // Intel excludes VBMI permutes from DOIT. A VBMI CPU also supports AVX-512BW.
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of either kernel exists only on a CPU that has AVX-512BW.
            Kernel::Avx512Bw | Kernel::Avx512Vbmi if bytes.len() <= x86::ONE_MAX => unsafe {
                x86::avx512bw::encode_one(profile, bytes, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of either kernel exists only on a CPU that has AVX-512BW.
            Kernel::Avx512Bw | Kernel::Avx512Vbmi => unsafe { x86::avx512bw::encode(profile, bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::encode(class, profile, bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::encode(class, profile, bytes, out),
        }
    }

    #[inline(always)]
    pub(super) fn decode<C: Class>(class: C, backend: Backend, profile: Profile, text: &[u8], out: &mut [u8]) -> bool {
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
            Kernel::Avx512Vbmi if C::TIMING.runs(Timing::Public) && out.len() <= x86::ONE_MAX => unsafe {
                x86::avx512vbmi::decode_one(profile, text, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Vbmi if C::TIMING.runs(Timing::Public) => unsafe {
                x86::avx512vbmi::decode(profile, text, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of either kernel exists only on a CPU that has AVX-512BW.
            Kernel::Avx512Bw | Kernel::Avx512Vbmi if out.len() <= x86::ONE_MAX => unsafe {
                x86::avx512bw::decode_one(profile, text, out)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of either kernel exists only on a CPU that has AVX-512BW.
            Kernel::Avx512Bw | Kernel::Avx512Vbmi => unsafe { x86::avx512bw::decode(profile, text, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::decode(class, profile, text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::decode(class, profile, text, out),
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", doc = "# Safety\nThe CPU must support Neon.")]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
fn encode_steps(
    class: impl Class,
    bytes: &[u8],
    last: &[u8; 16],
    out: &mut [u8],
    to_end: impl Fn(&[u8; 16], usize) -> [u8; 16],
    block: impl Fn(&[u8; 16], u8, &mut [u8; 16]),
) {
    let chars = 16 - (out.len().wrapping_neg() & 15);
    let skip = 16 - 3 * chars / 4;
    let mut dst = [0; 16];
    block(last, u8::try_from(skip).unwrap_or_default(), &mut dst);
    // `out` has at least 22 characters. A slice pattern gives the last chunk with no `Option`, and
    // thus with no test of its address for null.
    if let [.., end] = out.as_rchunks_mut().1 {
        *end = to_end(&dst, chars);
    }
    class.clear(&mut dst);
    let mut outs = out.as_chunks_mut().0.iter_mut();
    let mut rest = bytes;
    while let Some(src) = rest.first_chunk()
        && let Some(dst) = outs.next()
    {
        block(src, 0, dst);
        rest = rest.get(12..).unwrap_or_default();
    }
    if let Some(skip) = 16_usize.checked_sub(rest.len()).filter(|&skip| skip < 4)
        && let Some(dst) = outs.next()
    {
        block(last, u8::try_from(skip).unwrap_or_default(), dst);
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", doc = "# Safety\nThe CPU must support Neon.")]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
fn decode_steps(
    class: impl Class,
    text: &[u8],
    out: &mut [u8],
    ends: (
        impl Fn(&[u8; 16], usize, u8) -> [u8; 16],
        impl Fn(&[u8; 16], usize) -> u8,
    ),
    mut block: impl FnMut(&[u8; 16], &mut [u8; 16]),
) -> u8 {
    let (from_end, byte_at) = ends;
    let Some(last) = text.last_chunk::<16>() else {
        return short_blocks::<16, 16>(class, text, out, 16, b'A', bytes_len, block);
    };
    let (chunks, rest) = text.as_chunks::<16>();
    for (src, at) in chunks.iter().zip((0..).step_by(12)) {
        if let Some(dst) = out.get_mut(at..).and_then(<[u8]>::first_chunk_mut) {
            block(src, dst);
        } else if let Some(bytes) = out.get_mut(at..at + 12) {
            let mut dst = [0; 16];
            block(src, &mut dst);
            write16(bytes, &dst);
            class.clear(&mut dst);
        }
    }
    if rest.is_empty() {
        return 0;
    }
    let (mut src, mut dst) = (from_end(last, rest.len(), b'A'), [0; 16]);
    block(&src, &mut dst);
    let rest_out = out.get_mut(12 * chunks.len()..).unwrap_or_default();
    let after = byte_at(&dst, rest_out.len());
    write16(rest_out, &dst);
    class.clear(&mut src);
    class.clear(&mut dst);
    after
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
struct Tables {
    #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
    nibbles: Nibbles,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    alphabet: [u8; 64],
    #[cfg(target_arch = "x86_64")]
    sextets: [[u8; 64]; 2],
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
    roll: [u8; 16],
    special: u8,
    special_index: u8,
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
                sextets[index as usize] = 0x80 | crate::ct::low(sextet);
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
                sextets[(byte / 64) as usize][(byte % 64) as usize] = crate::ct::low(sextet);
            }
            byte += 1;
        }
        sextets
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::Kernel;
    use crate::cpu::{
        Caps, Timing,
        x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ, SSSE3},
    };

    fn chosen(features: &[Caps]) -> [Kernel; 2] {
        let caps = features.iter().copied().fold(Caps::NONE, Caps::union);
        [Timing::Public, Timing::Secret].map(|class| Kernel::ALL[Kernel::fastest(caps, class)])
    }

    /// Intel's DOIT list excludes VBMI permutes. Secret dispatch must use AVX-512BW on a VBMI CPU.
    #[test]
    fn the_public_class_runs_vbmi_and_the_secret_class_runs_avx512bw() {
        let xeon = [
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
        assert_eq!(chosen(&xeon), [Kernel::Avx512Vbmi, Kernel::Avx512Bw]);
        assert_eq!(chosen(&xeon[..8]), [Kernel::Avx512Bw, Kernel::Avx512Bw]);
        assert_eq!(chosen(&xeon[..5]), [Kernel::Avx2, Kernel::Avx2]);
        assert_eq!(chosen(&[]), [Kernel::Portable, Kernel::Portable]);
    }
}
