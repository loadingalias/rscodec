//! Lowercase hex ([RFC 4648, section 8]): each byte is two characters of `0`–`9` and `a`–`f`.
//!
//! The decoder takes only lowercase text, so each byte string has one text.
//!
//! The functions have two timing classes:
//!
//! - [`encode`] and [`decode`] are for public data, for example a digest. They run the fastest
//!   kernel of the CPU.
//! - [`encode_secret`] and [`decode_secret`] are for secret data, for example a key. They run only
//!   the kernels whose instructions have data independent timing (see [`Kernel`]). They take the
//!   proof [`Dit`] of a scope that holds `PSTATE.DIT`. After a refusal, `decode_secret` writes zero
//!   to each byte of the output that it wrote.
//!
//! The source of each kernel has no branch and no table load that depends on the bytes or the
//! characters. The time is to depend only on the lengths, and for an error, on its offset. A check
//! of the compiled code holds this rule in the shipped builds on x86_64 and aarch64 Linux. A build
//! whose target features add AVX-512 VBMI or AVX512-FP16 has no such check, and neither has
//! WebAssembly. An [`Error`] holds no byte of the input.
//!
//! ```
//! use rscodec::hex;
//!
//! // A digest is public.
//! let mut text = [0; 4];
//! assert_eq!(hex::encode(&[0xc0, 0xde], &mut text)?, b"c0de");
//! // A key is secret.
//! let mut key = [0; 2];
//! # // A build that makes no proof has no secret call: `with_dit` gives `DitUnknown`.
//! # if rscodec::with_dit(|_| ()).is_err() { return Ok(()); }
//! rscodec::with_dit(|dit| hex::decode_secret(dit, b"c0de", &mut key).map(|key| key.len()))??;
//! assert_eq!(key, [0xc0, 0xde]);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
//!
//! [RFC 4648, section 8]: https://www.rfc-editor.org/rfc/rfc4648#section-8

use core::fmt;

use crate::{
    Dit,
    cpu::{self, Caps, Timing},
    ct::{self, Class, Public, Secret},
};

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
#[inline]
#[must_use]
pub const fn encoded_len(len: usize) -> Option<usize> {
    len.checked_mul(2)
}

/// The number of bytes that `text` decodes to. It reads only the length.
///
/// # Errors
///
/// Returns [`ErrorKind::Length`] if `text` has an odd length.
#[inline]
pub const fn decoded_len(text: &[u8]) -> Result<usize, Error> {
    if text.len().is_multiple_of(2) {
        Ok(text.len() / 2)
    } else {
        Err(Error::new(ErrorKind::Length, text.len()))
    }
}

/// Encodes `bytes` into the start of `out`, and gives the text, which is ASCII. It runs the fastest
/// kernel of the CPU, for public bytes. Encode a secret with [`encode_secret`].
///
/// When `-C target-feature` fixes a SIMD kernel, that kernel runs inline at each length.
///
/// # Errors
///
/// Returns [`ErrorKind::Output`] if `out` is shorter than [`encoded_len`].
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn encode<'o>(bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = text_of(bytes, out)?;
    kernel::encode(Public, Backend(Kernel::chosen(Timing::Public)), bytes, out);
    Ok(out)
}

/// [`encode`] for secret bytes, for example a key. It runs only the kernels whose instructions have
/// data independent timing. The proof [`Dit`] shows that the scope of [`with_dit`](crate::with_dit)
/// holds `PSTATE.DIT`, so the function itself writes no DIT.
///
/// # Errors
///
/// As [`encode`].
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn encode_secret<'o>(_: &Dit, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = text_of(bytes, out)?;
    kernel::encode(Secret, Backend(Kernel::chosen(Timing::Secret)), bytes, out);
    Ok(out)
}

/// [`encode`] on the kernel of `backend`.
///
/// # Errors
///
/// As [`encode`].
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn encode_with<'o>(backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = text_of(bytes, out)?;
    kernel::encode(Public, backend, bytes, out);
    Ok(out)
}

/// The start of `out` that holds the text of `bytes`.
#[inline(always)]
fn text_of<'o>(bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    // A slice has at most `isize::MAX` bytes, so the length cannot overflow.
    let len = encoded_len(bytes.len()).unwrap_or(usize::MAX);
    out.get_mut(..len).ok_or_else(|| Error::new(ErrorKind::Output, len))
}

/// Decodes the lowercase hex `text` into the start of `out`, and gives the bytes. It runs the
/// fastest kernel of the CPU, for public text. Decode a secret with [`decode_secret`].
///
/// On x86_64, a result of up to 16 bytes comes from SSE2, inline and with no choice of kernel. When
/// `-C target-feature` fixes a SIMD kernel, that kernel runs inline at each length.
///
/// The bytes of `out` after the result, and all of `out` after an error, are unspecified.
///
/// # Errors
///
/// - [`ErrorKind::Length`] if `text` has an odd length.
/// - [`ErrorKind::Output`] if `out` is shorter than [`decoded_len`].
/// - [`ErrorKind::Symbol`] at the first byte of `text` that is not `0`–`9` or `a`–`f`. The decoder
///   reads all of `text` before it refuses it.
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn decode<'o>(text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = bytes_of(text, out)?;
    let valid = kernel::decode_chosen(Public, Timing::Public, text, out);
    checked(Public, valid, text, out)
}

/// [`decode`] for secret text, for example a key. It runs only the kernels whose instructions have
/// data independent timing, in the scope of the proof [`Dit`], as [`encode_secret`].
///
/// After a refusal for a symbol, each byte that it wrote to `out` is zero: the first
/// [`decoded_len`] bytes. After a refusal for the length or the output, it wrote no byte.
///
/// # Errors
///
/// As [`decode`].
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn decode_secret<'o>(_: &Dit, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = bytes_of(text, out)?;
    let valid = kernel::decode_chosen(Secret, Timing::Secret, text, out);
    checked(Secret, valid, text, out)
}

/// [`decode`] on the kernel of `backend`.
///
/// # Errors
///
/// As [`decode`].
// The short path runs inline in the caller: a call costs more than its work.
#[inline(always)]
pub fn decode_with<'o>(backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let out = bytes_of(text, out)?;
    let valid = kernel::decode(Public, backend, text, out);
    checked(Public, valid, text, out)
}

/// The start of `out` that holds the bytes of `text`.
#[inline(always)]
fn bytes_of<'o>(text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let len = decoded_len(text)?;
    out.get_mut(..len).ok_or_else(|| Error::new(ErrorKind::Output, len))
}

/// The result of a decode of `text` into `out`, in which `valid` tells whether each character is a
/// digit. In the secret `class`, a refusal writes zero to the output first.
#[inline(always)]
fn checked<'o>(class: impl Class, valid: bool, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    // One branch for the whole text.
    if !valid {
        return Err(refusal(class, text, out));
    }
    Ok(out)
}

/// The error of `text`, which has a byte that is not a digit. Its time depends only on the length.
/// In the secret `class`, it writes zero to `out` first.
///
/// It is cold and out of line, so that the decoder keeps its registers for the valid text: a call
/// on the hot path makes the caller save and load its vector registers (C109, Graviton4).
#[cold]
#[inline(never)]
fn refusal(class: impl Class, text: &[u8], out: &mut [u8]) -> Error {
    class.wipe(out);
    Error::new(
        ErrorKind::Symbol,
        ct::first_negative(text.iter().map(|&byte| portable::nibble(byte))),
    )
}

/// A kernel of the codec: its code for one instruction set. The variants depend on the target
/// architecture.
///
/// Each kernel uses only the instructions that Intel lists with data operand independent timing
/// (DOIT) and Arm lists with data independent timing (DIT), so [`encode_secret`] and
/// [`decode_secret`] can run it. WebAssembly has no timing list, so its kernels keep only the rule
/// of the source.
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
    /// AVX-512F and AVX-512BW, with 64-byte vectors. An encode of 64 KiB or more runs the AVX2
    /// code, with 32-byte vectors.
    #[cfg(target_arch = "x86_64")]
    Avx512Bw,
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
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512F, SSSE3};
        match self {
            Self::Portable => (Caps::NONE, Timing::Secret),
            #[cfg(target_arch = "x86_64")]
            Self::Ssse3 => (SSSE3, Timing::Secret),
            // The AVX2 kernels give the rest of each input to the SSSE3 kernels.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => (AVX2.union(SSSE3), Timing::Secret),
            // The AVX-512BW encoder runs the AVX2 code for a long input.
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bw => (AVX512F.union(AVX512BW).union(AVX2), Timing::Secret),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => (cpu::aarch64::NEON, Timing::Secret),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => (cpu::wasm32::SIMD128, Timing::Secret),
        }
    }

    crate::cpu::choice!(Portable);
}

/// A kernel that this CPU runs. [`encode_with`] and [`decode_with`] run it, for example to compare
/// the kernels.
///
/// Only [`Backend::new`] makes a `Backend`, and it checks the CPU first. Thus a `Backend` exists
/// only for a kernel whose features the CPU has, and no safe call can run a kernel that the CPU
/// does not support.
///
/// ```
/// use rscodec::hex::{self, Backend, Kernel};
///
/// // Each CPU runs the portable kernel. `Backend::new` gives `None` for a kernel whose features
/// // the CPU does not have.
/// let portable = Backend::new(Kernel::Portable).expect("each CPU runs the portable kernel");
/// let mut text = [0; 2];
/// assert_eq!(hex::encode_with(portable, &[0xc0], &mut text)?, b"c0");
/// # Ok::<(), hex::Error>(())
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
    use super::{Backend, Kernel, portable};
    use crate::{cpu::Timing, ct::Class};

    /// The largest result of the SSE2 short path of x86_64 that decodes: a call costs more than its
    /// work. An encode has no short path: the constants of an inline path load again after each
    /// call of the kernel in a loop of the caller, and with no short path the encode of 32 bytes
    /// took 0.73 times as long as faster-hex on the Xeon, against 0.90 (C109).
    #[cfg(target_arch = "x86_64")]
    const DECODE_INLINE_MAX: usize = 16;

    /// Whether the short path runs for `timing`: unless the build fixes a SIMD kernel, which then
    /// runs inline at each length.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn short(timing: Timing) -> bool {
        const fn fixes_simd(timing: Timing) -> bool {
            matches!(Kernel::fixed(timing), Some(index) if !matches!(Kernel::ALL[index], Kernel::Portable))
        }
        match timing {
            Timing::Public => const { !fixes_simd(Timing::Public) },
            Timing::Secret => const { !fixes_simd(Timing::Secret) },
        }
    }

    /// Decodes `text` into `out` in a function of `timing` with no backend argument: on x86_64, a
    /// short result on the short path, and otherwise on the chosen kernel.
    #[inline(always)]
    pub(super) fn decode_chosen(class: impl Class, timing: Timing, text: &[u8], out: &mut [u8]) -> bool {
        #[cfg(target_arch = "x86_64")]
        if short(timing) && out.len() <= DECODE_INLINE_MAX {
            // SAFETY: each x86_64 target has SSE2.
            return unsafe { x86::sse2::decode(class, text, out) };
        }
        decode(class, Backend(Kernel::chosen(timing)), text, out)
    }

    /// Encodes `bytes` into `out`, which has twice the length, on the kernel of `backend`.
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(unused_variables, reason = "only the SIMD kernels have a temporary array")
    )]
    #[inline(always)]
    pub(super) fn encode(class: impl Class, backend: Backend, bytes: &[u8], out: &mut [u8]) {
        match backend.0 {
            Kernel::Portable => portable::encode(bytes, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Ssse3 => unsafe { x86::ssse3::encode(class, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx2 => unsafe { x86::avx2::encode(class, bytes, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw => unsafe { x86::avx512bw::encode(class, bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::encode(class, bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::encode(class, bytes, out),
        }
    }

    /// Decodes `text` into `out`, which has half the length, on the kernel of `backend`, and gives
    /// whether each character is a digit.
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(unused_variables, reason = "only the SIMD kernels have a temporary array")
    )]
    #[inline(always)]
    pub(super) fn decode(class: impl Class, backend: Backend, text: &[u8], out: &mut [u8]) -> bool {
        match backend.0 {
            Kernel::Portable => portable::decode(text, out),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Ssse3 => unsafe { x86::ssse3::decode(class, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx2 => unsafe { x86::avx2::decode(class, text, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Avx512Bw => unsafe {
                // The caller has checked both lengths. Pass fixed blocks across the CPU feature boundary.
                let mut valid = true;
                match out.len() {
                    32..64 => {
                        crate::simd::overlapping(text, out, |src, dst| valid &= x86::avx512bw::decode32(src, dst))
                    }
                    64..128 => {
                        crate::simd::overlapping(text, out, |src, dst| valid &= x86::avx512bw::decode64(src, dst))
                    }
                    _ => return x86::avx512bw::decode(text, out),
                }
                valid
            },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::decode(class, text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::decode(class, text, out),
        }
    }
}

/// The kernel table of x86_64.
/// Encodes `bytes` into `out` with the `block` of a kernel of 16-byte vectors. Up to 64 bytes, it
/// runs straight-line size classes: one block at each end below 32 bytes, and two blocks at each end
/// from 32 bytes, so no class counts blocks. An input shorter than one vector runs on arrays on the
/// stack, and `long` encodes an input of more than 64 bytes.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(
    target_arch = "aarch64",
    expect(
        clippy::missing_safety_doc,
        reason = "the function is safe; only its target feature makes a call from code without the feature unsafe"
    )
)]
fn encode_steps(
    class: impl Class,
    bytes: &[u8],
    out: &mut [u8],
    block: impl Fn(&[u8; 16], &mut [u8; 32]),
    long: impl FnOnce(&[u8], &mut [u8]),
) {
    // The long input first: its test then costs a long input one compare.
    if bytes.len() > 64 {
        long(bytes, out);
    } else if bytes.len() >= 32 {
        crate::simd::ends::<32, 64>(bytes, out, |src, dst| {
            for (src, dst) in src.as_chunks().0.iter().zip(dst.as_chunks_mut().0) {
                block(src, dst);
            }
        });
    } else if bytes.len() >= 16 {
        crate::simd::ends(bytes, out, block);
    } else {
        crate::simd::short_blocks(class, bytes, out, 16, 0, |len| len * 2, block);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::Kernel;
    use crate::cpu::{
        Caps, Timing,
        x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ, SSSE3},
    };

    /// On the CPU of the C001 x86 host, a Xeon 6975P-C, both timing classes run the AVX-512BW
    /// kernel, which uses only instructions of the DOIT list of Intel.
    #[test]
    fn each_class_runs_avx512bw() {
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
        ]
        .into_iter()
        .fold(Caps::NONE, Caps::union);
        for class in [Timing::Public, Timing::Secret] {
            assert_eq!(Kernel::ALL[Kernel::fastest(c001, class)], Kernel::Avx512Bw);
        }
    }
}
