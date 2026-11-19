//! Lowercase hex ([RFC 4648, section 8]): each byte is two characters of `0`–`9` and `a`–`f`.
//!
//! The decoder takes only lowercase text, so each byte string has one text.
//!
//! The functions have two timing classes:
//!
//! - [`encode`] and [`decode`] are for public data, for example a digest. They run the fastest
//!   kernel of the CPU.
//! - [`encode_secret`] and [`decode_secret`] are for secret data, for example a key. They run only
//!   the kernels whose instructions have data independent timing (see [`Kernel`]). On aarch64, they
//!   set `PSTATE.DIT` for the call. After a refusal, `decode_secret` writes zero to each byte of the
//!   output that it wrote.
//!
//! The source of each kernel has no branch and no table load that depends on the bytes or the
//! characters. The time is to depend only on the lengths, and for an error, on its offset. A check
//! of the compiled code and a timing test of this rule are not complete yet. An [`Error`] holds no
//! byte of the input.
//!
//! ```
//! use rscodec::hex;
//!
//! // A digest is public.
//! let mut text = [0; 4];
//! assert_eq!(hex::encode(&[0xc0, 0xde], &mut text)?, b"c0de");
//! // A key is secret.
//! let mut key = [0; 2];
//! assert_eq!(hex::decode_secret(b"c0de", &mut key)?, [0xc0, 0xde]);
//! # Ok::<(), rscodec::hex::Error>(())
//! ```
//!
//! [RFC 4648, section 8]: https://www.rfc-editor.org/rfc/rfc4648#section-8

use core::fmt;

use crate::{
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

/// Encodes `bytes` into the start of `out`, and gives the text, which is ASCII. It runs the fastest
/// kernel of the CPU, for public bytes. Encode a secret with [`encode_secret`].
///
/// # Errors
///
/// Returns [`ErrorKind::Output`] if `out` is shorter than [`encoded_len`].
#[inline]
pub fn encode<'o>(bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    encode_with(Backend(Kernel::chosen(Timing::Public)), bytes, out)
}

/// [`encode`] for secret bytes, for example a key. It runs only the kernels whose instructions have
/// data independent timing. On aarch64, it sets `PSTATE.DIT` for the call.
///
/// # Errors
///
/// As [`encode`].
#[inline]
pub fn encode_secret<'o>(bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let backend = Backend(Kernel::chosen(Timing::Secret));
    cpu::data_independent_timing(|| encode_by(Secret, backend, bytes, out))
}

/// [`encode`] on the kernel of `backend`.
///
/// # Errors
///
/// As [`encode`].
#[inline]
pub fn encode_with<'o>(backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    encode_by(Public, backend, bytes, out)
}

/// Encodes `bytes` on `backend`, in `class`.
#[inline(always)]
fn encode_by<'o>(class: impl Class, backend: Backend, bytes: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    // A slice has at most `isize::MAX` bytes, so the length cannot overflow.
    let len = encoded_len(bytes.len()).unwrap_or(usize::MAX);
    let out = out.get_mut(..len).ok_or(Error {
        kind: ErrorKind::Output,
        offset: len,
    })?;
    kernel::encode(class, backend, bytes, out);
    Ok(out)
}

/// Decodes the lowercase hex `text` into the start of `out`, and gives the bytes. It runs the
/// fastest kernel of the CPU, for public text. Decode a secret with [`decode_secret`].
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
    decode_with(Backend(Kernel::chosen(Timing::Public)), text, out)
}

/// [`decode`] for secret text, for example a key. It runs only the kernels whose instructions have
/// data independent timing. On aarch64, it sets `PSTATE.DIT` for the call.
///
/// After a refusal for a symbol, each byte that it wrote to `out` is zero: the first
/// [`decoded_len`] bytes. After a refusal for the length or the output, it wrote no byte.
///
/// # Errors
///
/// As [`decode`].
#[inline]
pub fn decode_secret<'o>(text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let backend = Backend(Kernel::chosen(Timing::Secret));
    cpu::data_independent_timing(|| decode_by(Secret, backend, text, out))
}

/// [`decode`] on the kernel of `backend`.
///
/// # Errors
///
/// As [`decode`].
#[inline]
pub fn decode_with<'o>(backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    decode_by(Public, backend, text, out)
}

/// Decodes `text` on `backend`, in `class`. In the secret class, a refusal of the characters writes
/// zero to the output first.
#[inline(always)]
fn decode_by<'o>(class: impl Class, backend: Backend, text: &[u8], out: &'o mut [u8]) -> Result<&'o mut [u8], Error> {
    let len = decoded_len(text)?;
    let out = out.get_mut(..len).ok_or(Error {
        kind: ErrorKind::Output,
        offset: len,
    })?;
    // One branch for the whole text.
    if !kernel::decode(class, backend, text, out) {
        class.wipe(out);
        let offset = ct::first_negative(text.iter().map(|&byte| portable::nibble(byte)));
        return Err(Error {
            kind: ErrorKind::Symbol,
            offset,
        });
    }
    Ok(out)
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
    /// AVX-512F and AVX-512BW, with 64-byte vectors.
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
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bw => (AVX512F.union(AVX512BW), Timing::Secret),
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
    use crate::ct::Class;

    /// Encodes `bytes` into `out`, which has twice the length.
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(unused_variables, reason = "only the SIMD kernels have a temporary array")
    )]
    #[inline]
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
            Kernel::Avx512Bw => unsafe { x86::avx512bw::encode(bytes, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::encode(class, bytes, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::encode(class, bytes, out),
        }
    }

    /// Decodes `text` into `out`, which has half the length, and gives whether each character is a
    /// digit.
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        expect(unused_variables, reason = "only the SIMD kernels have a temporary array")
    )]
    #[inline]
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
            Kernel::Avx512Bw => unsafe { x86::avx512bw::decode(text, out) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: a `Backend` of this kernel exists only on a CPU that has its features.
            Kernel::Neon => unsafe { aarch64::decode(class, text, out) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Kernel::Simd128 => wasm32::decode(class, text, out),
        }
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
