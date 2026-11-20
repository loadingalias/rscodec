//! JSON ([RFC 8259]): a pull reader with two profiles, limits, and a strict walk for typed
//! decoders, and a writer.
//!
//! [`read`] reads one document from a slice in memory. It gives a [`Reader`] to its closure, and
//! the closure reads the document with it:
//!
//! - [`Reader::token`] gives each [`Token`] in order. Each string and number borrows the input.
//! - [`Reader::value`] gives the next [`Value`], to read once with a type. [`Value::members`] and
//!   [`Value::elements`] walk a container in a loop, and they skip what the loop does not read.
//!
//! The reader checks each rule of its [`Profile`] and each of its [`Limits`] on each byte, also on
//! each byte that it skips. [`read`] reads to the end of the document after the closure returns,
//! so a document is valid exactly when `read` gives `Ok`. Each refusal is an [`Error`] with its
//! cause and its offset, and the errors come in the order of their offsets.
//!
//! The reader allocates no memory. The caller lends a scratch of [`Slot`]s, which holds each open
//! container and the keys of each open object for the duplicate check. A string with escapes
//! decodes into a buffer that the caller lends ([`Str::decode`]). No code recurses, so a deep
//! document does not grow the stack.
//!
//! ```
//! use rscodec::json::{self, ErrorKind, Limits, Profile, Slot};
//!
//! const LIMITS: Limits = Limits {
//!     input_bytes_max: 8192,
//!     depth_max: 8,
//!     string_bytes_max: 4096,
//!     members_max: 32,
//!     elements_max: 32,
//!     number_digits_max: 32,
//! };
//!
//! // A JOSE header. The walk compares each key after unescape, so `\u0061lg` is `alg`.
//! let header = br#"{"\u0061lg":"ES256","kid":"key-1","x-unknown":[1,2]}"#;
//! let mut scratch = [Slot::EMPTY; 8];
//! let (alg, kid) = json::read(header, &LIMITS, Profile::IJson, &mut scratch, #[inline(always)] |reader| {
//!     let (mut alg, mut kid) = (None, None);
//!     let mut members = reader.value().members()?;
//!     while let Some((key, value)) = members.next()? {
//!         if key == "alg" {
//!             alg = Some(value.str()?);
//!         } else if key == "kid" {
//!             kid = Some(value.str()?);
//!         }
//!     }
//!     Ok((alg, kid))
//! })?;
//! // Each `Str` borrows the header. `==` compares after unescape, and `decode` copies only a
//! // string with escapes.
//! assert!(alg.is_some_and(|alg| alg == "ES256"));
//! let mut buf = [0; 64];
//! if let Some(kid) = kid {
//!     assert_eq!(kid.decode(&mut buf)?, "key-1");
//! }
//!
//! // Each profile refuses a duplicate key, also after unescape.
//! let twice = br#"{"alg":"none","\u0061lg":"ES256"}"#;
//! let error = json::read(twice, &LIMITS, Profile::IJson, &mut scratch, |_| Ok(())).unwrap_err();
//! assert_eq!((error.kind(), error.offset()), (ErrorKind::Duplicate, 14));
//! # Ok::<(), json::Error>(())
//! ```
//!
//! [`write()`] writes compact JSON into a slice that the caller lends, with no allocation. Its
//! closure gets a [`Place`] for one value, and the types of [`Place`], [`Object`], and [`Array`]
//! give each value one place, so the text is always valid JSON.
//!
//! [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod block;
mod cursor;
mod error;
mod escape;
mod float;
mod integer;
mod number;
mod portable;
mod reader;
mod scratch;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod simd;
mod string;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
mod write;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use error::{Error, ErrorKind, Limit};
pub use escape::string_len;
pub use number::{Decimal, Number, NumberClass};
pub use reader::{Elements, Kind, Members, Reader, Token, Value};
pub use scratch::Slot;
pub use string::Str;
pub use write::{Array, Object, Place, write};

/// The most bytes that [`Place::f64`] writes: `-2.2250738585072014e-308`.
pub const F64_BYTES_MAX: usize = float::BYTES_MAX;
/// The most bytes that [`Place::i64`] writes: `-9223372036854775808`.
pub const I64_BYTES_MAX: usize = 20;
/// The most bytes that [`Place::u64`] writes: `18446744073709551615`.
pub const U64_BYTES_MAX: usize = 20;

use crate::cpu::{self, Caps, Kept, caps_static};
use block::{Block, Carry};

/// The rules that a document must meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// [RFC 8259], and no duplicate key in an object. A lone surrogate escape is an error, because
    /// a Rust `str` cannot hold it.
    ///
    /// [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259
    Rfc8259,
    /// [`Profile::Rfc8259`] and I-JSON ([RFC 7493]): no noncharacter, and only numbers that
    /// binary64 holds. The reader refuses a number whose magnitude rounds to infinity, a nonzero
    /// number that rounds to zero, a number of more than 17 significant digits, and an integer
    /// outside ±(2^53 − 1). An integer has no fraction and no exponent.
    ///
    /// [RFC 7493]: https://www.rfc-editor.org/rfc/rfc7493
    IJson,
}

/// The limits of one document. Each limit refuses the first item past it with
/// [`ErrorKind::Limit`] and its offset. No limit truncates or saturates.
///
/// No default exists, because a limit that fits a header of 150 bytes refuses a file of 2 MB, and a
/// limit that fits the file lets a hostile header use memory and time. State each limit for the
/// documents that you read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Limits {
    /// The bytes of the input.
    pub input_bytes_max: u32,
    /// The containers that one value can be in. A document of one scalar has depth 0, and `[]`
    /// has depth 1.
    pub depth_max: u16,
    /// The bytes of one string or key between its quotes, as written: an escape counts with each
    /// of its bytes. The decoded string is not longer.
    pub string_bytes_max: u32,
    /// The members of one object.
    pub members_max: u32,
    /// The elements of one array.
    pub elements_max: u32,
    /// The digits of one number: of its integer, its fraction, and its exponent.
    pub number_digits_max: u16,
}

/// Reads the document `input` with `decode`, which reads its values with the [`Reader`]. Then it
/// reads the rest of the document, with each check. Thus it gives the result of `decode` only for
/// a valid document.
///
/// It selects the kernel of the CPU once, and it calls `decode` at one place. Mark `decode`
/// `#[inline(always)]`: the compiler does not inline a large closure without it, and the reader
/// keeps its state in registers only when `decode` inlines. The kernel runs as one call for each
/// block of 64 bytes, with plain values: a direct call when the build fixes the kernel, from
/// `-C target-feature` or without `std`, and a call through a function pointer when the CPU selects
/// it at run time.
///
/// # Errors
///
/// The first error of `decode`, or the first refusal of the input. [`ErrorKind::Limit`] with
/// [`Limit::InputBytes`] at offset `limits.input_bytes_max` if the input is longer.
#[inline]
pub fn read<'a, T>(
    input: &'a [u8],
    limits: &Limits,
    profile: Profile,
    scratch: &mut [Slot],
    decode: impl FnOnce(&mut Reader<'a, '_>) -> Result<T, Error>,
) -> Result<T, Error> {
    read_with(Backend::chosen(), input, limits, profile, scratch, decode)
}

/// [`read`] on the kernel of `backend`.
///
/// It calls `decode` at one place. A call in each arm of a `match` on the kernel would give
/// `decode` one caller for each kernel, and the compiler would not inline it into any of them.
#[inline(always)]
fn read_with<'a, T>(
    backend: Backend,
    input: &'a [u8],
    limits: &Limits,
    profile: Profile,
    scratch: &mut [Slot],
    decode: impl FnOnce(&mut Reader<'a, '_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let input_bytes_max = limits.input_bytes_max as usize;
    if input.len() > input_bytes_max {
        return Err(Error::new(ErrorKind::Limit(Limit::InputBytes), input_bytes_max));
    }
    let mut reader = Reader::new(input, backend, *limits, profile, scratch);
    let value = decode(&mut reader)?;
    reader.finish()?;
    Ok(value)
}

/// A kernel of the reader: its code that classifies one block, for one instruction set. The
/// variants depend on the target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    /// The 64-bit SWAR reference. It runs on each CPU.
    Swar,
    /// AVX-512 BW, VL, and VBMI2, with PCLMULQDQ, BMI1, and BMI2. VBMI2 marks the CPUs that keep
    /// their clock with AVX-512.
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// AVX2, with PCLMULQDQ, BMI1, and BMI2.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Neon, with PMULL for the string mask.
    #[cfg(target_arch = "aarch64")]
    NeonPmull,
    /// Neon without PMULL.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// WebAssembly SIMD. It exists only with `-C target-feature=+simd128`, because WebAssembly has
    /// no run-time detection.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

/// The function of a kernel: the block of the input at an offset under its length. The block has
/// 64 bytes, with spaces after them when the input has fewer, after the 4 bytes before the offset
/// and after the carry. A call needs the CPU features of the kernel. No
/// kernel inlines: each block costs one call with plain values, so the reader stays in registers.
type Classify = unsafe fn(&[u8], usize, Carry) -> Block;

impl Kernel {
    /// Each kernel of the target architecture, the fastest first.
    const ALL: &[Self] = &[
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        #[cfg(target_arch = "aarch64")]
        Self::NeonPmull,
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Self::Simd128,
        Self::Swar,
    ];

    /// The CPU features that the kernel needs: the features of its `#[target_feature]` functions.
    /// The detection reports a feature only with each feature that it implies.
    const fn needs(self) -> Caps {
        #[cfg(target_arch = "x86_64")]
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ};
        match self {
            Self::Swar => Caps::NONE,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => AVX512BW
                .union(AVX512VL)
                .union(AVX512VBMI2)
                .union(PCLMULQDQ)
                .union(BMI1)
                .union(BMI2),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => AVX2.union(PCLMULQDQ).union(BMI1).union(BMI2),
            #[cfg(target_arch = "aarch64")]
            Self::NeonPmull => cpu::aarch64::NEON.union(cpu::aarch64::AES),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => cpu::aarch64::NEON,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => cpu::wasm32::SIMD128,
        }
    }

    /// The index in [`Kernel::ALL`] of the fastest kernel that a CPU with `caps` runs. The last
    /// kernel, the reference, runs on each CPU.
    const fn fastest(caps: Caps) -> usize {
        let mut index = 0;
        while index + 1 < Self::ALL.len() && !caps.has(Self::ALL[index].needs()) {
            index += 1;
        }
        index
    }

    /// The index of the kernel that the build fixes: if the build has no run-time detection, or if
    /// `caps_static()` holds the features of the fastest kernel.
    const fn fixed() -> Option<usize> {
        let index = Self::fastest(caps_static());
        if !cpu::DETECTS || index == Self::fastest(Caps::ALL) {
            Some(index)
        } else {
            None
        }
    }

    /// The function of the kernel.
    ///
    /// The cursor takes the function, not the kernel. When the build fixes the kernel, the function
    /// is a constant, each call of it is a direct call, and the kernel inlines into the reader. When
    /// the CPU selects the kernel, the reader calls the function once for each block.
    #[inline(always)]
    fn function(self) -> Classify {
        match self {
            Self::Swar => portable::block,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => x86::avx512,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => x86::avx2,
            #[cfg(target_arch = "aarch64")]
            Self::NeonPmull => aarch64::pmull::block,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => aarch64::shift::block,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => wasm32::simd128,
        }
    }
}

/// A kernel that this CPU runs.
///
/// Only [`Backend::chosen`] makes a `Backend`, and it checks the CPU first. Thus a `Backend` exists
/// only for a kernel whose features the CPU has, and a cursor runs only such a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Backend(Kernel);

impl Backend {
    /// The backend of the fastest kernel of this CPU. If the build does not fix it, the first call
    /// chooses it, and each later call reads the choice.
    #[inline(always)]
    fn chosen() -> Self {
        const FIXED: Option<usize> = Kernel::fixed();
        static KEPT: Kept = Kept::new();
        let index = FIXED.unwrap_or_else(|| KEPT.index(|| Kernel::fastest(cpu::caps())));
        Self(Kernel::ALL.get(index).copied().unwrap_or(Kernel::Swar))
    }
}
