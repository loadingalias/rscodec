//! Read or write one [JSON](https://www.rfc-editor.org/rfc/rfc8259) document in caller-owned storage.
//!
//! [`read`] lends a typed [`Reader`]. [`visit`] emits borrowed [`Token`]s in input order.
//! Both enforce the selected [`Profile`] and [`Limits`], including on skipped input.
//! Lend one [`Slot`] per open container and per key of each open object.
//! After a closure succeeds, the reader validates the remaining input.
//!
//! ```
//! use rscodec::json::{self, ErrorKind, Limits, Profile, Slot};
//!
//! let limits = Limits {
//!     input_bytes_max: 128, depth_max: 2, string_bytes_max: 32,
//!     members_max: 4, elements_max: 4, number_digits_max: 8,
//! };
//! let mut slots = [Slot::EMPTY; 8];
//! let doc = br#"{"alg":"none","\u0061lg":"ES256"}"#;
//! let error = json::read(doc, &limits, Profile::IJson, &mut slots, |_| Ok(())).unwrap_err();
//! assert_eq!((error.kind(), error.offset()), (ErrorKind::Duplicate, 14));
//! ```
//!
//! [`write()`] writes compact JSON and returns its byte length.
//! [Reader and writer examples](https://github.com/loadingalias/rscodec/blob/main/docs/content/json.md)

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

use crate::cpu::{Caps, Timing};
use block::{Block, Carry};

/// The rules that a document must meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// [RFC 8259], and no duplicate key in an object. A lone surrogate escape is an error, because
    /// a Rust `str` cannot hold it.
    ///
    /// [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259
    Rfc8259,
    /// [`Profile::Rfc8259`] with an [I-JSON](https://www.rfc-editor.org/rfc/rfc7493) interoperability policy.
    /// Rejects noncharacters, binary64 overflow, nonzero underflow, and more than 17 significant digits.
    /// Integers without a fraction or exponent must lie within ±(2^53 − 1).
    IJson,
}

/// Resource limits for one document. Set all six for the input you accept.
/// Exceeding a limit gives [`ErrorKind::Limit`] with its offset.
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

/// Reads one document with a typed decoder. After `decode` succeeds, the reader validates all remaining input.
///
/// # Errors
///
/// Returns the caller error or the reader error. Oversized input gives [`Limit::InputBytes`] before parsing.
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

/// Visits the tokens of one document in input order. Strings and numbers borrow `input`.
/// A successful return means the full document passed validation. A callback can see tokens before a later error.
/// Large callbacks on a measured hot path can benefit from `#[inline(always)]`.
///
/// # Errors
///
/// Returns the callback error or the reader error. Oversized input gives [`Limit::InputBytes`] before parsing.
#[inline]
pub fn visit<'a>(
    input: &'a [u8],
    limits: &Limits,
    profile: Profile,
    scratch: &mut [Slot],
    each: impl FnMut(Token<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    visit_with(Backend::chosen(), input, limits, profile, scratch, each)
}

/// Keep the length check local. Sharing reader construction here changed the generated code of typed decoders.
#[inline(always)]
fn visit_with<'a>(
    backend: Backend,
    input: &'a [u8],
    limits: &Limits,
    profile: Profile,
    scratch: &mut [Slot],
    each: impl FnMut(Token<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    let input_bytes_max = limits.input_bytes_max as usize;
    if input.len() > input_bytes_max {
        return Err(Error::new(ErrorKind::Limit(Limit::InputBytes), input_bytes_max));
    }
    if reader::numbers_dense(input) {
        return reader::visit_numbers(input, backend, *limits, profile, scratch, each);
    }
    Reader::new(input, backend, *limits, profile, scratch).visit(each)
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

    /// The CPU features that the kernel needs, the features of its `#[target_feature]` functions,
    /// and its timing class, which is public. The detection reports a feature only with each
    /// feature that it implies.
    const fn needs(self) -> (Caps, Timing) {
        #[cfg(target_arch = "aarch64")]
        use crate::cpu::aarch64::{AES, NEON};
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        use crate::cpu::wasm32::SIMD128;
        #[cfg(target_arch = "x86_64")]
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ};
        let features = match self {
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
            Self::NeonPmull => NEON.union(AES),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => NEON,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => SIMD128,
        };
        (features, Timing::Public)
    }

    crate::cpu::choice!(Swar);

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
        Self(Kernel::chosen(Timing::Public))
    }
}
