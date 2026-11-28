//! The refusals of the reader and of the writer.

use core::fmt;

/// The cause of a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The grammar of RFC 8259 does not allow the byte at the offset there. A byte order mark, `NaN`,
    /// and a byte after the value of the document are examples.
    Syntax,
    /// The input ends before the value of the document ends. The offset is the length of the input.
    End,
    /// The bytes from the offset are not UTF-8. The offset is the `valid_up_to` of
    /// [`core::str::from_utf8`].
    Utf8,
    /// The escape at the offset is a surrogate without its pair. A Rust `str` cannot hold it, so
    /// each profile refuses it.
    Surrogate,
    /// [`Profile::IJson`](super::Profile::IJson): the character at the offset, or the escape at the
    /// offset, is a noncharacter.
    Noncharacter,
    /// The number at the offset violates the numeric limits of [`Profile::IJson`](super::Profile::IJson).
    Number,
    /// The value of the number at the offset does not fit the type that the accessor of
    /// [`Number`](super::Number) reads. The recovery is a wider type:
    /// [`Number::to_decimal`](super::Number::to_decimal), [`Number::to_f64`](super::Number::to_f64),
    /// or the text of [`Number::as_str`](super::Number::as_str).
    Range,
    /// The key at the offset equals an earlier key of its object, after unescape.
    Duplicate,
    /// The input passes a limit of [`Limits`](super::Limits). The offset is the first byte past
    /// the limit.
    Limit(Limit),
    /// The scratch has no slot for the container or the key at the offset. The recovery is a
    /// longer scratch: one slot for each open container and for each key of each open object.
    Scratch,
    /// The buffer of [`Str::decode`](super::Str::decode) is too short. The offset is the length of
    /// the decoded string, and the recovery is a buffer of that length.
    Output,
    /// The value at the offset is not of the type that the call reads, or no value comes at the
    /// offset. [`Number::to_i64`](super::Number::to_i64) and
    /// [`Number::to_u64`](super::Number::to_u64) give it for a number with a fraction or an
    /// exponent. The writer gives it at offset 0 when its closure writes no value.
    Type,
    /// Insufficient output space. The offset is the logical length before the failed write.
    /// Retry the document with a larger buffer.
    Full,
    /// The writer: JSON has no number for the NaN or the infinity that the call writes. The offset
    /// is the written length. The recovery is a finite value, or `null`.
    NonFinite,
}

/// The limit of [`Limits`](super::Limits) that the input passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Limit {
    /// [`Limits::input_bytes_max`](super::Limits::input_bytes_max).
    InputBytes,
    /// [`Limits::depth_max`](super::Limits::depth_max).
    Depth,
    /// [`Limits::string_bytes_max`](super::Limits::string_bytes_max).
    StringBytes,
    /// [`Limits::members_max`](super::Limits::members_max).
    Members,
    /// [`Limits::elements_max`](super::Limits::elements_max).
    Elements,
    /// [`Limits::number_digits_max`](super::Limits::number_digits_max).
    NumberDigits,
}

/// A refusal: its cause and the offset of the cause in the input, or the written length of the
/// writer. It holds no byte of the input.
///
/// It is two words: the offset and the code of its kind. In each `Result` of the reader, its
/// fields are the size of the words of a token. A smaller field would make the compiler split the
/// words of each token at its edges and join them again on each step.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Error {
    offset: usize,
    code: usize,
}

/// The kinds of [`ErrorKind`] with no field, in the order of their codes. A code past them is
/// [`ErrorKind::Limit`] of the limit at code − 13.
const KINDS: [ErrorKind; 13] = [
    ErrorKind::Syntax,
    ErrorKind::End,
    ErrorKind::Utf8,
    ErrorKind::Surrogate,
    ErrorKind::Noncharacter,
    ErrorKind::Number,
    ErrorKind::Range,
    ErrorKind::Duplicate,
    ErrorKind::Scratch,
    ErrorKind::Output,
    ErrorKind::Type,
    ErrorKind::Full,
    ErrorKind::NonFinite,
];

/// The limits of [`ErrorKind::Limit`], in the order of their codes.
const LIMITS: [Limit; 6] = [
    Limit::InputBytes,
    Limit::Depth,
    Limit::StringBytes,
    Limit::Members,
    Limit::Elements,
    Limit::NumberDigits,
];

impl Error {
    /// The refusal of `kind` at `offset`. A decoder that refuses a valid document, for example
    /// one that lacks a member that its type needs, gives its own refusal with it.
    #[inline]
    #[must_use]
    pub const fn new(kind: ErrorKind, offset: usize) -> Self {
        let code = match kind {
            ErrorKind::Syntax => 0,
            ErrorKind::End => 1,
            ErrorKind::Utf8 => 2,
            ErrorKind::Surrogate => 3,
            ErrorKind::Noncharacter => 4,
            ErrorKind::Number => 5,
            ErrorKind::Range => 6,
            ErrorKind::Duplicate => 7,
            ErrorKind::Scratch => 8,
            ErrorKind::Output => 9,
            ErrorKind::Type => 10,
            ErrorKind::Full => 11,
            ErrorKind::NonFinite => 12,
            ErrorKind::Limit(limit) => 13 + limit as usize,
        };
        Self { offset, code }
    }

    /// The cause.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        if self.code < KINDS.len() {
            KINDS[self.code]
        } else {
            ErrorKind::Limit(LIMITS[self.code - KINDS.len()])
        }
    }

    /// The offset in the input, or for [`ErrorKind::Output`], the length of the decoded string, or
    /// for the writer, the written length.
    #[inline]
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind())
            .field("offset", &self.offset)
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = self.offset;
        let cause = match self.kind() {
            ErrorKind::Syntax => "JSON does not allow the byte",
            ErrorKind::End => "the JSON text ends inside its value",
            ErrorKind::Utf8 => "the JSON text is not UTF-8",
            ErrorKind::Surrogate => "the JSON string has a surrogate escape without its pair",
            ErrorKind::Noncharacter => "the JSON string has a noncharacter, which I-JSON does not allow",
            ErrorKind::Number => "binary64 does not hold the JSON number, which I-JSON requires",
            ErrorKind::Range => "the JSON number does not fit the type that the decoder reads",
            ErrorKind::Duplicate => "the JSON key equals an earlier key of its object",
            ErrorKind::Limit(limit) => {
                let name = match limit {
                    Limit::InputBytes => "input_bytes_max",
                    Limit::Depth => "depth_max",
                    Limit::StringBytes => "string_bytes_max",
                    Limit::Members => "members_max",
                    Limit::Elements => "elements_max",
                    Limit::NumberDigits => "number_digits_max",
                };
                return write!(f, "the JSON text passes the limit {name} at offset {offset}");
            }
            ErrorKind::Scratch => "the scratch of the JSON reader has no free slot",
            ErrorKind::Output => {
                return write!(f, "the buffer is shorter than the {offset} bytes of the decoded string");
            }
            ErrorKind::Type => "the JSON value does not have the type that the decoder reads",
            ErrorKind::Full => {
                return write!(f, "the slice of the JSON writer ends after {offset} written bytes");
            }
            ErrorKind::NonFinite => "JSON has no number for NaN or an infinity",
        };
        write!(f, "{cause} at offset {offset}")
    }
}

impl core::error::Error for Error {}
