//! The refusals of the reader.

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
    /// [`Profile::IJson`](super::Profile::IJson): binary64 does not hold the number at the offset.
    Number,
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
    /// offset.
    Type,
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

/// A refusal: its cause and the offset of the cause in the input. It holds no byte of the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Error {
    /// The cause.
    pub kind: ErrorKind,
    /// The offset in the input, or for [`ErrorKind::Output`], the length of the decoded string.
    pub offset: usize,
}

impl Error {
    #[inline]
    pub(super) const fn new(kind: ErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = self.offset;
        let cause = match self.kind {
            ErrorKind::Syntax => "JSON does not allow the byte",
            ErrorKind::End => "the JSON text ends inside its value",
            ErrorKind::Utf8 => "the JSON text is not UTF-8",
            ErrorKind::Surrogate => "the JSON string has a surrogate escape without its pair",
            ErrorKind::Noncharacter => "the JSON string has a noncharacter, which I-JSON does not allow",
            ErrorKind::Number => "binary64 does not hold the JSON number, which I-JSON requires",
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
        };
        write!(f, "{cause} at offset {offset}")
    }
}

impl core::error::Error for Error {}
