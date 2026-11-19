//! The block cursor. It classifies one block of 64 bytes when the reader reaches it, and gives the
//! tokens of the block in order. It keeps no position buffer and makes no pass before the reader.
//! The input needs no padding: each kernel reads a last block of fewer than 64 bytes with spaces
//! after its end, and it reads no byte after the input.

use super::{
    Backend, Classify,
    block::{self, Carry},
};

/// The offset of the first byte of the input that is not UTF-8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InvalidUtf8 {
    pub(super) offset: usize,
}

/// The cursor over one input.
///
/// It checks the UTF-8 of each block before it gives a token of the block. Thus each byte before a
/// token that it gives is UTF-8, up to the last character boundary. At the first byte that is not
/// UTF-8, it gives the error, and no token after that byte. The token of a byte at 0x80 or more
/// outside a string can come before its error, because its block can end before the check finds the
/// error. No JSON value starts with such a byte.
///
/// Each method inlines, so that the cursor stays in registers in the function of the reader. A
/// cursor in memory hides the function of its kernel from the compiler.
#[derive(Debug)]
pub(super) struct Cursor<'a> {
    input: &'a [u8],
    classify: Classify,
    carry: Carry,
    /// The tokens of the current block that the cursor did not give.
    tokens: u64,
    /// The offset of the current block.
    start: usize,
    /// The offset of the next block.
    next: usize,
    /// The first byte that is not UTF-8, after the cursor found it.
    invalid: Option<InvalidUtf8>,
}

impl<'a> Cursor<'a> {
    /// The cursor at the start of `input`, on the kernel of `backend`.
    #[inline(always)]
    pub(super) fn new(input: &'a [u8], backend: Backend) -> Self {
        Self {
            input,
            classify: backend.0.function(),
            carry: Carry::default(),
            tokens: 0,
            start: 0,
            next: 0,
            invalid: None,
        }
    }

    /// The input.
    #[inline(always)]
    pub(super) const fn input(&self) -> &'a [u8] {
        self.input
    }

    /// The offset of the next token, which the cursor keeps, or `None` at the end of the input.
    ///
    /// # Errors
    ///
    /// The first byte that is not UTF-8, after each token before it. Each later call gives the same
    /// error.
    #[inline(always)]
    pub(super) fn peek(&mut self) -> Result<Option<usize>, InvalidUtf8> {
        while self.tokens == 0 {
            if !self.advance()? {
                return Ok(None);
            }
        }
        // `start` is a multiple of 64, so `|` adds the bit with no carry.
        Ok(Some(self.start | self.tokens.trailing_zeros() as usize))
    }

    /// Drops the token that [`Cursor::peek`] gave.
    #[inline(always)]
    pub(super) const fn take(&mut self) {
        self.tokens &= self.tokens.wrapping_sub(1);
    }

    /// The offset of the next token, which the cursor drops, or `None` at the end of the input.
    ///
    /// # Errors
    ///
    /// As [`Cursor::peek`].
    #[inline(always)]
    pub(super) fn next(&mut self) -> Result<Option<usize>, InvalidUtf8> {
        let at = self.peek()?;
        self.take();
        Ok(at)
    }

    /// Classifies the next block, or gives `false` at the end of the input.
    #[inline(always)]
    #[expect(
        unsafe_code,
        reason = "the function of the kernel of a `Backend` runs on each CPU that has the backend"
    )]
    fn advance(&mut self) -> Result<bool, InvalidUtf8> {
        if let Some(invalid) = self.invalid {
            return Err(invalid);
        }
        let start = self.next;
        let Some(rest) = self.input.get(start..).filter(|rest| !rest.is_empty()) else {
            // The spaces after a short last block end each sequence, so only an input of whole
            // blocks can end in a sequence that needs more bytes.
            if block::ends_incomplete(before(self.input, self.input.len())) {
                let invalid = invalid(self.input, self.input.len());
                self.invalid = Some(invalid);
                return Err(invalid);
            }
            return Ok(false);
        };
        // SAFETY: `classify` is the function of the kernel of a `Backend`, and a `Backend` exists only
        // on a CPU that has the features of its kernel.
        let block = unsafe { (self.classify)(rest, before(self.input, start), self.carry) };
        self.carry = block.carry;
        self.start = start;
        self.next = start + 64;
        self.tokens = block.tokens;
        if block.classes.utf8_error {
            let invalid = invalid(self.input, start);
            self.invalid = Some(invalid);
            // The tokens before the error. The error is in this block or in the 3 bytes before it,
            // so the count is under 64.
            let valid = u32::try_from(invalid.offset.saturating_sub(start)).unwrap_or(u32::MAX);
            self.tokens &= !u64::MAX.unbounded_shl(valid);
        }
        Ok(true)
    }
}

/// The 4 bytes before `at` in little-endian order, or zero before the first block.
#[inline]
fn before(input: &[u8], at: usize) -> u32 {
    input[..at].last_chunk().map_or(0, |bytes| u32::from_le_bytes(*bytes))
}

/// The first byte of `input` that is not UTF-8. The check of the block at `start`, or the check of
/// the end of the input at `start`, found an error.
#[cold]
fn invalid(input: &[u8], start: usize) -> InvalidUtf8 {
    // Each block before `start` passed the check, so the bytes before it are UTF-8 except for a
    // sequence at their end that needs more bytes. Thus the first of their last three bytes that is
    // not a continuation byte starts a character, or `start` does.
    let from = (start.saturating_sub(3)..start)
        .find(|&at| input[at] & 0xc0 != 0x80)
        .unwrap_or(start);
    let end = input.len().min(start + 64);
    let error = core::str::from_utf8(&input[from..end])
        .expect_err("the check of a block finds an error only in bytes that are not UTF-8");
    InvalidUtf8 {
        offset: from + error.valid_up_to(),
    }
}
