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
/// cursor in memory hides the function of its kernel from the compiler. A function out of line
/// takes the cursor by value and gives it back, so that its caller keeps the cursor in registers.
#[derive(Clone, Copy, Debug)]
pub(super) struct Cursor<'a> {
    input: &'a [u8],
    classify: Classify,
    carry: Carry,
    /// The tokens of the current block that the cursor did not give.
    tokens: u64,
    /// The offset of the current block. Before the first block it is 64 under 0, in wrapping
    /// arithmetic, so the next block is always at `start + 64`.
    start: usize,
    /// The offset of the first byte that is not UTF-8, after the cursor found it, else
    /// [`NO_ERROR`]. A word and not an `Option`, so the cursor takes one register less.
    invalid: usize,
}

/// [`Cursor::invalid`] before the cursor finds a byte that is not UTF-8. No offset of an input is
/// `usize::MAX`.
const NO_ERROR: usize = usize::MAX;

impl<'a> Cursor<'a> {
    /// The cursor at the start of `input`, on the kernel of `backend`.
    #[inline(always)]
    pub(super) fn new(input: &'a [u8], backend: Backend) -> Self {
        Self {
            input,
            classify: backend.0.function(),
            carry: Carry::default(),
            tokens: 0,
            start: 0_usize.wrapping_sub(64),
            invalid: NO_ERROR,
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

    /// Resume at `at` after consuming only ASCII number and array syntax outside strings.
    /// Reuse the current block's masks when possible. Discard earlier tokens in either case.
    #[inline(always)]
    pub(super) fn resume(&mut self, at: usize) {
        let offset = at.wrapping_sub(self.start);
        if offset < 64 {
            self.tokens &= u64::MAX << offset;
        } else {
            *self = resume_block(*self, at);
        }
    }

    /// Classifies the next block, or gives `false` at the end of the input.
    #[inline(always)]
    #[expect(
        unsafe_code,
        reason = "the function of the kernel of a `Backend` runs on each CPU that has the backend"
    )]
    fn advance(&mut self) -> Result<bool, InvalidUtf8> {
        if self.invalid != NO_ERROR {
            return Err(InvalidUtf8 { offset: self.invalid });
        }
        let start = self.start.wrapping_add(64);
        if start >= self.input.len() {
            // The spaces after a short last block end each sequence, so only an input of whole
            // blocks can end in a sequence that needs more bytes.
            if block::ends_incomplete(block::before(self.input, self.input.len())) {
                let invalid = invalid(self.input, self.input.len());
                self.invalid = invalid.offset;
                return Err(invalid);
            }
            return Ok(false);
        }
        // SAFETY: `classify` is the function of the kernel of a `Backend`, and a `Backend` exists only
        // on a CPU that has the features of its kernel.
        let block = unsafe { (self.classify)(self.input, start, self.carry) };
        self.carry = block.carry;
        self.start = start;
        self.tokens = block.tokens;
        if block.carry.utf8_error() {
            let invalid = invalid(self.input, start);
            self.invalid = invalid.offset;
            // The tokens before the error. The error is in this block or in the 3 bytes before it,
            // so the count is under 64.
            let valid = u32::try_from(invalid.offset.saturating_sub(start)).unwrap_or(u32::MAX);
            self.tokens &= !u64::MAX.unbounded_shl(valid);
        }
        Ok(true)
    }
}

/// [`Cursor::resume`] at `at` in a later block than the block of `cursor`: the cursor at the block
/// of `at`, after the carry of the run. The run skipped each block before it, and each of their
/// bytes is ASCII, so their UTF-8 needs no check. It does not inline, and it takes the cursor by
/// value, so the reader of the caller stays in registers.
#[inline(never)]
#[expect(
    unsafe_code,
    reason = "the function of the kernel of a `Backend` runs on each CPU that has the backend"
)]
fn resume_block(mut cursor: Cursor<'_>, at: usize) -> Cursor<'_> {
    let start = at & !63;
    cursor.tokens = 0;
    if start >= cursor.input.len() {
        // The next block starts at or past the end, so `advance` gives the end.
        cursor.start = start.wrapping_sub(64);
        return cursor;
    }
    // The run read the byte before `start`: `at` is in a later block than the cursor, so `start`
    // is at least 64.
    let carry = Carry::after_run(cursor.input[start - 1]);
    // SAFETY: `classify` is the function of the kernel of a `Backend`, and a `Backend` exists only on
    // a CPU that has the features of its kernel.
    let block = unsafe { (cursor.classify)(cursor.input, start, carry) };
    cursor.carry = block.carry;
    cursor.start = start;
    cursor.tokens = block.tokens & u64::MAX << (at - start);
    if block.carry.utf8_error() {
        let invalid = invalid(cursor.input, start);
        cursor.invalid = invalid.offset;
        // The tokens before the error, as in `advance`.
        let valid = u32::try_from(invalid.offset.saturating_sub(start)).unwrap_or(u32::MAX);
        cursor.tokens &= !u64::MAX.unbounded_shl(valid);
    }
    cursor
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
