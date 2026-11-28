use super::{
    Backend, Classify,
    block::{self, Carry},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InvalidUtf8 {
    pub(super) offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Cursor<'a> {
    input: &'a [u8],
    classify: Classify,
    carry: Carry,
    tokens: u64,
    start: usize,
    invalid: usize,
}

const NO_ERROR: usize = usize::MAX;

impl<'a> Cursor<'a> {
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

    #[inline(always)]
    pub(super) const fn input(&self) -> &'a [u8] {
        self.input
    }

    #[inline(always)]
    pub(super) fn peek(&mut self) -> Result<Option<usize>, InvalidUtf8> {
        while self.tokens == 0 {
            if !self.advance()? {
                return Ok(None);
            }
        }
        Ok(Some(self.start | self.tokens.trailing_zeros() as usize))
    }

    #[inline(always)]
    pub(super) const fn take(&mut self) {
        self.tokens &= self.tokens.wrapping_sub(1);
    }

    #[inline(always)]
    pub(super) fn next(&mut self) -> Result<Option<usize>, InvalidUtf8> {
        let at = self.peek()?;
        self.take();
        Ok(at)
    }

    #[inline(always)]
    pub(super) fn resume(&mut self, at: usize) {
        let offset = at.wrapping_sub(self.start);
        if offset < 64 {
            self.tokens &= u64::MAX << offset;
        } else {
            *self = resume_block(*self, at);
        }
    }

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
            let valid = u32::try_from(invalid.offset.saturating_sub(start)).unwrap_or(u32::MAX);
            self.tokens &= !u64::MAX.unbounded_shl(valid);
        }
        Ok(true)
    }
}

#[inline(never)]
#[expect(
    unsafe_code,
    reason = "the function of the kernel of a `Backend` runs on each CPU that has the backend"
)]
fn resume_block(mut cursor: Cursor<'_>, at: usize) -> Cursor<'_> {
    let start = at & !63;
    cursor.tokens = 0;
    if start >= cursor.input.len() {
        cursor.start = start.wrapping_sub(64);
        return cursor;
    }
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
        let valid = u32::try_from(invalid.offset.saturating_sub(start)).unwrap_or(u32::MAX);
        cursor.tokens &= !u64::MAX.unbounded_shl(valid);
    }
    cursor
}

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
