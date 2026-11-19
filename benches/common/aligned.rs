//! Buffers at a 64-byte boundary. A benchmark gives each case one input and one output of this
//! kind, and each row of the case uses them. Thus the rows of a cell differ only in their code, not
//! in the place of their buffers in the cache lines and the pages. A benchmark opts in with
//! `#[path = "common/aligned.rs"] mod aligned;`.

use core::ops::{Deref, DerefMut};

/// Bytes from a 64-byte boundary.
pub(crate) struct Aligned {
    buffer: Vec<u8>,
    start: usize,
    len: usize,
}

impl Aligned {
    /// A copy of `bytes`.
    pub(crate) fn new(bytes: &[u8]) -> Self {
        let mut buffer = vec![0; bytes.len() + 63];
        let start = buffer.as_ptr().align_offset(64);
        buffer[start..start + bytes.len()].copy_from_slice(bytes);
        Self {
            buffer,
            start,
            len: bytes.len(),
        }
    }
}

impl Deref for Aligned {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.buffer[self.start..self.start + self.len]
    }
}

impl DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buffer[self.start..self.start + self.len]
    }
}
