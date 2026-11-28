use core::ops::{Deref, DerefMut};

pub(crate) struct Aligned {
    buffer: Vec<u8>,
    start: usize,
    len: usize,
}

impl Aligned {
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
