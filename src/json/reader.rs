//! The reader on the block cursor: one push loop of the grammar, and the strict walk on it. No
//! code recurses. The scratch of the caller holds each open container and the keys of each open
//! object.

#[cfg(test)]
pub(super) mod tests;

use super::{
    Backend, Error, ErrorKind, Limit, Limits, Profile,
    cursor::Cursor,
    number::{self, Number},
    scratch::{self, Container, Frame, Key, Slot},
    string::{self, Str},
};

/// Check short objects for duplicates as each key arrives. Sort longer objects at close or on error.
const LINEAR_KEYS_MAX: u32 = 16;

/// A token of the input, which [`visit`](super::visit) gives. Each borrows the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token<'a> {
    /// `{`.
    ObjectStart,
    /// `}`.
    ObjectEnd,
    /// `[`.
    ArrayStart,
    /// `]`.
    ArrayEnd,
    /// An object key. The reader has consumed its colon.
    Key(Str<'a>),
    /// A string value.
    Str(Str<'a>),
    /// A number.
    Number(Number<'a>),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

/// The type of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// An object.
    Object,
    /// An array.
    Array,
    /// A string.
    String,
    /// A number.
    Number,
    /// `true` or `false`.
    Bool,
    /// `null`.
    Null,
}

/// What the grammar allows next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// A value: at the start of the document, after a `:`, or after a `,` in an array.
    Value,
    /// After `[`, a value or `]`. After `{`, a key or `}`.
    First,
    /// A key, after a `,` in an object.
    Key,
    /// After a value: a `,` or the end of its container, or the end of the input.
    After,
    /// After the end of the input.
    Done,
    /// After an error, which each later call gives.
    Failed(Error),
}

/// The next item of the input that the reader did not take, at its offset.
#[derive(Clone, Copy, Debug)]
enum Item {
    Value(usize, Kind),
    Key(usize),
    Close(usize),
    End,
}

/// The reader of one document, which [`read`](super::read) lends to its closure.
///
/// [`Reader::value`] gives the next value, to read it with a type. The walk checks each rule of the
/// profile on each byte that it passes, and each limit. After an error, each call gives that error.
#[derive(Debug)]
pub struct Reader<'a, 's> {
    cursor: Cursor<'a>,
    slots: &'s mut [Slot],
    /// The number of slots in use.
    top: usize,
    /// The innermost open container. Its parent is in the slot before its base.
    frame: Frame,
    depth: u16,
    state: State,
    profile: Profile,
    limits: Limits,
}

/// The offset or the length `value` in a slot. [`read`](super::read) refuses an input of more
/// than `u32::MAX` bytes, and it uses at most `u32::MAX` slots, so each fits.
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "each offset, length, and slot index is at most u32::MAX"
)]
pub(super) const fn index(value: usize) -> u32 {
    value as u32
}

/// The error of the byte at `at`, which the grammar does not allow there: the end of the input,
/// or the UTF-8 error if the bytes from `at` start no character. At one offset, the UTF-8 error
/// takes precedence over a syntax error. A byte that
/// the reader reaches before the cursor checked its block can be such a byte.
#[cold]
pub(super) fn unexpected(input: &[u8], at: usize) -> Error {
    let Some(&byte) = input.get(at) else {
        return Error::new(ErrorKind::End, input.len());
    };
    let window = &input[at..input.len().min(at + 4)];
    let invalid = byte >= 0x80 && core::str::from_utf8(window).is_err_and(|error| error.valid_up_to() == 0);
    Error::new(if invalid { ErrorKind::Utf8 } else { ErrorKind::Syntax }, at)
}

/// Gives the refusal `$error` of the input from the function of the reader `$reader`, after
/// [`Reader::fail`].
macro_rules! refuse {
    ($reader:ident, $error:expr) => {
        return Err($reader.fail($error))
    };
}

/// The value of `$result`, or its refusal of the input, as [`refuse!`] gives it.
macro_rules! input {
    ($reader:ident, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => refuse!($reader, error),
        }
    };
}

/// The offset of the next token of the cursor of `$reader`, which the cursor keeps. The input must
/// not end there.
macro_rules! peek {
    ($reader:ident) => {
        match $reader.cursor.peek() {
            Ok(Some(at)) => at,
            Ok(None) => refuse!($reader, Error::new(ErrorKind::End, $reader.cursor.input().len())),
            Err(invalid) => refuse!($reader, Error::new(ErrorKind::Utf8, invalid.offset)),
        }
    };
}

impl<'a, 's> Reader<'a, 's> {
    #[inline(always)]
    pub(super) fn new(
        input: &'a [u8],
        backend: Backend,
        limits: Limits,
        profile: Profile,
        scratch: &'s mut [Slot],
    ) -> Self {
        let len = scratch.len().min(u32::MAX as usize);
        Self {
            cursor: Cursor::new(input, backend),
            slots: &mut scratch[..len],
            top: 0,
            frame: Frame::ROOT,
            depth: 0,
            state: State::Value,
            profile,
            limits,
        }
    }

    /// The next token, or `None` after the end of the document: one item of the walk. A `,`, a `:`,
    /// and whitespace are no tokens.
    #[inline(always)]
    fn token(&mut self) -> Result<Option<Token<'a>>, Error> {
        let token = match self.item()? {
            Item::Value(at, kind) => self.take(at, kind),
            Item::Key(at) => self.key(at).map(Token::Key),
            Item::Close(_) => self.close(),
            Item::End => return Ok(None),
        };
        self.check(token).map(Some)
    }

    /// The next value, to read once with a type.
    #[inline(always)]
    pub fn value(&mut self) -> Value<'_, 'a, 's> {
        Value(self)
    }

    /// Reads the document from its start and gives each token to `emit`, with the push loop that
    /// has no number run.
    #[inline(always)]
    pub(super) fn visit(&mut self, emit: impl FnMut(Token<'a>) -> Result<(), Error>) -> Result<(), Error> {
        self.run::<false>(emit)
    }

    /// Use the typed walk to finish validation. Duplicating the push loop slowed typed decoders by 2–15%.
    #[inline(always)]
    pub(super) fn finish(&mut self) -> Result<(), Error> {
        while self.token()?.is_some() {}
        Ok(())
    }

    /// Emit tokens from the start of the document. Control flow encodes the grammar state.
    /// Input errors pass through [`Reader::fail`]. Callback errors return directly.
    /// `NUMBER_RUN` enables the fast path for arrays of numbers.
    #[inline(always)]
    fn run<const NUMBER_RUN: bool>(
        &mut self,
        mut emit: impl FnMut(Token<'a>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let input = self.cursor.input();
        'value: loop {
            // One match on the byte, and one call of `emit` for each kind of token, so the match
            // of `emit` on the token goes at each call. The count of the element comes after the
            // check of the byte, as in `Reader::take`.
            let at = peek!(self);
            match input[at] {
                b'"' => {
                    input!(self, self.element(at));
                    emit(Token::Str(input!(self, self.string(at))))?;
                }
                b'-' | b'0'..=b'9' if NUMBER_RUN => {
                    // One call of `emit` gives each number, also each number of the number run,
                    // so the loop holds one copy of the work of the consumer for a number.
                    let mut at = at;
                    loop {
                        input!(self, self.element(at));
                        let number = input!(self, self.number(at));
                        let end = at + number.as_str().len();
                        emit(Token::Number(number))?;
                        if self.frame.container != Container::Array {
                            break;
                        }
                        match self.number_run(end, &mut emit)? {
                            Run::Number(next) => at = next,
                            Run::After => break,
                            Run::Value => continue 'value,
                        }
                    }
                }
                b'-' | b'0'..=b'9' => {
                    input!(self, self.element(at));
                    emit(Token::Number(input!(self, self.number(at))))?;
                }
                b't' | b'f' => {
                    input!(self, self.element(at));
                    emit(Token::Bool(input!(self, self.literal(at))))?;
                }
                b'n' => {
                    input!(self, self.element(at));
                    input!(self, self.literal(at));
                    emit(Token::Null)?;
                }
                b'[' => {
                    input!(self, self.element(at));
                    input!(self, self.open(at, Container::Array));
                    emit(Token::ArrayStart)?;
                    let first = peek!(self);
                    if input[first] != b']' {
                        continue 'value;
                    }
                    emit(input!(self, self.close()))?;
                }
                b'{' => {
                    input!(self, self.element(at));
                    input!(self, self.open(at, Container::Object));
                    emit(Token::ObjectStart)?;
                    let first = peek!(self);
                    match input[first] {
                        b'}' => emit(input!(self, self.close()))?,
                        b'"' => {
                            emit(Token::Key(input!(self, self.key(first))))?;
                            continue 'value;
                        }
                        _ => refuse!(self, unexpected(input, first)),
                    }
                }
                _ => refuse!(self, unexpected(input, at)),
            }
            // After a value.
            loop {
                let at = match self.cursor.peek() {
                    Ok(Some(at)) => at,
                    Ok(None) if self.frame.container == Container::Root => {
                        self.state = State::Done;
                        return Ok(());
                    }
                    Ok(None) => refuse!(self, Error::new(ErrorKind::End, input.len())),
                    Err(invalid) => refuse!(self, Error::new(ErrorKind::Utf8, invalid.offset)),
                };
                match (self.frame.container, input[at]) {
                    (Container::Array, b',') => {
                        self.cursor.take();
                        continue 'value;
                    }
                    (Container::Object, b',') => {
                        self.cursor.take();
                        let at = peek!(self);
                        if input[at] != b'"' {
                            refuse!(self, unexpected(input, at));
                        }
                        emit(Token::Key(input!(self, self.key(at))))?;
                        continue 'value;
                    }
                    (Container::Array, b']') | (Container::Object, b'}') => emit(input!(self, self.close()))?,
                    _ => refuse!(self, unexpected(input, at)),
                }
            }
        }
    }

    /// Keeps `error` for each later call, and gives it. It inlines, as each method of the reader
    /// does, and no closure that the compiler can leave as a call takes the reader: a call that
    /// takes the reader keeps the reader in memory, not in registers.
    #[inline(always)]
    fn fail(&mut self, error: Error) -> Error {
        if let State::Failed(first) = self.state {
            return first;
        }
        let error = earlier_duplicate(self.slots, self.frame, self.top, self.cursor.input(), error);
        self.state = State::Failed(error);
        error
    }

    #[inline(always)]
    fn check<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => Err(self.fail(error)),
        }
    }

    /// Continue an array after the number ending at `end`. Consume commas and array delimiters
    /// directly. Resume the cursor at the first other byte and discard only tokens before it.
    #[inline(always)]
    fn number_run(
        &mut self,
        mut end: usize,
        emit: &mut impl FnMut(Token<'a>) -> Result<(), Error>,
    ) -> Result<Run, Error> {
        let input = self.cursor.input();
        loop {
            // After a value, at `end`. A `]` can close the last array of the run.
            if self.frame.container != Container::Array {
                self.cursor.resume(end);
                return Ok(Run::After);
            }
            let mut value = match input.get(end) {
                Some(b',') => end + 1,
                Some(b']') => {
                    emit(input!(self, self.close()))?;
                    end += 1;
                    continue;
                }
                _ => {
                    self.cursor.resume(end);
                    return Ok(Run::After);
                }
            };
            // At a value of an array, after a `,` or a `[`.
            let mut opened = false;
            loop {
                match input.get(value) {
                    Some(b'-' | b'0'..=b'9') => return Ok(Run::Number(value)),
                    Some(b'[') => {
                        input!(self, self.element(value));
                        input!(self, self.open(value, Container::Array));
                        emit(Token::ArrayStart)?;
                        value += 1;
                        opened = true;
                    }
                    // An empty array: the `]` after a value closes it, at one place.
                    Some(b']') if opened => {
                        end = value;
                        break;
                    }
                    _ => {
                        // The cursor reads the rest from `value`. After a `[`, a `]` can come
                        // after whitespace.
                        self.cursor.resume(value);
                        if opened && input[peek!(self)] == b']' {
                            emit(input!(self, self.close()))?;
                            return Ok(Run::After);
                        }
                        return Ok(Run::Value);
                    }
                }
            }
        }
    }

    /// The next item, which it does not take. It takes each `,` before it.
    ///
    /// After a `,`, it reads the next item in the arm of its container, so the branches of the
    /// common sequences, a value, a `,`, and a key, are each at their own place in the code.
    #[inline(always)]
    fn item(&mut self) -> Result<Item, Error> {
        let Some(at) = self.peek()? else {
            return self.end();
        };
        let input = self.cursor.input();
        let byte = input[at];
        let container = self.frame.container;
        let item = match self.state {
            State::Value => kind(byte).map(|kind| Item::Value(at, kind)),
            State::First => match (container, byte) {
                (Container::Array, b']') | (Container::Object, b'}') => Some(Item::Close(at)),
                (Container::Object, b'"') => Some(Item::Key(at)),
                (Container::Object | Container::Root, _) => None,
                (Container::Array, _) => kind(byte).map(|kind| Item::Value(at, kind)),
            },
            State::Key => (byte == b'"').then_some(Item::Key(at)),
            State::After => match (container, byte) {
                (Container::Object, b',') => {
                    self.cursor.take();
                    self.state = State::Key;
                    let Some(at) = self.peek()? else {
                        return self.end();
                    };
                    return if input[at] == b'"' {
                        Ok(Item::Key(at))
                    } else {
                        Err(self.fail(unexpected(input, at)))
                    };
                }
                (Container::Array, b',') => {
                    self.cursor.take();
                    self.state = State::Value;
                    let Some(at) = self.peek()? else {
                        return self.end();
                    };
                    return match kind(input[at]) {
                        Some(kind) => Ok(Item::Value(at, kind)),
                        None => Err(self.fail(unexpected(input, at))),
                    };
                }
                (Container::Array, b']') | (Container::Object, b'}') => Some(Item::Close(at)),
                _ => None,
            },
            State::Done => Some(Item::End),
            State::Failed(error) => return Err(error),
        };
        match item {
            Some(item) => Ok(item),
            None => Err(self.fail(unexpected(input, at))),
        }
    }

    /// The offset of the next token of the cursor, which it does not take.
    #[inline(always)]
    fn peek(&mut self) -> Result<Option<usize>, Error> {
        match self.cursor.peek() {
            Ok(at) => Ok(at),
            Err(invalid) => Err(self.fail(Error::new(ErrorKind::Utf8, invalid.offset))),
        }
    }

    /// The end of the input.
    #[inline(always)]
    fn end(&mut self) -> Result<Item, Error> {
        match self.state {
            State::Failed(error) => Err(error),
            State::Done => Ok(Item::End),
            State::After if self.frame.container == Container::Root => {
                self.state = State::Done;
                Ok(Item::End)
            }
            _ => Err(self.fail(Error::new(ErrorKind::End, self.cursor.input().len()))),
        }
    }

    /// The next item, which must be a value. Otherwise the reader takes nothing, and it gives
    /// [`ErrorKind::Type`].
    #[inline(always)]
    fn next_value(&mut self) -> Result<(usize, Kind), Error> {
        match self.item()? {
            Item::Value(at, kind) => Ok((at, kind)),
            Item::Key(at) | Item::Close(at) => Err(Error::new(ErrorKind::Type, at)),
            Item::End => Err(Error::new(ErrorKind::Type, self.cursor.input().len())),
        }
    }

    /// The next item, which must be a value of `kind`. Otherwise the reader takes nothing, and it
    /// gives [`ErrorKind::Type`].
    #[inline(always)]
    fn expect(&mut self, kind: Kind) -> Result<usize, Error> {
        match self.next_value()? {
            (at, found) if found == kind => Ok(at),
            (at, _) => Err(Error::new(ErrorKind::Type, at)),
        }
    }

    /// Counts the value at `at` in its array.
    #[inline(always)]
    fn element(&mut self, at: usize) -> Result<(), Error> {
        if self.frame.container == Container::Array {
            if self.frame.count >= self.limits.elements_max {
                return Err(Error::new(ErrorKind::Limit(Limit::Elements), at));
            }
            // An input of at most `u32::MAX` bytes has fewer elements.
            self.frame.count += 1;
        }
        Ok(())
    }

    /// Takes the value at `at` of `kind`.
    #[inline(always)]
    fn take(&mut self, at: usize, kind: Kind) -> Result<Token<'a>, Error> {
        self.element(at)?;
        let token = match kind {
            Kind::Object => {
                self.open(at, Container::Object)?;
                return Ok(Token::ObjectStart);
            }
            Kind::Array => {
                self.open(at, Container::Array)?;
                return Ok(Token::ArrayStart);
            }
            Kind::String => Token::Str(self.string(at)?),
            Kind::Number => Token::Number(self.number(at)?),
            Kind::Bool => Token::Bool(self.literal(at)?),
            Kind::Null => {
                self.literal(at)?;
                Token::Null
            }
        };
        self.state = State::After;
        Ok(token)
    }

    /// Opens the container at `at`.
    #[inline(always)]
    fn open(&mut self, at: usize, container: Container) -> Result<(), Error> {
        if self.depth >= self.limits.depth_max {
            return Err(Error::new(ErrorKind::Limit(Limit::Depth), at));
        }
        let Some(slot) = self.slots.get_mut(self.top) else {
            return Err(Error::new(ErrorKind::Scratch, at));
        };
        *slot = self.frame.save();
        self.top += 1;
        self.depth += 1;
        self.frame = Frame {
            container,
            count: 0,
            base: index(self.top),
            keys: 0,
        };
        self.state = State::First;
        self.cursor.take();
        Ok(())
    }

    /// Closes the innermost container at its end. A key of an object with more than
    /// [`LINEAR_KEYS_MAX`] members can equal an earlier key that no comparison checked yet.
    #[inline(always)]
    fn close(&mut self) -> Result<Token<'a>, Error> {
        let base = self.frame.base as usize;
        let token = if self.frame.container == Container::Object {
            if self.frame.count > LINEAR_KEYS_MAX
                && let Some(duplicate) = scratch::first_duplicate(&mut self.slots[base..self.top], self.cursor.input())
            {
                return Err(Error::new(ErrorKind::Duplicate, duplicate));
            }
            Token::ObjectEnd
        } else {
            Token::ArrayEnd
        };
        // The parent of each container is in the slot before its base.
        self.top = base - 1;
        self.frame = Frame::load(self.slots[self.top]);
        self.depth -= 1;
        self.state = State::After;
        self.cursor.take();
        Ok(token)
    }

    /// Takes the key at `at` and its `:`.
    #[inline(always)]
    fn key(&mut self, at: usize) -> Result<Str<'a>, Error> {
        if self.frame.count >= self.limits.members_max {
            return Err(Error::new(ErrorKind::Limit(Limit::Members), at));
        }
        if self.top >= self.slots.len() {
            return Err(Error::new(ErrorKind::Scratch, at));
        }
        let key = self.string(at)?;
        // An input of at most `u32::MAX` bytes has fewer members.
        self.frame.count += 1;
        let input = self.cursor.input();
        let raw = key.raw().as_bytes();
        let new = Key::new(index(at), raw, index(raw.len()), index(key.len()));
        if self.frame.count <= LINEAR_KEYS_MAX {
            // Only a key whose bit of the filter is set already can equal an earlier key.
            let bit = new.bit();
            let base = self.frame.base as usize;
            if self.frame.keys & bit != 0 && new.in_keys(&self.slots[base..self.top], input) {
                return Err(Error::new(ErrorKind::Duplicate, at));
            }
            self.frame.keys |= bit;
        }
        self.slots[self.top] = new.save();
        self.top += 1;
        match self.cursor.next() {
            Ok(Some(colon)) if input[colon] == b':' => {}
            Ok(Some(other)) => return Err(unexpected(input, other)),
            Ok(None) => return Err(Error::new(ErrorKind::End, input.len())),
            Err(invalid) => return Err(Error::new(ErrorKind::Utf8, invalid.offset)),
        }
        self.state = State::Value;
        Ok(key)
    }

    /// Takes the string whose opening quote is at `open`. It checks each escape, each byte under
    /// 0x20, the length, and in [`Profile::IJson`], each noncharacter.
    ///
    /// Most strings have no escape and no byte under 0x20, so their next token is their closing
    /// quote, and this method reads them. [`escaped`] reads each other string out of line, with the
    /// cursor by value, so the loop of the caller stays small and in registers.
    #[inline(always)]
    fn string(&mut self, open: usize) -> Result<Str<'a>, Error> {
        self.cursor.take();
        let input = self.cursor.input();
        let start = open + 1;
        // The first offset past the limit. A byte of the string there passes the limit, and the
        // error of the limit comes before each error after it.
        let limit = start.saturating_add(self.limits.string_bytes_max as usize);
        let ijson = self.profile == Profile::IJson;
        if let Ok(Some(at)) = self.cursor.peek()
            && input[at] == b'"'
            && at <= limit
            && (!ijson || string::first_noncharacter(&input[start..at]).is_none())
        {
            self.cursor.take();
            return Ok(Str::new(text(input, start, at), at - start));
        }
        let (cursor, string) = escaped(self.cursor, start, limit, ijson);
        self.cursor = cursor;
        string
    }

    /// Takes the number at `at`.
    #[inline(always)]
    fn number(&mut self, at: usize) -> Result<Number<'a>, Error> {
        self.cursor.take();
        let input = self.cursor.input();
        let number = number::scan(input, at, self.limits.number_digits_max)?;
        if self.profile == Profile::IJson && !number::binary64_holds(&number) {
            return Err(Error::new(ErrorKind::Number, at));
        }
        Ok(number)
    }

    /// Takes the literal `true`, `false`, or `null` at `at`, and gives whether it is `true`. Each
    /// literal is one compare of 4 bytes, and `false` one more byte.
    #[inline(always)]
    fn literal(&mut self, at: usize) -> Result<bool, Error> {
        self.cursor.take();
        let input = self.cursor.input();
        let word = input.get(at..).and_then(<[u8]>::first_chunk::<4>);
        let (end, value) = match word {
            Some(b"true") => (at + 4, true),
            Some(b"null") => (at + 4, false),
            Some(b"fals") if input.get(at + 4) == Some(&b'e') => (at + 5, false),
            _ => return Err(literal_error(input, at)),
        };
        match input.get(end) {
            None | Some(b' ' | b'\t' | b'\n' | b'\r' | b',' | b':' | b'[' | b']' | b'{' | b'}' | b'"') => Ok(value),
            Some(_) => Err(unexpected(input, end)),
        }
    }
}

/// The documents that the number run reads: at least this many bytes. A shorter document has too
/// few numbers for the run to pay.
const NUMBER_RUN_BYTES_MIN: usize = 4096;

/// Whether the push loop takes the number run for `input`: a document of at least
/// [`NUMBER_RUN_BYTES_MIN`] bytes whose 64 bytes at its middle are each a byte of a number, `,`,
/// `[`, or `]`, as in an array of numbers with no whitespace. The test reads at most 64 bytes, and
/// it changes the speed of the read, not its result.
#[inline(always)]
pub(super) fn numbers_dense(input: &[u8]) -> bool {
    input.len() >= NUMBER_RUN_BYTES_MIN
        && input[input.len() / 2..][..64].iter().all(|byte| {
            matches!(
                byte,
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' | b',' | b'[' | b']'
            )
        })
}

/// Keep the number loop separate. Inlining it slowed other documents by 2–9% on Xeon.
/// Construct the reader here so the compiler can keep its state in registers.
#[inline(never)]
pub(super) fn visit_numbers<'a>(
    input: &'a [u8],
    backend: Backend,
    limits: Limits,
    profile: Profile,
    scratch: &mut [Slot],
    emit: impl FnMut(Token<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    Reader::new(input, backend, limits, profile, scratch).run::<true>(emit)
}

/// Where the push loop goes on after a step of the number run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Run {
    /// The number at this offset, which the run reached with no cursor.
    Number(usize),
    /// After a value. The cursor resumed after it.
    After,
    /// At a value of an array, after a `,` or a `[`. The cursor resumed at it.
    Value,
}

/// The error of the literal at `at` whose bytes are not `true`, `false`, or `null`: the first byte
/// that differs from the literal of its first byte, or the end of the input.
#[cold]
fn literal_error(input: &[u8], at: usize) -> Error {
    let text: &[u8] = match input[at] {
        b't' => b"true",
        b'f' => b"false",
        _ => b"null",
    };
    let rest = &input[at..];
    let mismatch = text.iter().zip(rest).position(|(expected, byte)| expected != byte);
    unexpected(
        input,
        mismatch.map_or_else(|| (at + text.len()).min(input.len()), |index| at + index),
    )
}

/// [`Reader::string`] of a string whose next token is not its closing quote: a string with an
/// escape, a byte under 0x20, a noncharacter in [`Profile::IJson`], a byte past the limit, or no
/// end. `start` is the offset after its opening quote, and `limit` the first offset past the limit.
/// It takes the cursor at the next token of the string and gives it back after the string. It does
/// not inline: the cursor is a value, so the reader of the caller stays in registers.
#[inline(never)]
fn escaped(mut cursor: Cursor<'_>, start: usize, limit: usize, ijson: bool) -> (Cursor<'_>, Result<Str<'_>, Error>) {
    let string = escaped_string(&mut cursor, start, limit, ijson);
    (cursor, string)
}

/// The string of [`escaped`].
#[inline(always)]
fn escaped_string<'a>(cursor: &mut Cursor<'a>, start: usize, limit: usize, ijson: bool) -> Result<Str<'a>, Error> {
    let input = cursor.input();
    let past = |error: Error| {
        if error.offset() > limit {
            Error::new(ErrorKind::Limit(Limit::StringBytes), limit)
        } else {
            error
        }
    };
    // The bytes that unescape removes.
    let mut shrink = 0;
    // The start of the text after the last escape.
    let mut from = start;
    // A high surrogate escape, at its offset, before its low surrogate escape.
    let mut high: Option<(usize, u32)> = None;
    loop {
        let (at, end) = match cursor.next() {
            Ok(Some(at)) => (at, None),
            Ok(None) => (input.len(), Some(Error::new(ErrorKind::End, input.len()))),
            Err(invalid) => (invalid.offset, Some(Error::new(ErrorKind::Utf8, invalid.offset))),
        };
        if let Some((surrogate, _)) = high
            && (at != surrogate + 6 || end.is_some() || input[at] != b'\\')
        {
            return Err(past(Error::new(ErrorKind::Surrogate, surrogate)));
        }
        if ijson && let Some(offset) = string::first_noncharacter(&input[from..at]) {
            return Err(past(Error::new(ErrorKind::Noncharacter, from + offset)));
        }
        if let Some(end) = end {
            return Err(past(end));
        }
        let byte = input[at];
        if at > limit || at == limit && byte == b'\\' {
            return Err(Error::new(ErrorKind::Limit(Limit::StringBytes), limit));
        }
        match byte {
            b'"' => return Ok(Str::new(text(input, start, at), at - start - shrink)),
            b'\\' => {}
            _ => return Err(Error::new(ErrorKind::Syntax, at)),
        }
        let escape = at;
        match input.get(escape + 1) {
            Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') if high.is_none() => {
                shrink += 1;
                from = escape + 2;
            }
            // `\u00` and two hex digits, the common escape of a control byte: a value under U+0100
            // is no surrogate and no noncharacter. A byte that is not a hex digit takes the path
            // below, which gives its error.
            Some(b'u')
                if high.is_none()
                    && let Some([b'0', b'0', first, second]) =
                        input.get(escape + 2..).and_then(<[u8]>::first_chunk)
                    && let Ok(value) = string::hex4(&[b'0', b'0', *first, *second]) =>
            {
                shrink += if value < 0x80 { 5 } else { 4 };
                from = escape + 6;
            }
            Some(b'u') => {
                let value = string::hex4(&input[escape + 2..]).map_err(|index| {
                    past(high.map_or_else(
                        || unexpected(input, escape + 2 + index),
                        |(surrogate, _)| Error::new(ErrorKind::Surrogate, surrogate),
                    ))
                })?;
                match (high, value) {
                    (Some((surrogate, first)), 0xdc00..0xe000) => {
                        let value = 0x1_0000 + ((first - 0xd800) << 10) + (value - 0xdc00);
                        if ijson && string::noncharacter(value) {
                            return Err(past(Error::new(ErrorKind::Noncharacter, surrogate)));
                        }
                        high = None;
                        // 12 bytes of escapes give 4 bytes of UTF-8.
                        shrink += 8;
                    }
                    (Some((surrogate, _)), _) => {
                        return Err(past(Error::new(ErrorKind::Surrogate, surrogate)));
                    }
                    (None, 0xdc00..0xe000) => return Err(past(Error::new(ErrorKind::Surrogate, escape))),
                    (None, 0xd800..0xdc00) => high = Some((escape, value)),
                    (None, _) => {
                        if ijson && string::noncharacter(value) {
                            return Err(past(Error::new(ErrorKind::Noncharacter, escape)));
                        }
                        shrink += 6 - string::utf8_len(value);
                    }
                }
                from = escape + 6;
            }
            _ => {
                let error = high.map_or_else(
                    || unexpected(input, escape + 1),
                    |(surrogate, _)| Error::new(ErrorKind::Surrogate, surrogate),
                );
                return Err(past(error));
            }
        }
    }
}

/// The text of `input` from `start` to `end`: a string between its quotes, or a number.
#[inline(always)]
#[expect(
    unsafe_code,
    reason = "the cursor checked the UTF-8 of each string, and a number has only ASCII bytes"
)]
pub(super) fn text(input: &[u8], start: usize, end: usize) -> &str {
    let bytes = &input[start..end];
    // SAFETY: each caller passes a string or a number. For a string, the cursor gave the token
    // at `end`, so each byte before it is UTF-8, to the last start of a character. The byte at
    // `end` and the byte before `start` are ASCII quotes, so `bytes` starts and ends at a
    // character. Each byte of a number passed its grammar, which has only ASCII bytes.
    unsafe { core::str::from_utf8_unchecked(bytes) }
}

/// `error`, or the first duplicate key of an open object before it. An object of more than
/// [`LINEAR_KEYS_MAX`] members checks its later keys only when it ends, so an error inside the
/// object can come first. The keys of each open object are in the scratch, from the innermost
/// object at `frame` and `top` to the root. Thus the errors come in the order of their offsets,
/// and a valid document pays nothing. The reader stops after the error, so the order of the keys
/// in the scratch can change.
#[cold]
fn earlier_duplicate(slots: &mut [Slot], mut frame: Frame, mut top: usize, input: &[u8], error: Error) -> Error {
    let mut first = error;
    loop {
        let base = frame.base as usize;
        if frame.container == Container::Object
            && frame.count > LINEAR_KEYS_MAX
            && let Some(duplicate) = scratch::first_duplicate(&mut slots[base..top], input)
            && duplicate < first.offset()
        {
            first = Error::new(ErrorKind::Duplicate, duplicate);
        }
        if frame.container == Container::Root {
            return first;
        }
        // The parent of each container is in the slot before its base.
        top = base - 1;
        frame = Frame::load(slots[top]);
    }
}

/// The kind of the value that starts with `byte`.
#[inline(always)]
const fn kind(byte: u8) -> Option<Kind> {
    Some(match byte {
        b'{' => Kind::Object,
        b'[' => Kind::Array,
        b'"' => Kind::String,
        b'-' | b'0'..=b'9' => Kind::Number,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => return None,
    })
}

/// Takes the scalar value of `$kind` with the method `$take` of `$reader`. It is a macro and not a
/// function with a closure, so that the method inlines: a closure that the compiler leaves as a
/// call takes the reader, which then stays in memory.
macro_rules! scalar {
    ($reader:expr, $kind:expr, $take:ident) => {{
        let reader = $reader;
        let at = reader.expect($kind)?;
        let result = match reader.element(at) {
            Ok(()) => reader.$take(at),
            Err(error) => Err(error),
        };
        if result.is_ok() {
            reader.state = State::After;
        }
        reader.check(result)
    }};
}

/// The next value of a [`Reader`], to read once with a type: the strict walk.
///
/// Each method takes the value. If the value has another type, the method gives
/// [`ErrorKind::Type`] at its offset and takes nothing. [`Value::members`] and [`Value::elements`]
/// open a container, to walk it one item at a time. The walk skips what the loop does not read,
/// with each check of the profile.
#[derive(Debug)]
pub struct Value<'r, 'a, 's>(&'r mut Reader<'a, 's>);

impl<'r, 'a, 's> Value<'r, 'a, 's> {
    /// The type of the value. It takes nothing.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if no value comes next, or the refusal of the input before the value.
    #[inline(always)]
    pub fn kind(&mut self) -> Result<Kind, Error> {
        self.0.next_value().map(|(_, kind)| kind)
    }

    /// Takes a string.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not a string, or the refusal of the input.
    #[inline(always)]
    pub fn str(self) -> Result<Str<'a>, Error> {
        scalar!(self.0, Kind::String, string)
    }

    /// Takes a number.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not a number, or the refusal of the input.
    #[inline(always)]
    pub fn number(self) -> Result<Number<'a>, Error> {
        scalar!(self.0, Kind::Number, number)
    }

    /// Takes `true` or `false`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not `true` or `false`, or the refusal of the input.
    #[inline(always)]
    pub fn bool(self) -> Result<bool, Error> {
        scalar!(self.0, Kind::Bool, literal)
    }

    /// Takes `null`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not `null`, or the refusal of the input.
    #[inline(always)]
    pub fn null(self) -> Result<(), Error> {
        scalar!(self.0, Kind::Null, literal).map(drop)
    }

    /// Takes the value, of any type, with each check of the profile.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if no value comes next, or the refusal of the input.
    #[inline(always)]
    pub fn skip(self) -> Result<(), Error> {
        let reader = self.0;
        let depth = reader.depth;
        let (at, kind) = reader.next_value()?;
        let token = reader.take(at, kind);
        reader.check(token)?;
        while reader.depth > depth {
            reader.token()?;
        }
        Ok(())
    }

    /// Takes the `{` of an object, to walk its members with [`Members::next`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not an object, or the refusal of the input.
    #[inline(always)]
    pub fn members(self) -> Result<Members<'r, 'a, 's>, Error> {
        self.open(Kind::Object).map(|reader| Members(Walk::new(reader)))
    }

    /// Takes the `[` of an array, to walk its elements with [`Elements::next`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Type`] if the value is not an array, or the refusal of the input.
    #[inline(always)]
    pub fn elements(self) -> Result<Elements<'r, 'a, 's>, Error> {
        self.open(Kind::Array).map(|reader| Elements(Walk::new(reader)))
    }

    /// Takes the start of the container of `kind`.
    #[inline(always)]
    fn open(self, kind: Kind) -> Result<&'r mut Reader<'a, 's>, Error> {
        let reader = self.0;
        let at = reader.expect(kind)?;
        let token = reader.take(at, kind);
        reader.check(token)?;
        Ok(reader)
    }
}

/// The walk of one open container, at its depth.
#[derive(Debug)]
struct Walk<'r, 'a, 's> {
    reader: &'r mut Reader<'a, 's>,
    depth: u16,
}

impl<'a, 's> Walk<'_, 'a, 's> {
    #[inline(always)]
    fn new<'r>(reader: &'r mut Reader<'a, 's>) -> Walk<'r, 'a, 's> {
        let depth = reader.depth;
        Walk { reader, depth }
    }

    /// The reader at the next item of the container, or `None` after its end. It takes the rest of
    /// the item before, which the loop did not read: a value that it left, or the rest of a
    /// container that it opened. Each skip gets each check of the profile. A key or the end of an
    /// object is the token of the reader that it gives, so it gives that token.
    #[inline(always)]
    fn next(&mut self) -> Result<Option<(&mut Reader<'a, 's>, Option<Token<'a>>)>, Error> {
        let reader = &mut *self.reader;
        loop {
            if reader.depth < self.depth {
                return Ok(None);
            }
            // The walk gives an item in `State::Value`, so a value in that state is one that the
            // loop left.
            let item = reader.depth == self.depth && reader.state != State::Value;
            if item && reader.frame.container == Container::Array {
                return Ok(Some((reader, None)));
            }
            let token = reader.token()?;
            if item {
                return Ok(Some((reader, token)));
            }
        }
    }
}

/// The members of an object, which [`Value::members`] opens.
#[derive(Debug)]
pub struct Members<'r, 'a, 's>(Walk<'r, 'a, 's>);

impl<'a, 's> Members<'_, 'a, 's> {
    /// The key and the value of the next member, or `None` after the `}`. The value is for this
    /// pass of the loop. The next call skips what the loop did not read of it. The walk refuses a
    /// key that equals an earlier key of the object, after unescape.
    ///
    /// # Errors
    ///
    /// The refusal of the input, also in what the call skips.
    #[inline(always)]
    #[expect(
        clippy::should_implement_trait,
        reason = "each value borrows the walk until the next call, which `Iterator` cannot express"
    )]
    pub fn next(&mut self) -> Result<Option<(Str<'a>, Value<'_, 'a, 's>)>, Error> {
        // After `{` or a member, the grammar gives only a key or the end of the object.
        Ok(match self.0.next()? {
            Some((reader, Some(Token::Key(key)))) => Some((key, Value(reader))),
            _ => None,
        })
    }
}

/// The elements of an array, which [`Value::elements`] opens.
#[derive(Debug)]
pub struct Elements<'r, 'a, 's>(Walk<'r, 'a, 's>);

impl<'a, 's> Elements<'_, 'a, 's> {
    /// The next element, or `None` after the `]`. The value is for this pass of the loop. The next
    /// call skips what the loop did not read of it.
    ///
    /// # Errors
    ///
    /// The refusal of the input, also in what the call skips.
    #[inline(always)]
    #[expect(
        clippy::should_implement_trait,
        reason = "each value borrows the walk until the next call, which `Iterator` cannot express"
    )]
    pub fn next(&mut self) -> Result<Option<Value<'_, 'a, 's>>, Error> {
        let Some((reader, _)) = self.0.next()? else {
            return Ok(None);
        };
        // In an array, the grammar gives only a value or the end of the array.
        if let Item::Value(..) = reader.item()? {
            // After `[`, the state allows a `]` too. A value comes, so it allows only a value.
            reader.state = State::Value;
            return Ok(Some(Value(reader)));
        }
        let token = reader.close();
        reader.check(token).map(|_| None)
    }
}
