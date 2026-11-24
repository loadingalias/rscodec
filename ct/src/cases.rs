//! The calls under test and their inputs: each base-N function of each class, and each kernel of
//! the CPU through the `_with` functions.

use rscodec::{
    Dit,
    base64::{self, Profile},
    hex,
};

use crate::{Failure, random::Random};

/// A codec of the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    Hex,
    Base64(Profile),
}

impl Codec {
    const fn name(self) -> &'static str {
        match self {
            Self::Hex => "hex",
            Self::Base64(Profile::Url) => "url",
            Self::Base64(Profile::Standard) => "standard",
        }
    }

    /// Whether `byte` is a character of the alphabet.
    const fn has(self, byte: u8) -> bool {
        match self {
            Self::Hex => matches!(byte, b'0'..=b'9' | b'a'..=b'f'),
            Self::Base64(Profile::Url) => byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'),
            Self::Base64(Profile::Standard) => byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'),
        }
    }

    /// The text of `bytes`, from the public encoder.
    fn text(self, bytes: &[u8]) -> Result<Vec<u8>, Failure> {
        let len = match self {
            Self::Hex => hex::encoded_len(bytes.len()),
            Self::Base64(profile) => profile.encoded_len(bytes.len()),
        };
        let mut text = vec![0; len.ok_or_else(|| Failure::Host(format!("no text of {} bytes", bytes.len())))?];
        let written = match self {
            Self::Hex => hex::encode(bytes, &mut text)
                .map(|text| text.len())
                .map_err(|error| error.to_string()),
            Self::Base64(profile) => profile
                .encode(bytes, &mut text)
                .map(|text| text.len())
                .map_err(|error| error.to_string()),
        };
        written.map_err(Failure::Host)?;
        Ok(text)
    }

    /// The outcome of the public decoder for `text`, the text of `bytes` bytes.
    fn outcome(self, text: &[u8], bytes: usize) -> Result<Outcome, Failure> {
        let mut out = vec![0; bytes];
        let outcome = match self {
            Self::Hex => match hex::decode(text, &mut out) {
                Ok(_) => Some(Outcome::Valid),
                Err(error) => (error.kind() == hex::ErrorKind::Symbol).then_some(Outcome::Symbol),
            },
            Self::Base64(profile) => match profile.decode(text, &mut out) {
                Ok(_) => Some(Outcome::Valid),
                Err(error) => match error.kind() {
                    base64::ErrorKind::Symbol => Some(Outcome::Symbol),
                    base64::ErrorKind::TrailingBits => Some(Outcome::TrailingBits),
                    base64::ErrorKind::Length | base64::ErrorKind::Output => None,
                },
            },
        };
        outcome.ok_or_else(|| {
            Failure::Check(format!(
                "the {} text of {bytes} bytes has a refusal for its length",
                self.name()
            ))
        })
    }
}

/// The direction of a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Encode,
    Decode,
}

impl Direction {
    const fn name(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Decode => "decode",
        }
    }
}

/// The outcome of a call. The caller sees it, so the time may depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Valid,
    /// A refusal for a character outside the alphabet.
    Symbol,
    /// A refusal of base64 for nonzero bits after the last byte.
    TrailingBits,
}

/// The data of an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fill {
    Zero,
    Ones,
    Random,
}

/// The timing class of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Secret,
    Public,
}

/// The function of a case.
#[derive(Clone, Copy, Debug)]
enum Call {
    Plain(entry::Plain),
    Hex(entry::HexWith, hex::Backend),
    Base64(entry::Base64With, base64::Backend),
}

/// One function of one codec, in one class or on one kernel.
#[derive(Debug)]
pub(crate) struct Case {
    pub(crate) name: String,
    pub(crate) class: Class,
    codec: Codec,
    pub(crate) direction: Direction,
    call: Call,
}

impl Case {
    /// Each case of this CPU: the secret and the public function of each codec and direction, and
    /// the `_with` function on each kernel that the CPU runs.
    pub(crate) fn all() -> Vec<Self> {
        let mut cases = Vec::new();
        for codec in [
            Codec::Hex,
            Codec::Base64(Profile::Url),
            Codec::Base64(Profile::Standard),
        ] {
            for direction in [Direction::Encode, Direction::Decode] {
                let mut push = |class, label: &str, call| {
                    cases.push(Self {
                        name: format!("{}-{}-{label}", codec.name(), direction.name()),
                        class,
                        codec,
                        direction,
                        call,
                    });
                };
                let (secret, public) = entry::plain(codec, direction);
                push(Class::Secret, "secret", Call::Plain(secret));
                push(Class::Public, "public", Call::Plain(public));
                match codec {
                    Codec::Hex => {
                        for &kernel in hex::Kernel::ALL {
                            if let Some(backend) = hex::Backend::new(kernel) {
                                let label = format!("{kernel:?}").to_lowercase();
                                push(Class::Public, &label, Call::Hex(entry::hex_with(direction), backend));
                            }
                        }
                    }
                    Codec::Base64(profile) => {
                        for &kernel in base64::Kernel::ALL {
                            if let Some(backend) = base64::Backend::new(kernel) {
                                let label = format!("{kernel:?}").to_lowercase();
                                let call = Call::Base64(entry::base64_with(profile, direction), backend);
                                push(Class::Public, &label, call);
                            }
                        }
                    }
                }
            }
        }
        cases
    }

    /// The outcomes that an input of the case can have.
    pub(crate) const fn outcomes(&self) -> &'static [Outcome] {
        match (self.direction, self.codec) {
            (Direction::Encode, _) => &[Outcome::Valid],
            (Direction::Decode, Codec::Hex) => &[Outcome::Valid, Outcome::Symbol],
            (Direction::Decode, Codec::Base64(_)) => &[Outcome::Valid, Outcome::Symbol, Outcome::TrailingBits],
        }
    }

    /// The length of the output for `bytes` bytes of data.
    pub(crate) fn out_len(&self, bytes: usize) -> Result<usize, Failure> {
        match self.direction {
            Direction::Encode => Ok(self.codec.text(&vec![0; bytes])?.len()),
            Direction::Decode => Ok(bytes),
        }
    }

    /// An input of the case for `bytes` bytes of `fill` with `outcome`, or `None` if no input of
    /// this length has the outcome. A refusal for a symbol puts one byte outside the alphabet at
    /// the offset `at` of the characters (the last for an offset past them), or at a random offset.
    pub(crate) fn input(
        &self,
        bytes: usize,
        outcome: Outcome,
        fill: Fill,
        at: Option<usize>,
        random: &mut Random,
    ) -> Result<Option<Vec<u8>>, Failure> {
        let mut data = vec![0; bytes];
        match fill {
            Fill::Zero => {}
            Fill::Ones => data.fill(0xff),
            Fill::Random => random.fill(&mut data),
        }
        if self.direction == Direction::Encode {
            return Ok((outcome == Outcome::Valid).then_some(data));
        }
        let mut text = self.codec.text(&data)?;
        let symbols = text.len() - text.iter().rev().take_while(|&&byte| byte == b'=').count();
        let slot = match outcome {
            Outcome::Valid => None,
            Outcome::Symbol if symbols == 0 => return Ok(None),
            Outcome::Symbol => {
                let offset = at.map_or_else(|| random.below(symbols), |at| at.min(symbols - 1));
                let bad = loop {
                    let [byte, ..] = random.next().to_le_bytes();
                    if !self.codec.has(byte) && byte != b'=' {
                        break byte;
                    }
                };
                Some((offset, bad))
            }
            Outcome::TrailingBits if self.codec == Codec::Hex || bytes.is_multiple_of(3) => return Ok(None),
            // 'B' is the sextet 1: its lowest bit is a trailing bit for 1 and for 2 bytes.
            Outcome::TrailingBits => Some((symbols - 1, b'B')),
        };
        if let Some((offset, byte)) = slot
            && let Some(character) = text.get_mut(offset)
        {
            *character = byte;
        }
        let seen = self.codec.outcome(&text, bytes)?;
        if seen != outcome {
            return Err(Failure::Check(format!(
                "the {} input of {bytes} bytes has the outcome {seen:?}, not {outcome:?}",
                self.name
            )));
        }
        Ok(Some(text))
    }

    /// Replaces the codec with a copy over its full input and output ranges.
    pub(crate) fn use_buffer_control(&mut self) {
        self.name.push_str("-buffers");
        self.call = Call::Plain(|_, input, out| {
            if input.is_empty() || out.is_empty() {
                return 0;
            }
            if input.len() <= out.len() {
                for dst in out.chunks_mut(input.len()) {
                    dst.copy_from_slice(&input[..dst.len()]);
                }
            } else {
                for src in input.chunks(out.len()) {
                    out[..src.len()].copy_from_slice(src);
                    // Keep each store observable before the next chunk overwrites it.
                    core::hint::black_box(&*out);
                }
            }
            core::hint::black_box(out).len()
        });
    }

    /// Runs the function of the case. It gives the length of the result, or `usize::MAX` for a
    /// refusal.
    #[inline(always)]
    pub(crate) fn call(&self, dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        match self.call {
            Call::Plain(entry) => entry(dit, input, out),
            Call::Hex(entry, backend) => entry(backend, input, out),
            Call::Base64(entry, backend) => entry(backend, input, out),
        }
    }
}

/// The functions that the harness calls. Each is out of line, so that the scan finds the code of
/// each public function of rscodec by the name of its entry. Each gives the length of its result,
/// or `usize::MAX` for a refusal.
mod entry {
    use rscodec::{
        Dit,
        base64::{self, Profile},
        hex,
    };

    use super::{Codec, Direction};

    pub(super) type Plain = fn(&Dit, &[u8], &mut [u8]) -> usize;
    pub(super) type HexWith = fn(hex::Backend, &[u8], &mut [u8]) -> usize;
    pub(super) type Base64With = fn(base64::Backend, &[u8], &mut [u8]) -> usize;

    /// The length of the result, or `usize::MAX` for a refusal. A caller reads the kind and the
    /// offset of a refusal, so the whole result goes through `black_box`: else the compiler drops
    /// the work of the offset, and the harness would not run the code of a caller (C103).
    fn length<E>(result: Result<&mut [u8], E>) -> usize {
        core::hint::black_box(result).map_or(usize::MAX, |out| out.len())
    }

    /// The secret and the public function of `codec` in `direction`.
    pub(super) const fn plain(codec: Codec, direction: Direction) -> (Plain, Plain) {
        match (codec, direction) {
            (Codec::Hex, Direction::Encode) => (hex_encode_secret, hex_encode),
            (Codec::Hex, Direction::Decode) => (hex_decode_secret, hex_decode),
            (Codec::Base64(Profile::Url), Direction::Encode) => (url_encode_secret, url_encode),
            (Codec::Base64(Profile::Url), Direction::Decode) => (url_decode_secret, url_decode),
            (Codec::Base64(Profile::Standard), Direction::Encode) => (standard_encode_secret, standard_encode),
            (Codec::Base64(Profile::Standard), Direction::Decode) => (standard_decode_secret, standard_decode),
        }
    }

    /// The `_with` function of hex in `direction`.
    pub(super) const fn hex_with(direction: Direction) -> HexWith {
        match direction {
            Direction::Encode => hex_encode_with,
            Direction::Decode => hex_decode_with,
        }
    }

    /// The `_with` function of base64 for `profile` in `direction`.
    pub(super) const fn base64_with(profile: Profile, direction: Direction) -> Base64With {
        match (profile, direction) {
            (Profile::Url, Direction::Encode) => url_encode_with,
            (Profile::Url, Direction::Decode) => url_decode_with,
            (Profile::Standard, Direction::Encode) => standard_encode_with,
            (Profile::Standard, Direction::Decode) => standard_decode_with,
        }
    }

    #[inline(never)]
    fn hex_encode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::encode_secret(dit, input, out))
    }

    #[inline(never)]
    fn hex_decode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::decode_secret(dit, input, out))
    }

    #[inline(never)]
    fn url_encode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.encode_secret(dit, input, out))
    }

    #[inline(never)]
    fn url_decode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.decode_secret(dit, input, out))
    }

    #[inline(never)]
    fn standard_encode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.encode_secret(dit, input, out))
    }

    #[inline(never)]
    fn standard_decode_secret(dit: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.decode_secret(dit, input, out))
    }

    #[inline(never)]
    fn hex_encode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::encode(input, out))
    }

    #[inline(never)]
    fn hex_decode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::decode(input, out))
    }

    #[inline(never)]
    fn url_encode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.encode(input, out))
    }

    #[inline(never)]
    fn url_decode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.decode(input, out))
    }

    #[inline(never)]
    fn standard_encode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.encode(input, out))
    }

    #[inline(never)]
    fn standard_decode(_: &Dit, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.decode(input, out))
    }

    #[inline(never)]
    fn hex_encode_with(backend: hex::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::encode_with(backend, input, out))
    }

    #[inline(never)]
    fn hex_decode_with(backend: hex::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(hex::decode_with(backend, input, out))
    }

    #[inline(never)]
    fn url_encode_with(backend: base64::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.encode_with(backend, input, out))
    }

    #[inline(never)]
    fn url_decode_with(backend: base64::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Url.decode_with(backend, input, out))
    }

    #[inline(never)]
    fn standard_encode_with(backend: base64::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.encode_with(backend, input, out))
    }

    #[inline(never)]
    fn standard_decode_with(backend: base64::Backend, input: &[u8], out: &mut [u8]) -> usize {
        length(Profile::Standard.decode_with(backend, input, out))
    }
}
