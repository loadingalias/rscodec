//! The contracts of `rscodec::hex`: the RFC 4648 vectors, the strict lowercase profile, the error
//! and its offset, the same result on each backend and in each timing class, and the wipe and the
//! DIT state of the secret class.

mod common;

use common::Rng;
use rscodec::hex::{self, Backend, Error, ErrorKind, Kernel};

/// A call of the codec: on a backend, or a function of a timing class, with no backend argument.
#[derive(Clone, Copy, Debug)]
enum Call {
    With(Backend),
    Public,
    Secret,
}

/// Each backend that this CPU runs, then the functions of each timing class.
fn calls() -> Vec<Call> {
    let backends = Kernel::ALL.iter().filter_map(|&kernel| Backend::new(kernel));
    backends.map(Call::With).chain([Call::Public, Call::Secret]).collect()
}

/// Encodes `bytes` into `out` with `call`, and gives the length of the text.
fn encode_into(call: Call, bytes: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => hex::encode_with(backend, bytes, out),
        Call::Public => hex::encode(bytes, out),
        Call::Secret => hex::encode_secret(bytes, out),
    }
    .map(|text| text.len())
}

/// Decodes `text` into `out` with `call`, and gives the length of the bytes.
fn decode_into(call: Call, text: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => hex::decode_with(backend, text, out),
        Call::Public => hex::decode(text, out),
        Call::Secret => hex::decode_secret(text, out),
    }
    .map(|bytes| bytes.len())
}

fn encode(call: Call, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0; hex::encoded_len(bytes.len()).unwrap()];
    assert_eq!(encode_into(call, bytes, &mut out), Ok(out.len()));
    out
}

fn decode(call: Call, text: &[u8]) -> Result<Vec<u8>, Error> {
    // Room for more than the result, so that a decoder that writes past it fails the comparison.
    let mut out = vec![0x5a; text.len()];
    let len = decode_into(call, text, &mut out)?;
    assert_eq!(Ok(len), hex::decoded_len(text));
    Ok(out[..len].to_vec())
}

/// RFC 4648, section 10, in lowercase. The uppercase text of the RFC refuses at its first letter.
#[test]
fn rfc_4648_vectors() {
    let vectors = [
        ("", ""),
        ("f", "66"),
        ("fo", "666F"),
        ("foo", "666F6F"),
        ("foob", "666F6F62"),
        ("fooba", "666F6F6261"),
        ("foobar", "666F6F626172"),
    ];
    for call in calls() {
        for (bytes, text) in vectors {
            let lower = text.to_ascii_lowercase();
            assert_eq!(encode(call, bytes.as_bytes()), lower.as_bytes(), "{call:?}");
            assert_eq!(decode(call, lower.as_bytes()).unwrap(), bytes.as_bytes());
            let upper = text.find('F').map(|offset| Error {
                kind: ErrorKind::Symbol,
                offset,
            });
            assert_eq!(decode(call, text.as_bytes()).err(), upper, "{call:?}");
        }
    }
}

/// Each input of 0 to 2 bytes round-trips. Each change of one character of the text of each input
/// of one byte to any byte decodes only to the byte of its lowercase digits, and refuses at the
/// changed place if not.
#[test]
fn short_inputs_round_trip_and_each_other_byte_refuses() {
    let calls = calls();
    for value in 0..=u16::MAX {
        let bytes = value.to_le_bytes();
        for len in [0, 1, 2] {
            let text = data_encoding::HEXLOWER.encode(&bytes[..len]).into_bytes();
            for &call in &calls {
                assert_eq!(encode(call, &bytes[..len]), text);
                assert_eq!(decode(call, &text).unwrap(), bytes[..len]);
            }
        }
    }
    for value in 0..=u8::MAX {
        let text = data_encoding::HEXLOWER.encode(&[value]).into_bytes();
        for place in 0..2 {
            for byte in 0..=u8::MAX {
                let mut changed = text.clone();
                changed[place] = byte;
                let digit = byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
                for &call in &calls {
                    let decoded = decode(call, &changed);
                    if digit {
                        assert_eq!(data_encoding::HEXLOWER.encode(&decoded.unwrap()).as_bytes(), changed);
                    } else {
                        assert_eq!(
                            decoded,
                            Err(Error {
                                kind: ErrorKind::Symbol,
                                offset: place
                            }),
                            "{call:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Each backend and each timing class gives the result and the error of the portable backend: on
/// random inputs of each length to 64 bytes, on each byte at each place of their texts, and on
/// random inputs to 1 MiB.
#[test]
fn each_backend_equals_the_portable_backend() {
    let portable = Call::With(Backend::new(Kernel::Portable).unwrap());
    let calls = calls();
    let mut rng = Rng::new(0x0068_6578);
    for len in (0..=64).chain([1000, 4096, 65_535, 65_536, 1 << 20]) {
        let bytes = rng.bytes(len);
        let text = encode(portable, &bytes);
        let mut texts = vec![text.clone(), [&text[..], b"0"].concat()];
        for place in 0..text.len() {
            // Each byte value at a seventh of the places of each length, so that over the lengths
            // each place meets each byte. A random byte at the other places.
            let small = len <= 64 && place % 7 == len % 7;
            for byte in if small { 0..=255 } else { 0..=0 } {
                let mut changed = text.clone();
                changed[place] = if small { byte } else { rng.byte() };
                texts.push(changed);
            }
            if len > 64 && texts.len() > 16 {
                break;
            }
        }
        for &call in &calls {
            assert_eq!(encode(call, &bytes), text, "{call:?} {len}");
            for changed in &texts {
                assert_eq!(decode(call, changed), decode(portable, changed), "{call:?} {len}");
            }
        }
    }
}

/// A short output slice refuses with the length of the result, and the decoder checks the length
/// first.
#[test]
fn a_short_output_refuses_with_the_length() {
    let error = |kind, offset| Err(Error { kind, offset });
    for call in calls() {
        let mut out = [0; 3];
        assert_eq!(encode_into(call, &[1; 2], &mut out), error(ErrorKind::Output, 4));
        assert_eq!(decode_into(call, b"00000000", &mut out), error(ErrorKind::Output, 4));
        assert_eq!(decode_into(call, b"000", &mut out), error(ErrorKind::Length, 3));
    }
}

/// After `decode_secret` refuses a text for a symbol, each byte of the result, which it wrote, is
/// zero, and it wrote no other byte. A refusal for the length or the output writes no byte.
#[test]
fn decode_secret_wipes_its_output_after_a_refusal() {
    const UNWRITTEN: u8 = 0x5a;
    let mut rng = Rng::new(0x0000_7769_7065);
    // The lengths of each tail and of whole blocks of each kernel.
    for len in 1..=200 {
        let text = data_encoding::HEXLOWER.encode(&rng.bytes(len)).into_bytes();
        for place in [0, 2 * len - 1] {
            let mut changed = text.clone();
            // `g` is not a hex digit.
            changed[place] = b'g';
            let mut out = vec![UNWRITTEN; len + 4];
            assert_eq!(
                hex::decode_secret(&changed, &mut out),
                Err(Error {
                    kind: ErrorKind::Symbol,
                    offset: place
                }),
                "{len}"
            );
            assert!(out[..len].iter().all(|&byte| byte == 0), "{len} {place}");
            assert!(out[len..].iter().all(|&byte| byte == UNWRITTEN), "{len}");
        }
    }
    let mut out = [UNWRITTEN; 8];
    let error = |kind, offset| Err(Error { kind, offset });
    assert_eq!(hex::decode_secret(b"000", &mut out), error(ErrorKind::Length, 3));
    assert_eq!(hex::decode_secret(&[b'g'; 18], &mut out), error(ErrorKind::Output, 9));
    assert_eq!(out, [UNWRITTEN; 8]);
}

/// On aarch64 with DIT, `PSTATE.DIT` after each secret call equals its value before the call, set
/// or clear, after a result and after a refusal.
#[cfg(target_arch = "aarch64")]
#[test]
fn each_secret_call_leaves_pstate_dit_as_it_was() {
    use common::dit;
    // A CPU without DIT has no state to keep.
    let Some(before) = dit::get() else { return };
    for held in [false, true] {
        dit::set(held);
        let kept = || assert_eq!(dit::get(), Some(held));
        let error = |kind, offset| Err(Error { kind, offset });
        let mut out = [0; 64];
        assert_eq!(hex::encode_secret(&[0xa5; 32], &mut out).map(|text| text.len()), Ok(64));
        kept();
        assert_eq!(
            hex::encode_secret(&[0xa5; 32], &mut out[..3]).map(|text| text.len()),
            error(ErrorKind::Output, 64)
        );
        kept();
        assert_eq!(
            hex::decode_secret(&[b'a'; 64], &mut out).map(|bytes| bytes.len()),
            Ok(32)
        );
        kept();
        assert_eq!(
            hex::decode_secret(&[b'g'; 64], &mut out).map(|bytes| bytes.len()),
            error(ErrorKind::Symbol, 0)
        );
        kept();
    }
    dit::set(before);
}
