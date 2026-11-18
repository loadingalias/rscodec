//! The contracts of `rscodec::hex`: the RFC 4648 vectors, the strict lowercase profile, the error
//! and its offset, and the same result on each backend.

mod common;

use common::{Rng, backends};
use rscodec::{
    Backend, Isa,
    hex::{self, Error, ErrorKind},
};

fn encode(backend: Backend, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0; hex::encoded_len(bytes.len()).unwrap()];
    let len = hex::encode_with(backend, bytes, &mut out).unwrap().len();
    assert_eq!(len, out.len());
    out
}

fn decode(backend: Backend, text: &[u8]) -> Result<Vec<u8>, Error> {
    // Room for more than the result, so that a decoder that writes past it fails the comparison.
    let mut out = vec![0x5a; text.len()];
    let bytes = hex::decode_with(backend, text, &mut out)?.to_vec();
    assert_eq!(Ok(bytes.len()), hex::decoded_len(text));
    Ok(bytes)
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
    for backend in backends() {
        for (bytes, text) in vectors {
            let lower = text.to_ascii_lowercase();
            assert_eq!(encode(backend, bytes.as_bytes()), lower.as_bytes(), "{backend:?}");
            assert_eq!(decode(backend, lower.as_bytes()).unwrap(), bytes.as_bytes());
            let upper = text.find('F').map(|offset| Error {
                kind: ErrorKind::Symbol,
                offset,
            });
            assert_eq!(decode(backend, text.as_bytes()).err(), upper, "{backend:?}");
        }
    }
}

/// Each input of 0 to 2 bytes round-trips. Each change of one character of the text of each input
/// of one byte to any byte decodes only to the byte of its lowercase digits, and refuses at the
/// changed place if not.
#[test]
fn short_inputs_round_trip_and_each_other_byte_refuses() {
    let backends = backends();
    for value in 0..=u16::MAX {
        let bytes = value.to_le_bytes();
        for len in [0, 1, 2] {
            let text = data_encoding::HEXLOWER.encode(&bytes[..len]).into_bytes();
            for &backend in &backends {
                assert_eq!(encode(backend, &bytes[..len]), text);
                assert_eq!(decode(backend, &text).unwrap(), bytes[..len]);
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
                for &backend in &backends {
                    let decoded = decode(backend, &changed);
                    if digit {
                        assert_eq!(data_encoding::HEXLOWER.encode(&decoded.unwrap()).as_bytes(), changed);
                    } else {
                        assert_eq!(
                            decoded,
                            Err(Error {
                                kind: ErrorKind::Symbol,
                                offset: place
                            }),
                            "{backend:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Each backend gives the result and the error of the portable backend: on random inputs of each
/// length to 64 bytes, on each byte at each place of their texts, and on random inputs to 1 MiB.
#[test]
fn each_backend_equals_the_portable_backend() {
    let portable = Backend::new(Isa::Portable).unwrap();
    let backends = backends();
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
        for &backend in &backends {
            assert_eq!(encode(backend, &bytes), text, "{backend:?} {len}");
            for changed in &texts {
                assert_eq!(decode(backend, changed), decode(portable, changed), "{backend:?} {len}");
            }
        }
    }
}

/// A short output slice refuses with the length of the result, and the decoder checks the length
/// first.
#[test]
fn a_short_output_refuses_with_the_length() {
    for backend in backends() {
        let mut out = [0; 3];
        let error = |kind, offset| Err(Error { kind, offset });
        assert_eq!(
            hex::encode_with(backend, &[1; 2], &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            hex::decode_with(backend, b"00000000", &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(hex::decode_with(backend, b"000", &mut out), error(ErrorKind::Length, 3));
    }
}
