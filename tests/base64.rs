//! The contracts of `rscodec::base64`: the RFC 4648 vectors, the strict profiles, the error and
//! its offset, and the same result on each backend.

mod common;

use common::{Rng, backends};
use rscodec::{
    Isa,
    base64::{Error, ErrorKind, Profile},
};

const PROFILES: [Profile; 2] = [Profile::Url, Profile::Standard];

/// The characters of the alphabet of `profile`, from RFC 4648, tables 1 and 2.
fn alphabet(profile: Profile) -> &'static [u8; 64] {
    match profile {
        Profile::Url => b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
        Profile::Standard => b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
    }
}

/// The text of `bytes` from data-encoding, an independent implementation.
fn oracle(profile: Profile, bytes: &[u8]) -> Vec<u8> {
    match profile {
        Profile::Url => data_encoding::BASE64URL_NOPAD.encode(bytes),
        Profile::Standard => data_encoding::BASE64.encode(bytes),
    }
    .into_bytes()
}

fn encode(profile: Profile, backend: rscodec::Backend, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0; profile.encoded_len(bytes.len()).unwrap()];
    let len = profile.encode_with(backend, bytes, &mut out).unwrap().len();
    assert_eq!(len, out.len());
    out
}

fn decode(profile: Profile, backend: rscodec::Backend, text: &[u8]) -> Result<Vec<u8>, Error> {
    // Room for more than the result, so that a decoder that writes past it fails the comparison.
    let mut out = vec![0x5a; text.len()];
    let bytes = profile.decode_with(backend, text, &mut out)?.to_vec();
    assert_eq!(Ok(bytes.len()), profile.decoded_len(text));
    Ok(bytes)
}

/// The error that a strict decoder gives for `text`, from the rules of the profile: the length,
/// then the first byte outside the alphabet, then the trailing bits.
fn expected_error(profile: Profile, text: &[u8]) -> Option<Error> {
    let error = |kind, offset| Some(Error { kind, offset });
    let symbols = match profile {
        Profile::Url if text.len() % 4 == 1 => return error(ErrorKind::Length, text.len()),
        Profile::Url => text,
        Profile::Standard if text.len() % 4 != 0 => return error(ErrorKind::Length, text.len()),
        Profile::Standard => text
            .strip_suffix(b"==")
            .or_else(|| text.strip_suffix(b"="))
            .unwrap_or(text),
    };
    let value = |byte| alphabet(profile).iter().position(|&c| c == byte);
    if let Some(offset) = symbols.iter().position(|&byte| value(byte).is_none()) {
        return error(ErrorKind::Symbol, offset);
    }
    let bits = symbols.len() * 6 % 8;
    let last = symbols.last().map_or(0, |&byte| value(byte).unwrap());
    (last & ((1 << bits) - 1) != 0).then(|| Error {
        kind: ErrorKind::TrailingBits,
        offset: symbols.len() - 1,
    })
}

/// RFC 4648, section 10.
#[test]
fn rfc_4648_vectors() {
    let vectors = [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ];
    for backend in backends() {
        for (bytes, text) in vectors {
            assert_eq!(
                encode(Profile::Standard, backend, bytes.as_bytes()),
                text.as_bytes(),
                "{backend:?}"
            );
            assert_eq!(
                decode(Profile::Standard, backend, text.as_bytes()).unwrap(),
                bytes.as_bytes()
            );
            let url = text.trim_end_matches('=');
            assert_eq!(
                encode(Profile::Url, backend, bytes.as_bytes()),
                url.as_bytes(),
                "{backend:?}"
            );
            assert_eq!(decode(Profile::Url, backend, url.as_bytes()).unwrap(), bytes.as_bytes());
        }
    }
}

/// Each input of 0 to 3 bytes round-trips, in one text and in a call of its own. Each change to
/// the text of a short input, which replaces one character with any byte or removes or adds the
/// padding, decodes only if it is the canonical text of other bytes, and refuses with its offset
/// if not.
#[test]
fn short_inputs_round_trip_and_each_noncanonical_form_refuses() {
    let backends = backends();
    for profile in PROFILES {
        // Each input of 3 bytes in one text: the 3 bytes of each value are one quantum of 4
        // characters, which an encoder or a decoder maps on its own.
        let all: Vec<u8> = (0..1_u32 << 24)
            .flat_map(|value| <[u8; 3]>::try_from(&value.to_le_bytes()[..3]).unwrap())
            .collect();
        // data-encoding takes at most `usize::MAX / 512` bytes in one call: 8 MiB on wasm32.
        let text: Vec<u8> = all.chunks(3 << 20).flat_map(|chunk| oracle(profile, chunk)).collect();
        for &backend in &backends {
            assert_eq!(encode(profile, backend, &all), text, "{profile:?} {backend:?}");
            assert_eq!(decode(profile, backend, &text).unwrap(), all, "{profile:?} {backend:?}");
        }
        let mut rng = Rng::new(0x0053_686f_7274);
        for len in 0..=3 {
            let mut bytes = vec![0; len];
            for value in 0..1_u32 << (8 * len.min(2)) {
                bytes.copy_from_slice(&value.to_le_bytes()[..len]);
                if len == 3 {
                    bytes[2] = rng.byte();
                }
                let text = oracle(profile, &bytes);
                for &backend in &backends {
                    assert_eq!(encode(profile, backend, &bytes), text);
                    assert_eq!(decode(profile, backend, &text).unwrap(), bytes);
                }
                // The changes of each text of one byte, and of a sample of the longer texts.
                if len > 1 && value % 257 != 0 {
                    continue;
                }
                let mut texts = vec![
                    [&text[..], b"="].concat(),
                    text.strip_suffix(b"=").unwrap_or(&[]).to_vec(),
                ];
                for place in 0..text.len() {
                    for byte in 0..=255 {
                        let mut changed = text.clone();
                        changed[place] = byte;
                        texts.push(changed);
                    }
                }
                for changed in texts {
                    let expected = expected_error(profile, &changed);
                    for &backend in &backends {
                        let decoded = decode(profile, backend, &changed);
                        match expected {
                            Some(error) => assert_eq!(decoded, Err(error), "{profile:?} {backend:?} {changed:?}"),
                            None => assert_eq!(oracle(profile, &decoded.unwrap()), changed),
                        }
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
    let portable = rscodec::Backend::new(Isa::Portable).unwrap();
    let backends = backends();
    let mut rng = Rng::new(0x6261_7365_3634);
    for profile in PROFILES {
        for len in (0..=64).chain([1000, 4096, 65_535, 65_536, 1 << 20]) {
            let bytes = rng.bytes(len);
            let text = encode(profile, portable, &bytes);
            let mut texts = vec![text.clone()];
            for place in 0..text.len() {
                // Each byte value at a seventh of the places of each length, so that over the
                // lengths each place meets each byte. A random byte at the other places.
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
                assert_eq!(encode(profile, backend, &bytes), text, "{profile:?} {backend:?} {len}");
                for changed in &texts {
                    assert_eq!(
                        decode(profile, backend, changed),
                        decode(profile, portable, changed),
                        "{profile:?} {backend:?} {len}"
                    );
                }
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
            Profile::Url.encode_with(backend, &[1; 3], &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            Profile::Standard.encode_with(backend, &[1; 1], &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            Profile::Url.decode_with(backend, b"AAAAAA", &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            Profile::Url.decode_with(backend, b"AAAAA", &mut out),
            error(ErrorKind::Length, 5)
        );
        assert_eq!(
            Profile::Standard.decode_with(backend, b"AAAAAA", &mut out),
            error(ErrorKind::Length, 6)
        );
    }
}
