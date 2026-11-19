//! The contracts of `rscodec::base64`: the RFC 4648 vectors, the strict profiles, the error and
//! its offset, the same result on each backend and in each timing class, and the wipe of the
//! secret class.

mod common;

use common::{Rng, secret};
use rscodec::base64::{Backend, Error, ErrorKind, Kernel, Profile};

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

/// The portable backend, the reference of each other call.
fn portable() -> Call {
    Call::With(Backend::new(Kernel::Portable).unwrap())
}

/// Encodes `bytes` into `out` with `call`, and gives the length of the text.
fn encode_into(profile: Profile, call: Call, bytes: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => profile.encode_with(backend, bytes, out),
        Call::Public => profile.encode(bytes, out),
        Call::Secret => secret(|dit| profile.encode_secret(dit, bytes, out)),
    }
    .map(|text| text.len())
}

/// Decodes `text` into `out` with `call`, and gives the length of the bytes.
fn decode_into(profile: Profile, call: Call, text: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => profile.decode_with(backend, text, out),
        Call::Public => profile.decode(text, out),
        Call::Secret => secret(|dit| profile.decode_secret(dit, text, out)),
    }
    .map(|bytes| bytes.len())
}

fn encode(profile: Profile, call: Call, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0; profile.encoded_len(bytes.len()).unwrap()];
    assert_eq!(encode_into(profile, call, bytes, &mut out), Ok(out.len()));
    out
}

fn decode(profile: Profile, call: Call, text: &[u8]) -> Result<Vec<u8>, Error> {
    // Room for more than the result, so that a decoder that writes past it fails the comparison.
    let mut out = vec![0x5a; text.len()];
    let len = decode_into(profile, call, text, &mut out)?;
    assert_eq!(Ok(len), profile.decoded_len(text));
    Ok(out[..len].to_vec())
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
    for call in calls() {
        for (bytes, text) in vectors {
            assert_eq!(
                encode(Profile::Standard, call, bytes.as_bytes()),
                text.as_bytes(),
                "{call:?}"
            );
            assert_eq!(
                decode(Profile::Standard, call, text.as_bytes()).unwrap(),
                bytes.as_bytes()
            );
            let url = text.trim_end_matches('=');
            assert_eq!(encode(Profile::Url, call, bytes.as_bytes()), url.as_bytes(), "{call:?}");
            assert_eq!(decode(Profile::Url, call, url.as_bytes()).unwrap(), bytes.as_bytes());
        }
    }
}

/// Each input of 0 to 3 bytes round-trips, in one text and in a call of its own. Each change to
/// the text of a short input, which replaces one character with any byte or removes or adds the
/// padding, decodes only if it is the canonical text of other bytes, and refuses with its offset
/// if not.
#[test]
fn short_inputs_round_trip_and_each_noncanonical_form_refuses() {
    let calls = calls();
    for profile in PROFILES {
        // Each input of 3 bytes in one text: the 3 bytes of each value are one quantum of 4
        // characters, which an encoder or a decoder maps on its own.
        let all: Vec<u8> = (0..1_u32 << 24)
            .flat_map(|value| <[u8; 3]>::try_from(&value.to_le_bytes()[..3]).unwrap())
            .collect();
        // data-encoding takes at most `usize::MAX / 512` bytes in one call: 8 MiB on wasm32.
        let text: Vec<u8> = all.chunks(3 << 20).flat_map(|chunk| oracle(profile, chunk)).collect();
        for &call in &calls {
            assert_eq!(encode(profile, call, &all), text, "{profile:?} {call:?}");
            assert_eq!(decode(profile, call, &text).unwrap(), all, "{profile:?} {call:?}");
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
                for &call in &calls {
                    assert_eq!(encode(profile, call, &bytes), text);
                    assert_eq!(decode(profile, call, &text).unwrap(), bytes);
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
                    for &call in &calls {
                        let decoded = decode(profile, call, &changed);
                        match expected {
                            Some(error) => assert_eq!(decoded, Err(error), "{profile:?} {call:?} {changed:?}"),
                            None => assert_eq!(oracle(profile, &decoded.unwrap()), changed),
                        }
                    }
                }
            }
        }
    }
}

/// Each text of 2 or 3 symbols of any byte, with its padding in the standard profile, decodes only
/// if it is the canonical text of its bytes, and refuses with the error of the rules if not. Thus
/// each noncanonical form of a short text refuses: each quantum of 4 symbols of the alphabet is the
/// text of one input of 3 bytes, and the decoder takes each quantum on its own.
#[test]
fn each_short_text_decodes_only_if_it_is_canonical() {
    let calls = calls();
    let mut out = [0; 3];
    for profile in PROFILES {
        for len in [2, 3] {
            for value in 0..1_u32 << (8 * len) {
                let mut padded = [b'='; 4];
                padded[..len].copy_from_slice(&value.to_le_bytes()[..len]);
                let text = if profile == Profile::Standard {
                    &padded
                } else {
                    &padded[..len]
                };
                let expected = expected_error(profile, text);
                for &call in &calls {
                    let decoded = decode_into(profile, call, text, &mut out);
                    match expected {
                        Some(error) => assert_eq!(decoded, Err(error), "{profile:?} {call:?} {text:x?}"),
                        None => assert_eq!(oracle(profile, &out[..decoded.unwrap()]), text, "{profile:?} {call:?}"),
                    }
                }
            }
        }
    }
}

/// Each backend and each timing class gives the result and the error of the portable backend: on
/// random inputs of each length to 96 bytes, past the short path of 64 bytes, on each byte at each
/// place of their texts, and on random inputs to 1 MiB.
#[test]
fn each_backend_equals_the_portable_backend() {
    let portable = portable();
    let calls = calls();
    let mut rng = Rng::new(0x6261_7365_3634);
    for profile in PROFILES {
        for len in (0..=96).chain([1000, 4096, 65_535, 65_536, 1 << 20]) {
            let bytes = rng.bytes(len);
            let text = encode(profile, portable, &bytes);
            let mut texts = vec![text.clone()];
            for place in 0..text.len() {
                // Each byte value at a seventh of the places of each length, so that over the
                // lengths each place meets each byte. A random byte at the other places.
                let small = len <= 96 && place % 7 == len % 7;
                for byte in if small { 0..=255 } else { 0..=0 } {
                    let mut changed = text.clone();
                    changed[place] = if small { byte } else { rng.byte() };
                    texts.push(changed);
                }
                if len > 96 && texts.len() > 16 {
                    break;
                }
            }
            for &call in &calls {
                assert_eq!(encode(profile, call, &bytes), text, "{profile:?} {call:?} {len}");
                for changed in &texts {
                    assert_eq!(
                        decode(profile, call, changed),
                        decode(profile, portable, changed),
                        "{profile:?} {call:?} {len}"
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
    let error = |kind, offset| Err(Error { kind, offset });
    for call in calls() {
        let mut out = [0; 3];
        assert_eq!(
            encode_into(Profile::Url, call, &[1; 3], &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            encode_into(Profile::Standard, call, &[1; 1], &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            decode_into(Profile::Url, call, b"AAAAAA", &mut out),
            error(ErrorKind::Output, 4)
        );
        assert_eq!(
            decode_into(Profile::Url, call, b"AAAAA", &mut out),
            error(ErrorKind::Length, 5)
        );
        assert_eq!(
            decode_into(Profile::Standard, call, b"AAAAAA", &mut out),
            error(ErrorKind::Length, 6)
        );
    }
}

/// After `decode_secret` refuses a text for a symbol or for its trailing bits, each byte of the
/// result, which it wrote, is zero, and it wrote no other byte. A refusal for the length or the
/// output writes no byte.
#[test]
fn decode_secret_wipes_its_output_after_a_refusal() {
    secret(|dit| {
        const UNWRITTEN: u8 = 0x5a;
        let mut rng = Rng::new(0x0000_7769_7065);
        for profile in PROFILES {
            let value = |byte| alphabet(profile).iter().position(|&c| c == byte).unwrap();
            // The lengths of each tail and of whole blocks of each kernel.
            for len in 1..=200 {
                let bytes = rng.bytes(len);
                let text = oracle(profile, &bytes);
                let last = text.iter().rposition(|&c| c != b'=').unwrap();
                let symbol = |at: usize, byte: u8| {
                    let mut changed = text.clone();
                    changed[at] = byte;
                    changed
                };
                // `*` is in neither alphabet.
                let mut refusals = vec![
                    (symbol(0, b'*'), ErrorKind::Symbol, 0),
                    (symbol(last, b'*'), ErrorKind::Symbol, last),
                ];
                // The last character holds bits after the last byte unless the bytes fill 3n bytes.
                // Bit 0 is one of them.
                if len % 3 != 0 {
                    let odd = alphabet(profile)[value(text[last]) | 1];
                    refusals.push((symbol(last, odd), ErrorKind::TrailingBits, last));
                }
                for (changed, kind, offset) in refusals {
                    let mut out = vec![UNWRITTEN; len + 4];
                    assert_eq!(
                        profile.decode_secret(dit, &changed, &mut out),
                        Err(Error { kind, offset }),
                        "{profile:?} {len}"
                    );
                    assert!(out[..len].iter().all(|&byte| byte == 0), "{profile:?} {len} {kind:?}");
                    assert!(out[len..].iter().all(|&byte| byte == UNWRITTEN), "{profile:?} {len}");
                }
            }
            let mut out = [UNWRITTEN; 8];
            let length = match profile {
                Profile::Url => b"AAAAA".as_slice(),
                Profile::Standard => b"AAAAAA",
            };
            assert_eq!(
                profile.decode_secret(dit, length, &mut out),
                Err(Error {
                    kind: ErrorKind::Length,
                    offset: length.len()
                })
            );
            assert_eq!(
                profile.decode_secret(dit, b"AAAA*AAAAAAA", &mut out),
                Err(Error {
                    kind: ErrorKind::Output,
                    offset: 9
                })
            );
            assert_eq!(out, [UNWRITTEN; 8]);
        }
    });
}
