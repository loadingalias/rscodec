//! The contracts of `rscodec::hex`: the RFC 4648 vectors, the strict lowercase profile, the error
//! and its offset, the same result on each backend and in each timing class, the wipe of the secret
//! class, and the DIT state of the scope of the secret functions.

mod common;

use common::{Rng, proof, secret};
use rscodec::hex::{self, Backend, Error, ErrorKind, Kernel};

/// A call of the codec: on a backend, or a function of a timing class, with no backend argument.
#[derive(Clone, Copy, Debug)]
enum Call {
    With(Backend),
    Public,
    Secret,
}

/// Each backend that this CPU runs, then the functions of each timing class. A build that makes no
/// proof has no call of the secret class.
fn calls() -> Vec<Call> {
    let backends = Kernel::ALL.iter().filter_map(|&kernel| Backend::new(kernel));
    backends
        .map(Call::With)
        .chain([Call::Public])
        .chain(proof().then_some(Call::Secret))
        .collect()
}

/// Encodes `bytes` into `out` with `call`, and gives the length of the text.
fn encode_into(call: Call, bytes: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => hex::encode_with(backend, bytes, out),
        Call::Public => hex::encode(bytes, out),
        Call::Secret => secret(|dit| hex::encode_secret(dit, bytes, out)),
    }
    .map(|text| text.len())
}

/// Decodes `text` into `out` with `call`, and gives the length of the bytes.
fn decode_into(call: Call, text: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    match call {
        Call::With(backend) => hex::decode_with(backend, text, out),
        Call::Public => hex::decode(text, out),
        Call::Secret => secret(|dit| hex::decode_secret(dit, text, out)),
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
            let upper = text.find('F').map(|offset| Error::new(ErrorKind::Symbol, offset));
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
                        assert_eq!(decoded, Err(Error::new(ErrorKind::Symbol, place)), "{call:?}");
                    }
                }
            }
        }
    }
}

/// Without `std`, the crate detects no CPU, so a backend exists exactly for each kernel whose CPU
/// features the build turns on. A backend of another kernel runs instructions that the CPU of the
/// user can lack. `cargo xtask test` runs this build after the build with `std`.
#[cfg(not(feature = "std"))]
#[test]
fn without_std_a_backend_exists_exactly_for_each_kernel_of_the_build() {
    let avx512bw = cfg!(all(target_feature = "avx512f", target_feature = "avx512bw"));
    // The name of each kernel, the fastest first, and whether the build has its CPU features.
    let kernels = [
        ("Avx512Bw", avx512bw),
        ("Avx2", cfg!(target_feature = "avx2")),
        ("Ssse3", cfg!(target_feature = "ssse3")),
        ("Neon", cfg!(target_feature = "neon")),
        ("Simd128", cfg!(target_feature = "simd128")),
        ("Portable", true),
    ];
    let built: Vec<_> = kernels
        .into_iter()
        .filter_map(|(name, built)| built.then_some(name))
        .collect();
    let backends: Vec<_> = Kernel::ALL
        .iter()
        .filter(|&&kernel| Backend::new(kernel).is_some())
        .map(|kernel| format!("{kernel:?}"))
        .collect();
    assert_eq!(backends, built);
}

/// Each backend and each timing class gives the result and the error of the portable backend: on
/// random inputs of each length to 96 bytes, past the short path of 64 bytes, on each byte at each
/// place of their texts, and on random inputs to 1 MiB.
#[test]
fn each_backend_equals_the_portable_backend() {
    let portable = Call::With(Backend::new(Kernel::Portable).unwrap());
    let calls = calls();
    let mut rng = Rng::new(0x0068_6578);
    for len in (0..=96).chain([1000, 4096, 65_535, 65_536, 1 << 20]) {
        let bytes = rng.bytes(len);
        let text = encode(portable, &bytes);
        let mut texts = vec![text.clone(), [&text[..], b"0"].concat()];
        for place in 0..text.len() {
            // Each byte value at a seventh of the places of each length, so that over the lengths
            // each place meets each byte. A random byte at the other places.
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
            assert_eq!(encode(call, &bytes), text, "{call:?} {len}");
            for changed in &texts {
                assert_eq!(decode(call, changed), decode(portable, changed), "{call:?} {len}");
            }
        }
    }
}

/// Each short vector store fits the caller's unaligned slice, including the last overlapping store.
#[test]
fn short_hex_stores_fit_an_unaligned_output_slice() {
    #[repr(align(16))]
    struct Buffer([u8; 17]);
    let bytes = b"0123456789abcde";
    let text = b"303132333435363738396162636465";
    for call in calls() {
        for len in 8..=bytes.len() {
            let mut output = Buffer([0xff; 17]);
            assert_eq!(
                decode_into(call, &text[..2 * len], &mut output.0[1..=len]),
                Ok(len),
                "{call:?}"
            );
            assert_eq!(&output.0[1..=len], &bytes[..len]);
            assert_eq!(output.0[0], 0xff);
            assert!(output.0[len + 1..].iter().all(|&byte| byte == 0xff));
        }
    }
}

/// A short output slice refuses with the length of the result, and the decoder checks the length
/// first.
#[test]
fn a_short_output_refuses_with_the_length() {
    let error = |kind, offset| Err(Error::new(kind, offset));
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
    // A build that makes no proof has no secret call.
    if !proof() {
        return;
    }
    secret(|dit| {
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
                    hex::decode_secret(dit, &changed, &mut out),
                    Err(Error::new(ErrorKind::Symbol, place)),
                    "{len}"
                );
                assert!(out[..len].iter().all(|&byte| byte == 0), "{len} {place}");
                assert!(out[len..].iter().all(|&byte| byte == UNWRITTEN), "{len}");
            }
        }
        let mut out = [UNWRITTEN; 8];
        let error = |kind, offset| Err(Error::new(kind, offset));
        assert_eq!(hex::decode_secret(dit, b"000", &mut out), error(ErrorKind::Length, 3));
        assert_eq!(
            hex::decode_secret(dit, &[b'g'; 18], &mut out),
            error(ErrorKind::Output, 9)
        );
        assert_eq!(out, [UNWRITTEN; 8]);
    });
}

/// `with_dit` gives the proof and runs its scope exactly in a build that knows whether the CPU has
/// DIT: each build for another architecture, and on aarch64 a build with the `dit` target feature
/// or with the detection of `std` (Linux, Android, and the Apple systems). Another build gets
/// `DitUnknown`, and its scope does not run.
#[test]
fn with_dit_gives_the_proof_exactly_where_the_build_knows_dit() {
    let known = cfg!(any(
        not(target_arch = "aarch64"),
        target_feature = "dit",
        all(
            feature = "std",
            any(target_os = "linux", target_os = "android", target_vendor = "apple")
        )
    ));
    let mut ran = false;
    assert_eq!(rscodec::with_dit(|_| ran = true).is_ok(), known);
    assert_eq!(ran, known);
}

/// On aarch64 with DIT, the scope of `with_dit` holds `PSTATE.DIT` set for each secret call in it,
/// and after the scope, also after a panic in it, `PSTATE.DIT` is as it was before: set or clear.
#[cfg(target_arch = "aarch64")]
#[test]
fn the_dit_scope_holds_pstate_dit_and_restores_it() {
    // A build that makes no proof has no scope.
    if !proof() {
        return;
    }
    // A CPU without DIT has no state to keep.
    let Some(before) = dit::get() else { return };
    for held in [false, true] {
        dit::set(held);
        secret(|dit| {
            let mut out = [0; 64];
            assert_eq!(dit::get(), Some(true));
            assert_eq!(
                hex::encode_secret(dit, &[0xa5; 32], &mut out).map(|text| text.len()),
                Ok(64)
            );
            assert_eq!(
                hex::decode_secret(dit, &[b'g'; 64], &mut out).map(|bytes| bytes.len()),
                Err(Error::new(ErrorKind::Symbol, 0))
            );
            assert_eq!(dit::get(), Some(true));
        });
        assert_eq!(dit::get(), Some(held));
        assert!(std::panic::catch_unwind(|| secret(|_| panic!("a panic in the scope"))).is_err());
        assert_eq!(dit::get(), Some(held));
    }
    dit::set(before);
}

/// `PSTATE.DIT` of this thread: the system register `S3_3_C4_C2_5`, whose bit 24 is DIT.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "the asm reads or writes only PSTATE.DIT, and only on a CPU that has DIT"
)]
pub(crate) mod dit {
    use core::arch::asm;

    /// The bit of DIT in its register.
    const ON: u64 = 1 << 24;

    /// Whether DIT is set, or `None` if the CPU has no DIT.
    pub(crate) fn get() -> Option<bool> {
        if !std::arch::is_aarch64_feature_detected!("dit") {
            return None;
        }
        let value: u64;
        // SAFETY: the CPU has DIT, so the instruction is defined. It reads only DIT.
        unsafe { asm!("mrs {value}, S3_3_C4_C2_5", value = out(reg) value, options(nomem, nostack, preserves_flags)) };
        Some(value & ON != 0)
    }

    /// Sets DIT to `on`.
    pub(crate) fn set(on: bool) {
        assert!(std::arch::is_aarch64_feature_detected!("dit"));
        let value = if on { ON } else { 0 };
        // SAFETY: the assertion proves that the CPU has DIT, so the instruction is defined. It
        // writes only DIT, which changes no value.
        unsafe { asm!("msr S3_3_C4_C2_5, {value}", value = in(reg) value, options(nomem, nostack, preserves_flags)) };
    }
}
