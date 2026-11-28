//! Benchmark public, secret, and explicit kernels on the same 64-byte-aligned buffers.
//! Check encoded text against data-encoding and decoded output against the original bytes.

#[path = "common/aligned.rs"]
mod aligned;
mod common;

use core::hint::black_box;

use aligned::Aligned;
use base64::Engine as _;
use base64ct::Encoding as _;
use common::Group;

/// Runs `scope` in a scope of `rscodec::with_dit`. Each bench host can tell whether its CPU has
/// DIT.
fn secret<T>(scope: impl FnOnce(&rscodec::Dit) -> T) -> T {
    rscodec::with_dit(scope).expect("the bench host reports DIT")
}
use criterion::{Criterion, Throughput};
use rscodec::{base64::Profile, hex};

/// The raw sizes in bytes.
const SIZES: [usize; 6] = [16, 32, 64, 256, 4096, 65536];

/// The largest raw size that uses base64url without padding. A larger size uses padded standard
/// base64, the form of bulk data.
const URL_SIZE_MAX: usize = 4096;

/// One raw input, its text in one alphabet, and the output of each row.
struct Case {
    label: String,
    url: bool,
    raw: Aligned,
    text: Aligned,
    output: Aligned,
}

impl Case {
    fn new(label: String, url: bool, raw: &[u8], text: &[u8]) -> Self {
        // Each decoder may need room for the bytes of a full last quantum.
        let output = Aligned::new(&vec![0; text.len().max(raw.len() + 3)]);
        Self {
            label,
            url,
            raw: Aligned::new(raw),
            text: Aligned::new(text),
            output,
        }
    }
}

/// Deterministic pseudo-random bytes (xorshift64*), so that each run reads the same inputs.
fn random_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ u64::try_from(len).expect("a length fits u64");
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d).to_be_bytes()[0]
        })
        .collect()
}

fn base64_cases() -> Vec<Case> {
    SIZES
        .iter()
        .map(|&size| {
            let raw = random_bytes(size);
            let url = size <= URL_SIZE_MAX;
            let (label, oracle) = if url {
                (format!("url-{size}"), &data_encoding::BASE64URL_NOPAD)
            } else {
                (format!("std-{size}"), &data_encoding::BASE64)
            };
            Case::new(label, url, &raw, oracle.encode(&raw).as_bytes())
        })
        .collect()
}

fn hex_cases() -> Vec<Case> {
    SIZES
        .iter()
        .map(|&size| {
            let raw = random_bytes(size);
            Case::new(
                size.to_string(),
                false,
                &raw,
                data_encoding::HEXLOWER.encode(&raw).as_bytes(),
            )
        })
        .collect()
}

/// A row that encodes `case.raw` with `encode`, which returns the length of the text.
fn encode_row(group: &mut Group<'_>, case: &mut Case, codec: &str, mut encode: impl FnMut(&[u8], &mut [u8]) -> usize) {
    let Case {
        label,
        raw,
        text,
        output,
        ..
    } = case;
    let (raw, text) = (&**raw, &**text);
    let output = &mut output[..text.len()];
    let written = encode(raw, output);
    assert_eq!(&output[..written], text, "{codec} encodes {label}");
    timed_row(group, codec, label, raw, output, encode);
}

/// A row that decodes `case.text` with `decode`, which returns the length of the bytes, or `None`
/// for an error.
fn decode_row(
    group: &mut Group<'_>,
    case: &mut Case,
    codec: &str,
    output_len: usize,
    mut decode: impl FnMut(&[u8], &mut [u8]) -> Option<usize>,
) {
    let Case {
        label,
        raw,
        text,
        output,
        ..
    } = case;
    let (raw, text) = (&**raw, &**text);
    let output = &mut output[..output_len];
    let written = decode(text, output).expect("valid text decodes");
    assert_eq!(&output[..written], raw, "{codec} decodes {label}");
    timed_row(group, codec, label, text, output, decode);
}

/// Keep the input opaque and the output observable. Capturing the output slice keeps its address
/// in a register. Passing it through the harness state added a load to each short call on Graviton4.
#[expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports the place of the buffers of each row on standard output, beside Criterion"
)]
fn timed_row<T>(
    group: &mut Group<'_>,
    codec: &str,
    label: &str,
    input: &[u8],
    output: &mut [u8],
    mut operation: impl FnMut(&[u8], &mut [u8]) -> T,
) {
    // The offsets of the input and the output in their 4 KiB pages. A load or a store across a
    // page or a cache line costs more, so the offsets can explain a slow row.
    let page = |bytes: &[u8]| bytes.as_ptr().addr() % 4096;
    let (input_page, output_page) = (page(input), page(output));
    group.row(
        codec,
        label,
        (),
        |()| {
            let result = operation(black_box(input), output);
            black_box(output.as_ptr());
            result
        },
        |_, ()| println!("place {codec}/{label} input={input_page} output={output_page}"),
    );
}

/// The name of the row of `kernel`, for example `rscodec-avx2`.
fn kernel_row(kernel: impl core::fmt::Debug) -> String {
    format!("rscodec-{kernel:?}").to_lowercase()
}

/// The engines of `base64`: `GeneralPurpose`, the row `base64-engine-gp`, and `Simd`, the row
/// `base64-engine-simd`.
fn base64_engines(url: bool) -> (base64::engine::GeneralPurpose, base64::engine::Simd) {
    use base64::engine::{Simd, general_purpose};
    if url {
        (
            general_purpose::URL_SAFE_NO_PAD,
            Simd::url_safe(general_purpose::NO_PAD),
        )
    } else {
        (general_purpose::STANDARD, Simd::standard(general_purpose::PAD))
    }
}

fn base64_encode(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "base64-encode");
    for case in &mut base64_cases() {
        group.throughput(Throughput::Bytes(case.raw.len().try_into().expect("a length fits u64")));
        let g = &mut group;
        let profile = if case.url { Profile::Url } else { Profile::Standard };
        encode_row(g, case, "rscodec", |raw, out| {
            profile.encode(raw, out).expect("room").len()
        });
        secret(|dit| {
            encode_row(g, case, "rscodec-secret", |raw, out| {
                profile.encode_secret(dit, raw, out).expect("room").len()
            });
        });
        encode_row(g, case, "rscodec-secret-cold", |raw, out| {
            secret(|dit| profile.encode_secret(dit, raw, out).expect("room").len())
        });
        for &kernel in rscodec::base64::Kernel::ALL {
            if let Some(backend) = rscodec::base64::Backend::new(kernel) {
                encode_row(g, case, &kernel_row(kernel), |raw, out| {
                    profile.encode_with(backend, raw, out).expect("room").len()
                });
            }
        }
        let turbo = if case.url {
            base64_turbo::URL_SAFE_NO_PAD
        } else {
            base64_turbo::STANDARD
        };
        encode_row(g, case, "base64-turbo", |raw, out| {
            turbo.encode_slice(raw, out).expect("room")
        });
        #[cfg(target_arch = "x86_64")]
        encode_row(g, case, "base64-turbo-avx2", |raw, out| {
            turbo.encode_avx2(raw, out).expect("the host has AVX2");
            out.len()
        });
        let (general, simd) = base64_engines(case.url);
        encode_row(g, case, "base64-engine-gp", |raw, out| {
            general.encode_slice(raw, out).expect("room")
        });
        encode_row(g, case, "base64-engine-simd", |raw, out| {
            simd.encode_slice(raw, out).expect("room")
        });
        let simd_crate = if case.url {
            base64_simd::URL_SAFE_NO_PAD
        } else {
            base64_simd::STANDARD
        };
        encode_row(g, case, "base64-simd", |raw, out| {
            simd_crate.encode(raw, base64_simd::Out::from_slice(out)).len()
        });
        if case.url {
            let ng = base64_ng::STRICT_URL_SAFE_UNPADDED;
            encode_row(g, case, "base64-ng", |raw, out| ng.encode_into(raw, out).expect("room"));
        } else {
            let ng = base64_ng::STRICT_STANDARD_PADDED;
            encode_row(g, case, "base64-ng", |raw, out| ng.encode_into(raw, out).expect("room"));
        }
        let encodify = if case.url {
            encodify::base64::URL_SAFE_NO_PAD
        } else {
            encodify::base64::STANDARD
        };
        encode_row(g, case, "encodify", |raw, out| {
            encodify.encode_slice(raw, out).expect("room")
        });
        let data = if case.url {
            &data_encoding::BASE64URL_NOPAD
        } else {
            &data_encoding::BASE64
        };
        encode_row(g, case, "data-encoding", |raw, out| {
            data.encode_mut(raw, out);
            out.len()
        });
        if case.url {
            encode_row(g, case, "base64ct", |raw, out| {
                base64ct::Base64UrlUnpadded::encode(raw, out).expect("room").len()
            });
        } else {
            encode_row(g, case, "base64ct", |raw, out| {
                base64ct::Base64::encode(raw, out).expect("room").len()
            });
        }
        #[cfg(not(any(target_env = "msvc", target_family = "wasm")))]
        {
            let options = simdutf_base64::options(case.url);
            encode_row(g, case, "simdutf", |raw, out| simdutf_base64::encode(raw, out, options));
        }
    }
}

fn base64_decode(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "base64-decode");
    for case in &mut base64_cases() {
        group.throughput(Throughput::Bytes(case.raw.len().try_into().expect("a length fits u64")));
        let g = &mut group;
        let room = case.raw.len() + 3;
        let profile = if case.url { Profile::Url } else { Profile::Standard };
        decode_row(g, case, "rscodec", room, |text, out| {
            profile.decode(text, out).ok().map(|bytes| bytes.len())
        });
        secret(|dit| {
            decode_row(g, case, "rscodec-secret", room, |text, out| {
                profile.decode_secret(dit, text, out).ok().map(|bytes| bytes.len())
            });
        });
        decode_row(g, case, "rscodec-secret-cold", room, |text, out| {
            secret(|dit| profile.decode_secret(dit, text, out).ok().map(|bytes| bytes.len()))
        });
        for &kernel in rscodec::base64::Kernel::ALL {
            if let Some(backend) = rscodec::base64::Backend::new(kernel) {
                decode_row(g, case, &kernel_row(kernel), room, |text, out| {
                    profile.decode_with(backend, text, out).ok().map(|bytes| bytes.len())
                });
            }
        }
        let turbo = if case.url {
            base64_turbo::URL_SAFE_NO_PAD
        } else {
            base64_turbo::STANDARD
        };
        decode_row(g, case, "base64-turbo", room, |text, out| {
            turbo.decode_slice(text, out).ok()
        });
        #[cfg(target_arch = "x86_64")]
        decode_row(g, case, "base64-turbo-avx2", room, |text, out| {
            turbo.decode_avx2(text, out).expect("the host has AVX2").ok()
        });
        let (general, simd) = base64_engines(case.url);
        decode_row(g, case, "base64-engine-gp", room, |text, out| {
            general.decode_slice(text, out).ok()
        });
        decode_row(g, case, "base64-engine-simd", room, |text, out| {
            simd.decode_slice(text, out).ok()
        });
        let simd_crate = if case.url {
            base64_simd::URL_SAFE_NO_PAD
        } else {
            base64_simd::STANDARD
        };
        decode_row(g, case, "base64-simd", room, |text, out| {
            simd_crate
                .decode(text, base64_simd::Out::from_slice(out))
                .ok()
                .map(|bytes| bytes.len())
        });
        if case.url {
            let ng = base64_ng::STRICT_URL_SAFE_UNPADDED;
            decode_row(g, case, "base64-ng", room, |text, out| ng.decode_into(text, out).ok());
        } else {
            let ng = base64_ng::STRICT_STANDARD_PADDED;
            decode_row(g, case, "base64-ng", room, |text, out| ng.decode_into(text, out).ok());
        }
        let encodify = if case.url {
            encodify::base64::URL_SAFE_NO_PAD
        } else {
            encodify::base64::STANDARD
        };
        decode_row(g, case, "encodify", room, |text, out| {
            encodify.decode_slice(text, out).ok()
        });
        let data = if case.url {
            &data_encoding::BASE64URL_NOPAD
        } else {
            &data_encoding::BASE64
        };
        decode_row(g, case, "data-encoding", room, |text, out| {
            let len = data.decode_len(text.len()).ok()?;
            data.decode_mut(text, out.get_mut(..len)?).ok()
        });
        if case.url {
            decode_row(g, case, "base64ct", room, |text, out| {
                base64ct::Base64UrlUnpadded::decode(text, out).ok().map(<[u8]>::len)
            });
        } else {
            decode_row(g, case, "base64ct", room, |text, out| {
                base64ct::Base64::decode(text, out).ok().map(<[u8]>::len)
            });
        }
        #[cfg(not(any(target_env = "msvc", target_family = "wasm")))]
        {
            let options = simdutf_base64::options(case.url);
            decode_row(g, case, "simdutf", room, |text, out| {
                simdutf_base64::decode(text, out, options)
            });
        }
    }
}

fn hex_encode(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "hex-encode");
    for case in &mut hex_cases() {
        group.throughput(Throughput::Bytes(case.raw.len().try_into().expect("a length fits u64")));
        let g = &mut group;
        encode_row(g, case, "rscodec", |raw, out| {
            hex::encode(raw, out).expect("room").len()
        });
        secret(|dit| {
            encode_row(g, case, "rscodec-secret", |raw, out| {
                hex::encode_secret(dit, raw, out).expect("room").len()
            });
        });
        encode_row(g, case, "rscodec-secret-cold", |raw, out| {
            secret(|dit| hex::encode_secret(dit, raw, out).expect("room").len())
        });
        for &kernel in hex::Kernel::ALL {
            if let Some(backend) = hex::Backend::new(kernel) {
                encode_row(g, case, &kernel_row(kernel), |raw, out| {
                    hex::encode_with(backend, raw, out).expect("room").len()
                });
            }
        }
        encode_row(g, case, "const-hex", |raw, out| {
            const_hex::encode_to_slice(raw, out).expect("room");
            out.len()
        });
        encode_row(g, case, "faster-hex", |raw, out| {
            faster_hex::hex_encode(raw, out).expect("room").len()
        });
        encode_row(g, case, "base16ct", |raw, out| {
            base16ct::lower::encode(raw, out).expect("room").len()
        });
    }
}

fn hex_decode(criterion: &mut Criterion) {
    let mut group = Group::new(criterion, "hex-decode");
    for case in &mut hex_cases() {
        group.throughput(Throughput::Bytes(case.raw.len().try_into().expect("a length fits u64")));
        let g = &mut group;
        let room = case.raw.len();
        decode_row(g, case, "rscodec", room, |text, out| {
            hex::decode(text, out).ok().map(|bytes| bytes.len())
        });
        secret(|dit| {
            decode_row(g, case, "rscodec-secret", room, |text, out| {
                hex::decode_secret(dit, text, out).ok().map(|bytes| bytes.len())
            });
        });
        decode_row(g, case, "rscodec-secret-cold", room, |text, out| {
            secret(|dit| hex::decode_secret(dit, text, out).ok().map(|bytes| bytes.len()))
        });
        for &kernel in hex::Kernel::ALL {
            if let Some(backend) = hex::Backend::new(kernel) {
                decode_row(g, case, &kernel_row(kernel), room, |text, out| {
                    hex::decode_with(backend, text, out).ok().map(|bytes| bytes.len())
                });
            }
        }
        decode_row(g, case, "const-hex", room, |text, out| {
            const_hex::decode_to_slice(text, out).ok().map(|()| out.len())
        });
        decode_row(g, case, "faster-hex", room, |text, out| {
            faster_hex::hex_decode(text, out).ok().map(|bytes| bytes.len())
        });
        decode_row(g, case, "base16ct", room, |text, out| {
            base16ct::lower::decode(text, out).ok().map(<[u8]>::len)
        });
    }
}

/// The base64 functions of simdutf, which take raw pointers, behind slice bounds.
#[cfg(not(any(target_env = "msvc", target_family = "wasm")))]
#[expect(
    unsafe_code,
    reason = "simdutf has only raw-pointer functions, and each call passes the bounds of live slices"
)]
mod simdutf_base64 {
    use simdutf::{Base64Options, ErrorCode, LastChunkHandlingOptions};

    pub(crate) fn options(url: bool) -> Base64Options {
        if url {
            Base64Options::Url
        } else {
            Base64Options::Default
        }
    }

    pub(crate) fn encode(input: &[u8], output: &mut [u8], options: Base64Options) -> usize {
        // The length of the text, without a second call into the library.
        let text_len = if options == Base64Options::Url {
            input.len().strict_mul(4).div_ceil(3)
        } else {
            input.len().div_ceil(3).strict_mul(4)
        };
        assert!(output.len() >= text_len);
        // SAFETY: `input` is valid for its length, and the assert proves that `output` holds the
        // text. The two slices cannot overlap.
        unsafe { simdutf::binary_to_base64(input.as_ptr(), input.len(), output.as_mut_ptr(), options) }
    }

    /// Strict decode: the last chunk of padded text must have its padding.
    pub(crate) fn decode(input: &[u8], output: &mut [u8], options: Base64Options) -> Option<usize> {
        let last_chunk = if options == Base64Options::Url {
            LastChunkHandlingOptions::Loose
        } else {
            LastChunkHandlingOptions::Strict
        };
        let mut written = output.len();
        // SAFETY: `input` is valid for its length, `output` for `written` bytes, and the two slices
        // cannot overlap. simdutf writes at most `written` bytes and stores the count there.
        let result = unsafe {
            simdutf::base64_to_binary_safe(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                &raw mut written,
                options,
                last_chunk,
                false,
            )
        };
        (result.error == ErrorCode::Success).then_some(written)
    }
}

fn main() {
    common::run(&[base64_encode, base64_decode, hex_encode, hex_decode]);
}
