//! The contracts of the kernels and the cursor. Each kernel gives the classes, the tokens, and the
//! carry of the reference on each block. The cursor gives the tokens of a scan of one byte at a
//! time, and the UTF-8 error of `core::str::from_utf8`.

extern crate std;

use std::{format, fs, path::Path, vec, vec::Vec};

use super::{
    Backend, Kernel,
    block::{Block, Carry, Classes},
    cursor::{Cursor, InvalidUtf8},
};

/// Each kernel that this CPU runs, the fastest first. The last is the reference.
fn kernels() -> Vec<Backend> {
    let kernels: Vec<_> = Kernel::ALL.iter().filter_map(|&kernel| Backend::new(kernel)).collect();
    assert_eq!(kernels.first(), Some(&Backend::chosen()));
    assert_eq!(kernels.last(), Backend::new(Kernel::Swar).as_ref());
    kernels
}

#[expect(
    unsafe_code,
    reason = "a `Backend` exists only on a CPU that has the features of its kernel"
)]
fn classify(kernel: Backend, bytes: &[u8], before: u32, carry: Carry) -> Block {
    // SAFETY: as above.
    unsafe { kernel.0.function()(bytes, before, carry) }
}

/// The tokens that the cursor gives, and its end: `Ok` or the UTF-8 error.
fn cursor(input: &[u8], kernel: Backend) -> (Vec<usize>, Result<(), InvalidUtf8>) {
    let mut cursor = Cursor::new(input, kernel);
    let mut tokens = Vec::new();
    loop {
        match cursor.next() {
            Ok(Some(at)) => tokens.push(at),
            Ok(None) => return (tokens, Ok(())),
            Err(error) => {
                assert_eq!(cursor.next(), Err(error), "each call after the error gives it");
                return (tokens, Err(error));
            }
        }
    }
}

/// The tokens of `input` ([`Block::tokens`]), by a scan of one byte at a time. A `\` escapes the
/// next byte also outside a string, as in the kernels.
fn scan(input: &[u8]) -> Vec<usize> {
    let (mut in_string, mut escape, mut scalar) = (false, false, false);
    let mut tokens = Vec::new();
    for (at, &byte) in input.iter().enumerate() {
        let escaped = escape;
        escape = byte == b'\\' && !escaped;
        let token = if byte == b'"' && !escaped {
            in_string = !in_string;
            scalar = false;
            true
        } else if in_string {
            escape || byte < 0x20
        } else if b"{}[]:,".contains(&byte) {
            scalar = false;
            true
        } else if b" \t\n\r".contains(&byte) {
            scalar = false;
            false
        } else {
            !core::mem::replace(&mut scalar, true)
        };
        if token {
            tokens.push(at);
        }
    }
    tokens
}

/// The cursor of `kernel` gives the tokens of [`scan`] before the first byte that is not UTF-8,
/// then the error at the offset that `core::str::from_utf8` gives.
///
/// The one token at the offset of the error that can come first ([`Cursor`]) is a byte at 0x80 or
/// more outside a string, in the last 3 bytes of a block.
fn check_cursor(input: &[u8], tokens: &[usize], kernel: Backend) {
    let (given, end) = cursor(input, kernel);
    let expected = core::str::from_utf8(input).map(drop).map_err(|error| InvalidUtf8 {
        offset: error.valid_up_to(),
    });
    assert_eq!(end, expected, "{kernel:?}, input {input:x?}");
    let valid = expected.err().map_or(input.len(), |error| error.offset);
    let before = tokens.partition_point(|&at| at < valid);
    let early = tokens.get(before) == Some(&valid) && input[valid] >= 0x80 && valid % 64 >= 61;
    assert!(
        given == tokens[..before] || early && given == tokens[..=before],
        "{kernel:?}, input {input:x?}: the cursor gives {given:?}, the scan {tokens:?}"
    );
}

/// The blocks of `input` as the cursor gives them to a kernel: the bytes from the start of each
/// block to the end of the input, each with the 4 bytes before it.
fn blocks(input: &[u8]) -> impl Iterator<Item = (&[u8], u32)> {
    (0..input.len()).step_by(64).map(|start| {
        let before = input[..start]
            .last_chunk()
            .map_or(0, |before| u32::from_le_bytes(*before));
        (&input[start..], before)
    })
}

/// Each kernel gives the block of the reference, with its carry, on each block of `input`.
fn check_blocks(input: &[u8], kernels: &[Backend]) {
    let mut carries = vec![Carry::default(); kernels.len()];
    for (index, (bytes, before)) in blocks(input).enumerate() {
        let reference = classify(kernels[kernels.len() - 1], bytes, before, carries[kernels.len() - 1]);
        for (&kernel, carry) in kernels.iter().zip(&mut carries) {
            let block = classify(kernel, bytes, before, *carry);
            assert_eq!(block, reference, "{kernel:?}, block {index} of {input:x?}");
            *carry = block.carry;
        }
    }
}

/// Each kernel equals the reference on each block of each file, from each of the first 64 offsets.
/// Thus each byte comes in each lane, and the last block has each length from 1 to 64.
fn check_files(files: &[&Path]) {
    let kernels = kernels();
    for file in files {
        let text = fs::read(file).unwrap_or_else(|error| {
            panic!(
                "{}: {error}; benches/data/README.md gives the command that fetches it",
                file.display()
            )
        });
        for input in (0..64).filter_map(|offset| text.get(offset..)) {
            check_blocks(input, &kernels);
            let tokens = scan(input);
            for &kernel in &kernels {
                check_cursor(input, &tokens, kernel);
            }
        }
    }
}

/// Each corpus file of the repository.
#[test]
fn each_kernel_equals_the_reference_on_the_generated_corpora() {
    let mut files = Vec::new();
    for class in fs::read_dir("benches/data/json").unwrap() {
        for file in fs::read_dir(class.unwrap().path()).unwrap() {
            files.push(file.unwrap().path());
        }
    }
    assert!(!files.is_empty(), "benches/data/json has no file");
    check_files(&files.iter().map(|file| file.as_path()).collect::<Vec<_>>());
}

/// The three corpora that benches/data/README.md fetches. A missing file fails the test.
#[test]
fn each_kernel_equals_the_reference_on_the_fetched_corpora() {
    let files = ["twitter", "citm_catalog", "canada"].map(|name| format!("benches/data/fetched/{name}.json"));
    check_files(&files.each_ref().map(Path::new));
}

/// Each string of 6 bytes of quotes, `\`, a scalar byte, whitespace, a structural byte, and a byte
/// under 0x20, across the end of a block at each place. Thus each run of `\` before a quote, also an
/// escaped `\`, and each carry.
///
/// No corpus has an escaped `\` before a quote (`\\"`). Thus this test, not the corpus tests,
/// fails a quote mask that takes that quote as escaped.
#[test]
fn escapes_quotes_and_scalars_across_blocks_give_the_tokens_of_the_scan() {
    const BYTES: [u8; 6] = [b'"', b'\\', b'a', b' ', b',', 0x01];
    let kernels = kernels();
    let mut input = Vec::new();
    for index in 0..BYTES.len().pow(6) {
        let mut rest = index;
        let word: [u8; 6] = core::array::from_fn(|_| {
            let byte = BYTES[rest % BYTES.len()];
            rest /= BYTES.len();
            byte
        });
        for offset in 58..=64 {
            input.clear();
            input.resize(offset, b' ');
            input.extend_from_slice(&word);
            input.extend_from_slice(b"\"a");
            check_blocks(&input, &kernels);
            let tokens = scan(&input);
            for &kernel in &kernels {
                check_cursor(&input, &tokens, kernel);
            }
        }
    }
}

/// Each sequence of 4 bytes of the UTF-8 classes of the check, in a string across the end of a
/// block at each place, at the end of the input and before more bytes.
#[test]
fn each_utf8_sequence_across_blocks_gives_the_error_of_std() {
    // ASCII, the continuation bytes, and the lead bytes with each class of low nibble.
    const BYTES: [u8; 18] = [
        b'a', 0x80, 0x90, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xe1, 0xed, 0xee, 0xf0, 0xf1, 0xf4, 0xf5, 0xff,
    ];
    let kernels = kernels();
    let mut input = Vec::new();
    for index in 0..BYTES.len().pow(4) {
        let mut rest = index;
        let word: [u8; 4] = core::array::from_fn(|_| {
            let byte = BYTES[rest % BYTES.len()];
            rest /= BYTES.len();
            byte
        });
        for offset in 60..=64 {
            for after in [0, 3] {
                input.clear();
                input.push(b'"');
                input.resize(offset, b'a');
                input.extend_from_slice(&word);
                input.resize(offset + 4 + after, b'a');
                check_blocks(&input, &kernels);
                for &kernel in &kernels {
                    check_cursor(&input, &[0], kernel);
                }
            }
        }
    }
}

/// Each invalid sequence at each place in a run of structural bytes, so that tokens come before
/// and after the first invalid byte in its block and in the next block. The cursor gives each token
/// before that byte, then its error, so the first error by offset comes first.
#[test]
fn the_cursor_gives_the_tokens_before_the_first_invalid_byte_then_its_error() {
    // A continuation byte, a byte that no UTF-8 has, an overlong form, a sequence that ends too
    // soon, and a surrogate.
    const SEQUENCES: [&[u8]; 5] = [&[0x80], &[0xff], &[0xc0, 0x80], &[0xe2, 0x82], &[0xed, 0xa0, 0x80]];
    let kernels = kernels();
    for sequence in SEQUENCES {
        for at in 1..128 {
            let mut input = vec![b','; 132];
            input[0] = b'[';
            input[at..at + sequence.len()].copy_from_slice(sequence);
            check_blocks(&input, &kernels);
            let tokens = scan(&input);
            for &kernel in &kernels {
                check_cursor(&input, &tokens, kernel);
            }
        }
    }
}

/// The one token that can come at the offset of its UTF-8 error: the first byte of a scalar at
/// 0x80 or more in the last 3 bytes of a block, when the check finds the error in the next block.
/// At one offset the UTF-8 error wins over an error of the grammar, so the reader checks UTF-8 there.
#[test]
fn a_scalar_in_the_last_bytes_of_a_block_comes_before_its_utf8_error() {
    // A sequence that needs one, two, and three more bytes, at the end of the first block.
    for sequence in [&[0xc3][..], &[0xe2, 0x82], &[0xf0, 0x9f, 0x98]] {
        let at = 64 - sequence.len();
        let mut input = vec![b','; 70];
        input[0] = b'[';
        input[at..64].copy_from_slice(sequence);
        for kernel in kernels() {
            let expected: Vec<usize> = (0..=at).collect();
            assert_eq!(
                cursor(&input, kernel),
                (expected, Err(InvalidUtf8 { offset: at })),
                "{kernel:?}, {sequence:x?}"
            );
        }
    }
}

/// Each length of the last block, and an input of whole blocks. The kernel reads the bytes of a
/// short last block with spaces after them, and it reads no byte after the input.
#[test]
fn the_last_block_of_each_length_gives_the_tokens_of_the_scan() {
    // Structural bytes, whitespace, strings with an escaped quote and an escaped `\`, a number, and
    // literals.
    const TEXT: &[u8] = b"{\"a\\\"b\": [1, -2.5e3, \"\\\\\", true],\n\"c\": {\"d\": null}} ";
    let text: Vec<u8> = TEXT.iter().copied().cycle().take(130).collect();
    let kernels = kernels();
    for len in 0..=text.len() {
        let input = &text[..len];
        check_blocks(input, &kernels);
        let tokens = scan(input);
        for &kernel in &kernels {
            check_cursor(input, &tokens, kernel);
        }
    }
}

/// Each byte value in each lane of a block classifies as RFC 8259 defines it.
#[test]
fn each_byte_in_each_lane_has_its_class() {
    let kernels = kernels();
    for offset in 0..64 {
        let input: Vec<u8> = (0..=u8::MAX).map(|byte| byte.wrapping_add(offset)).collect();
        for &kernel in &kernels {
            let mut carry = Carry::default();
            for (bytes, before) in blocks(&input) {
                let mut expected = Classes::default();
                for (lane, &byte) in bytes[..64].iter().enumerate() {
                    let bit = |class: bool| u64::from(class) << lane;
                    expected.structural |= bit(b"{}[]:,".contains(&byte));
                    expected.whitespace |= bit(b" \t\n\r".contains(&byte));
                    expected.quote |= bit(byte == b'"');
                    expected.backslash |= bit(byte == b'\\');
                    expected.control |= bit(byte < 0x20);
                }
                let block = classify(kernel, bytes, before, carry);
                carry = block.carry;
                let classes = block.classes;
                assert_eq!(
                    Classes {
                        utf8_error: false,
                        ..classes
                    },
                    expected,
                    "{kernel:?}, offset {offset}"
                );
            }
            check_cursor(&input, &scan(&input), kernel);
        }
    }
}

/// The selection gives a kernel only to a CPU that has each feature of its `#[target_feature]`
/// functions. A kernel on a CPU without one of its features is undefined behavior. WebAssembly
/// selects at compile time, so it has no case.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn the_selection_gives_a_kernel_only_with_each_of_its_features() {
    use crate::cpu::Caps;
    let fastest = |features: &[Caps]| {
        Kernel::ALL[Kernel::fastest(features.iter().fold(Caps::NONE, |all, &feature| all.union(feature)))]
    };
    let check = |features: &[Caps], kernel: Kernel, without: &[Kernel]| {
        assert_eq!(fastest(features), kernel);
        for (missing, &expected) in without.iter().enumerate() {
            let mut fewer = features.to_vec();
            fewer.remove(missing);
            assert_eq!(fastest(&fewer), expected, "{kernel:?} without its feature {missing}");
        }
    };
    #[cfg(target_arch = "x86_64")]
    {
        use crate::cpu::x86::{AVX2, AVX512BW, AVX512F, AVX512VBMI2, AVX512VL, BMI1, BMI2, PCLMULQDQ};
        let (avx2, swar, avx512) = (Kernel::Avx2, Kernel::Swar, Kernel::Avx512);
        check(&[AVX2, PCLMULQDQ, BMI1, BMI2], avx2, &[swar; 4]);
        check(
            &[AVX512F, AVX512BW, AVX512VL, AVX512VBMI2, AVX2, PCLMULQDQ, BMI1, BMI2],
            avx512,
            &[avx2, avx2, avx2, avx2, swar, swar, swar, swar],
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        use crate::cpu::aarch64::{AES, NEON};
        check(&[NEON, AES], Kernel::NeonPmull, &[Kernel::Swar, Kernel::Neon]);
    }
    assert_eq!(fastest(&[]), Kernel::Swar);
}
