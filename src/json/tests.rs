extern crate std;

use std::{format, fs, path::Path, vec, vec::Vec};

use super::{
    Backend, Kernel,
    block::{Block, Carry},
    cursor::{Cursor, InvalidUtf8},
};

pub(super) fn kernels() -> Vec<Backend> {
    let kernels: Vec<_> = Kernel::ALL
        .iter()
        .filter(|kernel| crate::cpu::caps().has(kernel.needs().0))
        .map(|&kernel| Backend(kernel))
        .collect();
    assert_eq!(kernels.first(), Some(&Backend::chosen()));
    assert_eq!(kernels.last(), Some(&Backend(Kernel::Swar)));
    kernels
}

#[expect(
    unsafe_code,
    reason = "a `Backend` exists only on a CPU that has the features of its kernel"
)]
fn classify(kernel: Backend, input: &[u8], start: usize, carry: Carry) -> Block {
    // SAFETY: as above.
    unsafe { kernel.0.function()(input, start, carry) }
}

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

fn check_blocks(input: &[u8], kernels: &[Backend]) {
    let mut carries = vec![Carry::default(); kernels.len()];
    for (index, start) in (0..input.len()).step_by(64).enumerate() {
        let reference = classify(kernels[kernels.len() - 1], input, start, carries[kernels.len() - 1]);
        for (&kernel, carry) in kernels.iter().zip(&mut carries) {
            let block = classify(kernel, input, start, *carry);
            assert_eq!(block, reference, "{kernel:?}, block {index} of {input:x?}");
            *carry = block.carry;
        }
    }
}

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

#[test]
fn each_kernel_equals_the_reference_on_the_fuzz_corpus() {
    let files: Vec<_> = fs::read_dir("tests/data/fuzz")
        .unwrap()
        .map(|file| file.unwrap().path())
        .collect();
    assert!(!files.is_empty(), "tests/data/fuzz has no file");
    check_files(&files.iter().map(|file| file.as_path()).collect::<Vec<_>>());
}

#[test]
fn each_kernel_equals_the_reference_on_the_fetched_corpora() {
    let files = ["twitter", "citm_catalog", "canada"].map(|name| format!("benches/data/fetched/{name}.json"));
    check_files(&files.each_ref().map(Path::new));
}

fn word<const N: usize>(bytes: &[u8], index: usize) -> [u8; N] {
    let mut rest = index;
    core::array::from_fn(|_| {
        let byte = bytes[rest % bytes.len()];
        rest /= bytes.len();
        byte
    })
}

const UTF8_BYTES: [u8; 18] = [
    b'a', 0x80, 0x90, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xe1, 0xed, 0xee, 0xf0, 0xf1, 0xf4, 0xf5, 0xff,
];

#[test]
fn escapes_quotes_and_scalars_across_blocks_give_the_tokens_of_the_scan() {
    const BYTES: [u8; 6] = [b'"', b'\\', b'a', b' ', b',', 0x01];
    let kernels = kernels();
    let mut input = Vec::new();
    for index in 0..BYTES.len().pow(6) {
        let word: [u8; 6] = word(&BYTES, index);
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

#[test]
fn each_utf8_sequence_across_blocks_gives_the_error_of_std() {
    let kernels = kernels();
    let mut input = Vec::new();
    for index in 0..UTF8_BYTES.len().pow(4) {
        let word: [u8; 4] = word(&UTF8_BYTES, index);
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

#[test]
fn each_short_input_gives_the_utf8_error_of_std() {
    let kernels = kernels();
    let check = |input: &[u8]| {
        let expected = core::str::from_utf8(input).is_err();
        for &kernel in &kernels {
            let block = classify(kernel, input, 0, Carry::default());
            assert_eq!(block.carry.utf8_error(), expected, "{kernel:?} {input:x?}");
        }
    };
    for len in 1..=3 {
        for value in 0..1_u32 << (8 * len) {
            check(&value.to_le_bytes()[..len]);
        }
    }
    for first in 0..=u8::MAX {
        for index in 0..UTF8_BYTES.len().pow(3) {
            let [second, third, fourth] = word(&UTF8_BYTES, index);
            check(&[first, second, third, fourth]);
        }
    }
}

#[test]
fn the_cursor_gives_the_tokens_before_the_first_invalid_byte_then_its_error() {
    const SEQUENCES: [&[u8]; 5] = [&[0x80], &[0xff], &[0xc0, 0x80], &[0xe2, 0x82], &[0xed, 0xa0, 0x80]];
    let kernels = kernels();
    let check = |input: &[u8]| {
        check_blocks(input, &kernels);
        let tokens = scan(input);
        for &kernel in &kernels {
            check_cursor(input, &tokens, kernel);
        }
    };
    for sequence in SEQUENCES {
        for at in 1..128 {
            let mut input = vec![b','; 132];
            input[0] = b'[';
            input[at..at + sequence.len()].copy_from_slice(sequence);
            check(&input);
        }
        for at in 2..255 {
            let mut input = vec![b'a'; 260];
            input[..2].copy_from_slice(b"[\"");
            input[258..].copy_from_slice(b"\"]");
            input[at..at + sequence.len()].copy_from_slice(sequence);
            check(&input);
        }
    }
}

#[test]
fn a_scalar_in_the_last_bytes_of_a_block_comes_before_its_utf8_error() {
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

#[test]
fn the_last_block_of_each_length_gives_the_tokens_of_the_scan() {
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

#[test]
fn each_byte_in_each_lane_gives_the_tokens_of_its_class() {
    let kernels = kernels();
    for lane in 0..64 {
        for byte in 0..=u8::MAX {
            let mut outside = vec![b'x'; 64];
            outside[lane] = byte;
            let mut inside = [&b"\""[..], &[b'x'; 64]].concat();
            inside[lane + 1] = byte;
            for input in [outside, inside] {
                check_blocks(&input, &kernels);
                let tokens = scan(&input);
                for &kernel in &kernels {
                    check_cursor(&input, &tokens, kernel);
                }
            }
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn the_selection_gives_a_kernel_only_with_each_of_its_features() {
    use crate::cpu::{Caps, Timing, close_implications};
    let without =
        |missing: Caps| Kernel::ALL[Kernel::fastest(close_implications(Caps::ALL.difference(missing)), Timing::Public)];
    assert_eq!(without(Caps::NONE), Kernel::ALL[0]);
    #[cfg(target_arch = "x86_64")]
    {
        use crate::cpu::x86::{
            AVX, AVX2, AVX512BW, AVX512F, AVX512VBMI, AVX512VBMI2, AVX512VL, BMI1, BMI2, F16C, FMA, PCLMULQDQ, SSE3,
            SSE41, SSE42, SSSE3,
        };
        let cases = [
            (AVX512VBMI, Kernel::Avx512),
            (AVX512F, Kernel::Avx2),
            (AVX512BW, Kernel::Avx2),
            (AVX512VL, Kernel::Avx2),
            (AVX512VBMI2, Kernel::Avx2),
            (FMA, Kernel::Avx2),
            (F16C, Kernel::Avx2),
            (AVX2, Kernel::Swar),
            (AVX, Kernel::Swar),
            (SSE42, Kernel::Swar),
            (SSE41, Kernel::Swar),
            (SSSE3, Kernel::Swar),
            (SSE3, Kernel::Swar),
            (PCLMULQDQ, Kernel::Swar),
            (BMI1, Kernel::Swar),
            (BMI2, Kernel::Swar),
        ];
        for (missing, kernel) in cases {
            assert_eq!(without(missing), kernel, "without {missing:?}");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use crate::cpu::aarch64::{AES, DIT, NEON};
        for (missing, kernel) in [(DIT, Kernel::NeonPmull), (AES, Kernel::Neon), (NEON, Kernel::Swar)] {
            assert_eq!(without(missing), kernel, "without {missing:?}");
        }
    }
}
