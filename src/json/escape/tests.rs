extern crate std;

use super::{BLOCK, escape, escapes, escapes_of_words, first, tail};

fn first_escape(block: &[u8; BLOCK]) -> usize {
    match escapes(block) {
        0 => BLOCK,
        mask => first(mask),
    }
}

/// The bytes that RFC 8259 section 7 requires to escape.
fn needs_escape(byte: u8) -> bool {
    byte == b'"' || byte == b'\\' || byte < 0x20
}

#[test]
fn the_scan_finds_the_first_byte_to_escape_at_each_place_for_each_byte() {
    for place in 0..BLOCK {
        for byte in 0..=u8::MAX {
            let mut bytes = [b'a'; BLOCK];
            bytes[place] = byte;
            let expected = if needs_escape(byte) { place } else { BLOCK };
            assert_eq!(first_escape(&bytes), expected, "byte {byte:#04x} at {place}");
            for later in place + 1..BLOCK {
                bytes[later] = 0x1f;
                assert_eq!(first_escape(&bytes), if needs_escape(byte) { place } else { later });
                bytes[later] = b'a';
            }
        }
    }
}

#[test]
fn the_words_of_each_tail_length_hold_the_tail_and_spaces_after_it() {
    let text: [u8; BLOCK] = core::array::from_fn(|index| u8::try_from(index * 17 + 1).unwrap_or(0));
    for len in 0..BLOCK {
        let (low, high) = tail(&text[..len]);
        let mut bytes = [0; BLOCK];
        bytes[..8].copy_from_slice(&low.to_le_bytes());
        bytes[8..].copy_from_slice(&high.to_le_bytes());
        assert_eq!(bytes[..len], text[..len], "the bytes of a tail of {len}");
        assert!(
            bytes[len..].iter().all(|&byte| byte == b' '),
            "the spaces after a tail of {len}"
        );
        assert_eq!(
            escapes_of_words(low, high),
            escapes(&bytes),
            "the scan of the words of a tail of {len}"
        );
    }
}

#[test]
fn each_byte_to_escape_has_its_rfc_8785_escape() {
    for byte in (0..=u8::MAX).filter(|&byte| needs_escape(byte)) {
        let expected = match byte {
            b'"' => std::string::String::from("\\\""),
            b'\\' => "\\\\".into(),
            0x08 => "\\b".into(),
            0x09 => "\\t".into(),
            0x0a => "\\n".into(),
            0x0c => "\\f".into(),
            0x0d => "\\r".into(),
            _ => std::format!("\\u{byte:04x}"),
        };
        let (form, len) = escape(byte);
        assert_eq!(form[..len], *expected.as_bytes(), "the escape of {byte:#04x}");
    }
}
