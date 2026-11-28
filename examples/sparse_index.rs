//! Read the top-level version and checksum from a Cargo index entry.

use rscodec::json::{self, Limits, Profile, Slot, Token};

fn main() -> Result<(), json::Error> {
    let entry = concat!(
        r#"{"name":"demo","vers":"1.2.3","deps":[],"#,
        r#""cksum":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","#,
        r#""features":{"cksum":["other"]},"yanked":false}"#,
    )
    .as_bytes();
    let limits = Limits {
        input_bytes_max: 8192,
        depth_max: 8,
        string_bytes_max: 1024,
        members_max: 32,
        elements_max: 64,
        number_digits_max: 16,
    };
    let mut scratch = [Slot::EMPTY; 32];
    let (mut version, mut checksum) = (None, None);
    let mut depth = 0;
    let mut previous = Token::Null;
    json::visit(entry, &limits, Profile::Rfc8259, &mut scratch, |token| {
        match (depth, previous, token) {
            (_, _, Token::ObjectStart | Token::ArrayStart) => depth += 1,
            (_, _, Token::ObjectEnd | Token::ArrayEnd) => depth -= 1,
            (1, Token::Key(key), Token::Str(text)) if key == "vers" => version = Some(text),
            (1, Token::Key(key), Token::Str(text)) if key == "cksum" => checksum = Some(text),
            _ => {}
        }
        previous = token;
        Ok(())
    })?;
    assert!(version.is_some_and(|v| v == "1.2.3"));
    assert!(checksum.is_some_and(|v| v == "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    Ok(())
}
