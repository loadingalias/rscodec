//! Read the algorithm from a JOSE header.

use rscodec::json::{self, Limits, Profile, Slot};

fn main() -> Result<(), json::Error> {
    let header = br#"{"\u0061lg":"ES256","kid":"key-1"}"#;
    let limits = Limits {
        input_bytes_max: 8192,
        depth_max: 4,
        string_bytes_max: 256,
        members_max: 8,
        elements_max: 8,
        number_digits_max: 16,
    };
    let mut scratch = [Slot::EMPTY; 16];
    let algorithm = json::read(header, &limits, Profile::IJson, &mut scratch, |reader| {
        let mut members = reader.value().members()?;
        while let Some((key, value)) = members.next()? {
            if key == "alg" {
                return Ok(Some(value.str()?));
            }
        }
        Ok(None)
    })?;
    assert!(algorithm.is_some_and(|name| name == "ES256"));
    Ok(())
}
