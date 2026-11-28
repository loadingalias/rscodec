//! Decode a public identifier into a stack buffer.

use rscodec::base64::{Error, Profile};

fn main() -> Result<(), Error> {
    let mut bytes = [0; 4];
    let id = Profile::Url.decode(b"AQID_w", &mut bytes)?;
    assert_eq!(id, [1, 2, 3, 255]);
    Ok(())
}
