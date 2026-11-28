//! Write WebAuthn request options. A real ceremony supplies a fresh challenge.

use rscodec::{json, with_dit};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let challenge = [0x9f; 16];
    let mut out = [0; 128];
    let len = with_dit(|dit| {
        json::write(&mut out, |root| {
            root.object(|options| {
                options.member("challenge").base64url_secret(dit, &challenge)?;
                options.member("rpId").str("example.com")
            })
        })
    })??;
    assert_eq!(
        &out[..len],
        br#"{"challenge":"n5-fn5-fn5-fn5-fn5-fnw","rpId":"example.com"}"#
    );
    Ok(())
}
