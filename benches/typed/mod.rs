use std::{borrow::Cow, fmt};

use rscodec::json::{self, ErrorKind, Kind, Limits, Profile, Slot, Str, Value};
use serde::{
    Deserialize, Deserializer,
    de::{self, SeqAccess, Visitor},
};

const LIMITS: Limits = Limits {
    input_bytes_max: 1 << 16,
    depth_max: 16,
    string_bytes_max: 1 << 14,
    members_max: 256,
    elements_max: 256,
    number_digits_max: 32,
};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Text<'a>(Cow<'a, str>);

impl<'de: 'a, 'a> Deserialize<'de> for Text<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(TextVisitor)
    }
}

struct TextVisitor;

impl<'de> Visitor<'de> for TextVisitor {
    type Value = Text<'de>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string")
    }

    fn visit_borrowed_str<E: de::Error>(self, text: &'de str) -> Result<Self::Value, E> {
        Ok(Text(Cow::Borrowed(text)))
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
        Ok(Text(Cow::Owned(text.to_owned())))
    }

    fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
        Ok(Text(Cow::Owned(text)))
    }
}

/// The `aud` claim: one string or an array of strings (RFC 7519 section 4.1.3).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Audience<'a>(Vec<Text<'a>>);

impl<'de: 'a, 'a> Deserialize<'de> for Audience<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AudienceVisitor;
        impl<'de> Visitor<'de> for AudienceVisitor {
            type Value = Audience<'de>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or an array of strings")
            }

            fn visit_borrowed_str<E: de::Error>(self, text: &'de str) -> Result<Self::Value, E> {
                TextVisitor.visit_borrowed_str(text).map(|text| Audience(vec![text]))
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                TextVisitor.visit_str(text).map(|text| Audience(vec![text]))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut texts = Vec::new();
                while let Some(text) = seq.next_element()? {
                    texts.push(text);
                }
                Ok(Audience(texts))
            }
        }
        deserializer.deserialize_any(AudienceVisitor)
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct ClientData<'a> {
    #[serde(rename = "type")]
    kind: Text<'a>,
    challenge: Text<'a>,
    origin: Text<'a>,
    #[serde(rename = "crossOrigin")]
    cross_origin: Option<bool>,
    #[serde(rename = "topOrigin")]
    top_origin: Option<Text<'a>>,
}

/// A JWK (RFC 7517 section 4, RFC 7518 section 6).
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Jwk<'a> {
    kty: Text<'a>,
    crv: Option<Text<'a>>,
    x: Option<Text<'a>>,
    y: Option<Text<'a>>,
    d: Option<Text<'a>>,
    n: Option<Text<'a>>,
    e: Option<Text<'a>>,
    alg: Option<Text<'a>>,
    kid: Option<Text<'a>>,
    #[serde(rename = "use")]
    usage: Option<Text<'a>>,
    x5t: Option<Text<'a>>,
    #[serde(rename = "x5t#S256")]
    x5t_s256: Option<Text<'a>>,
    x5c: Option<Vec<Text<'a>>>,
}

/// A JWK set (RFC 7517 section 5).
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Jwks<'a> {
    keys: Vec<Jwk<'a>>,
}

/// A JOSE header (RFC 7515 section 4.1, RFC 7516 section 4.1, RFC 7797 section 3).
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Header<'a> {
    alg: Text<'a>,
    enc: Option<Text<'a>>,
    kid: Option<Text<'a>>,
    typ: Option<Text<'a>>,
    cty: Option<Text<'a>>,
    apu: Option<Text<'a>>,
    apv: Option<Text<'a>>,
    crit: Option<Vec<Text<'a>>>,
    b64: Option<bool>,
    exp: Option<u64>,
    epk: Option<Jwk<'a>>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Address<'a> {
    formatted: Option<Text<'a>>,
    street_address: Option<Text<'a>>,
    locality: Option<Text<'a>>,
    region: Option<Text<'a>>,
    postal_code: Option<Text<'a>>,
    country: Option<Text<'a>>,
}

/// JWT and OIDC claims (RFC 7519 section 4.1, OpenID Connect Core 1.0 sections 2 and 5.1).
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Claims<'a> {
    iss: Option<Text<'a>>,
    sub: Option<Text<'a>>,
    aud: Option<Audience<'a>>,
    exp: Option<u64>,
    nbf: Option<u64>,
    iat: Option<u64>,
    auth_time: Option<u64>,
    jti: Option<Text<'a>>,
    nonce: Option<Text<'a>>,
    azp: Option<Text<'a>>,
    acr: Option<Text<'a>>,
    amr: Option<Vec<Text<'a>>>,
    sid: Option<Text<'a>>,
    scope: Option<Text<'a>>,
    client_id: Option<Text<'a>>,
    name: Option<Text<'a>>,
    email: Option<Text<'a>>,
    email_verified: Option<bool>,
    groups: Option<Vec<Text<'a>>>,
    address: Option<Address<'a>>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub(crate) struct Discovery<'a> {
    issuer: Text<'a>,
    authorization_endpoint: Text<'a>,
    token_endpoint: Option<Text<'a>>,
    userinfo_endpoint: Option<Text<'a>>,
    jwks_uri: Text<'a>,
    registration_endpoint: Option<Text<'a>>,
    end_session_endpoint: Option<Text<'a>>,
    introspection_endpoint: Option<Text<'a>>,
    revocation_endpoint: Option<Text<'a>>,
    scopes_supported: Option<Vec<Text<'a>>>,
    response_types_supported: Vec<Text<'a>>,
    grant_types_supported: Option<Vec<Text<'a>>>,
    subject_types_supported: Vec<Text<'a>>,
    id_token_signing_alg_values_supported: Vec<Text<'a>>,
    token_endpoint_auth_methods_supported: Option<Vec<Text<'a>>>,
    code_challenge_methods_supported: Option<Vec<Text<'a>>>,
    claims_supported: Option<Vec<Text<'a>>>,
    request_parameter_supported: Option<bool>,
    require_pushed_authorization_requests: Option<bool>,
}

pub(crate) trait Typed {
    type Of<'a>: Deserialize<'a> + PartialEq;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<Self::Of<'a>, json::Error>;
}

const MISSING: json::Error = json::Error::new(ErrorKind::Type, 0);

#[inline(always)]
fn name<'k, 'a: 'k>(key: Str<'a>, buf: &'k mut [u8; 64]) -> &'k str {
    key.decode(buf).unwrap_or("")
}

#[inline(always)]
fn text<'a>(value: Value<'_, 'a, '_>) -> Result<Text<'a>, json::Error> {
    let text = value.str()?;
    Ok(Text(match text.borrowed() {
        Some(text) => Cow::Borrowed(text),
        None => {
            let mut buf = vec![0; text.len()];
            text.decode(&mut buf)?;
            Cow::Owned(String::from_utf8(buf).expect("a decoded string is UTF-8"))
        }
    }))
}

#[inline(always)]
fn texts<'a>(value: Value<'_, 'a, '_>) -> Result<Vec<Text<'a>>, json::Error> {
    let mut texts = Vec::new();
    let mut elements = value.elements()?;
    while let Some(value) = elements.next()? {
        texts.push(text(value)?);
    }
    Ok(texts)
}

#[inline(always)]
fn jwk<'a>(value: Value<'_, 'a, '_>) -> Result<Jwk<'a>, json::Error> {
    let mut key = Jwk {
        kty: Text(Cow::Borrowed("")),
        crv: None,
        x: None,
        y: None,
        d: None,
        n: None,
        e: None,
        alg: None,
        kid: None,
        usage: None,
        x5t: None,
        x5t_s256: None,
        x5c: None,
    };
    let mut kty = None;
    let mut buf = [0; 64];
    let mut members = value.members()?;
    while let Some((member, value)) = members.next()? {
        match name(member, &mut buf) {
            "kty" => kty = Some(text(value)?),
            "crv" => key.crv = Some(text(value)?),
            "x" => key.x = Some(text(value)?),
            "y" => key.y = Some(text(value)?),
            "d" => key.d = Some(text(value)?),
            "n" => key.n = Some(text(value)?),
            "e" => key.e = Some(text(value)?),
            "alg" => key.alg = Some(text(value)?),
            "kid" => key.kid = Some(text(value)?),
            "use" => key.usage = Some(text(value)?),
            "x5t" => key.x5t = Some(text(value)?),
            "x5t#S256" => key.x5t_s256 = Some(text(value)?),
            "x5c" => key.x5c = Some(texts(value)?),
            _ => {}
        }
    }
    key.kty = kty.ok_or(MISSING)?;
    Ok(key)
}

pub(crate) struct A1;

impl Typed for A1 {
    type Of<'a> = ClientData<'a>;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<ClientData<'a>, json::Error> {
        json::read(
            document,
            &LIMITS,
            Profile::Rfc8259,
            scratch,
            #[inline(always)]
            |reader| {
                let (mut kind, mut challenge, mut origin, mut cross_origin, mut top_origin) =
                    (None, None, None, None, None);
                let mut buf = [0; 64];
                let mut members = reader.value().members()?;
                while let Some((key, value)) = members.next()? {
                    match name(key, &mut buf) {
                        "type" => kind = Some(text(value)?),
                        "challenge" => challenge = Some(text(value)?),
                        "origin" => origin = Some(text(value)?),
                        "crossOrigin" => cross_origin = Some(value.bool()?),
                        "topOrigin" => top_origin = Some(text(value)?),
                        _ => {}
                    }
                }
                Ok(ClientData {
                    kind: kind.ok_or(MISSING)?,
                    challenge: challenge.ok_or(MISSING)?,
                    origin: origin.ok_or(MISSING)?,
                    cross_origin,
                    top_origin,
                })
            },
        )
    }
}

pub(crate) struct A2;

impl Typed for A2 {
    type Of<'a> = Header<'a>;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<Header<'a>, json::Error> {
        json::read(
            document,
            &LIMITS,
            Profile::Rfc8259,
            scratch,
            #[inline(always)]
            |reader| {
                let mut alg = None;
                let mut header = Header {
                    alg: Text(Cow::Borrowed("")),
                    enc: None,
                    kid: None,
                    typ: None,
                    cty: None,
                    apu: None,
                    apv: None,
                    crit: None,
                    b64: None,
                    exp: None,
                    epk: None,
                };
                let mut buf = [0; 64];
                let mut members = reader.value().members()?;
                while let Some((key, value)) = members.next()? {
                    match name(key, &mut buf) {
                        "alg" => alg = Some(text(value)?),
                        "enc" => header.enc = Some(text(value)?),
                        "kid" => header.kid = Some(text(value)?),
                        "typ" => header.typ = Some(text(value)?),
                        "cty" => header.cty = Some(text(value)?),
                        "apu" => header.apu = Some(text(value)?),
                        "apv" => header.apv = Some(text(value)?),
                        "crit" => header.crit = Some(texts(value)?),
                        "b64" => header.b64 = Some(value.bool()?),
                        "exp" => header.exp = Some(value.number()?.to_u64()?),
                        "epk" => header.epk = Some(jwk(value)?),
                        _ => {}
                    }
                }
                header.alg = alg.ok_or(MISSING)?;
                Ok(header)
            },
        )
    }
}

#[inline(always)]
fn address<'a>(value: Value<'_, 'a, '_>) -> Result<Address<'a>, json::Error> {
    let mut address = Address {
        formatted: None,
        street_address: None,
        locality: None,
        region: None,
        postal_code: None,
        country: None,
    };
    let mut buf = [0; 64];
    let mut members = value.members()?;
    while let Some((key, value)) = members.next()? {
        match name(key, &mut buf) {
            "formatted" => address.formatted = Some(text(value)?),
            "street_address" => address.street_address = Some(text(value)?),
            "locality" => address.locality = Some(text(value)?),
            "region" => address.region = Some(text(value)?),
            "postal_code" => address.postal_code = Some(text(value)?),
            "country" => address.country = Some(text(value)?),
            _ => {}
        }
    }
    Ok(address)
}

pub(crate) struct A3;

impl Typed for A3 {
    type Of<'a> = Claims<'a>;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<Claims<'a>, json::Error> {
        json::read(
            document,
            &LIMITS,
            Profile::Rfc8259,
            scratch,
            #[inline(always)]
            |reader| {
                let mut claims = Claims {
                    iss: None,
                    sub: None,
                    aud: None,
                    exp: None,
                    nbf: None,
                    iat: None,
                    auth_time: None,
                    jti: None,
                    nonce: None,
                    azp: None,
                    acr: None,
                    amr: None,
                    sid: None,
                    scope: None,
                    client_id: None,
                    name: None,
                    email: None,
                    email_verified: None,
                    groups: None,
                    address: None,
                };
                let mut buf = [0; 64];
                let mut members = reader.value().members()?;
                while let Some((key, mut value)) = members.next()? {
                    match name(key, &mut buf) {
                        "iss" => claims.iss = Some(text(value)?),
                        "sub" => claims.sub = Some(text(value)?),
                        "aud" => {
                            claims.aud = Some(Audience(if value.kind()? == Kind::String {
                                vec![text(value)?]
                            } else {
                                texts(value)?
                            }));
                        }
                        "exp" => claims.exp = Some(value.number()?.to_u64()?),
                        "nbf" => claims.nbf = Some(value.number()?.to_u64()?),
                        "iat" => claims.iat = Some(value.number()?.to_u64()?),
                        "auth_time" => claims.auth_time = Some(value.number()?.to_u64()?),
                        "jti" => claims.jti = Some(text(value)?),
                        "nonce" => claims.nonce = Some(text(value)?),
                        "azp" => claims.azp = Some(text(value)?),
                        "acr" => claims.acr = Some(text(value)?),
                        "amr" => claims.amr = Some(texts(value)?),
                        "sid" => claims.sid = Some(text(value)?),
                        "scope" => claims.scope = Some(text(value)?),
                        "client_id" => claims.client_id = Some(text(value)?),
                        "name" => claims.name = Some(text(value)?),
                        "email" => claims.email = Some(text(value)?),
                        "email_verified" => claims.email_verified = Some(value.bool()?),
                        "groups" => claims.groups = Some(texts(value)?),
                        "address" => claims.address = Some(address(value)?),
                        _ => {}
                    }
                }
                Ok(claims)
            },
        )
    }
}

pub(crate) struct A4;

impl Typed for A4 {
    type Of<'a> = Discovery<'a>;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<Discovery<'a>, json::Error> {
        json::read(
            document,
            &LIMITS,
            Profile::Rfc8259,
            scratch,
            #[inline(always)]
            |reader| {
                let (mut issuer, mut authorization, mut jwks, mut response_types, mut subject_types, mut algs) =
                    (None, None, None, None, None, None);
                let mut found = Discovery {
                    issuer: Text(Cow::Borrowed("")),
                    authorization_endpoint: Text(Cow::Borrowed("")),
                    token_endpoint: None,
                    userinfo_endpoint: None,
                    jwks_uri: Text(Cow::Borrowed("")),
                    registration_endpoint: None,
                    end_session_endpoint: None,
                    introspection_endpoint: None,
                    revocation_endpoint: None,
                    scopes_supported: None,
                    response_types_supported: Vec::new(),
                    grant_types_supported: None,
                    subject_types_supported: Vec::new(),
                    id_token_signing_alg_values_supported: Vec::new(),
                    token_endpoint_auth_methods_supported: None,
                    code_challenge_methods_supported: None,
                    claims_supported: None,
                    request_parameter_supported: None,
                    require_pushed_authorization_requests: None,
                };
                let mut buf = [0; 64];
                let mut members = reader.value().members()?;
                while let Some((key, value)) = members.next()? {
                    match name(key, &mut buf) {
                        "issuer" => issuer = Some(text(value)?),
                        "authorization_endpoint" => authorization = Some(text(value)?),
                        "token_endpoint" => found.token_endpoint = Some(text(value)?),
                        "userinfo_endpoint" => found.userinfo_endpoint = Some(text(value)?),
                        "jwks_uri" => jwks = Some(text(value)?),
                        "registration_endpoint" => found.registration_endpoint = Some(text(value)?),
                        "end_session_endpoint" => found.end_session_endpoint = Some(text(value)?),
                        "introspection_endpoint" => found.introspection_endpoint = Some(text(value)?),
                        "revocation_endpoint" => found.revocation_endpoint = Some(text(value)?),
                        "scopes_supported" => found.scopes_supported = Some(texts(value)?),
                        "response_types_supported" => response_types = Some(texts(value)?),
                        "grant_types_supported" => found.grant_types_supported = Some(texts(value)?),
                        "subject_types_supported" => subject_types = Some(texts(value)?),
                        "id_token_signing_alg_values_supported" => algs = Some(texts(value)?),
                        "token_endpoint_auth_methods_supported" => {
                            found.token_endpoint_auth_methods_supported = Some(texts(value)?);
                        }
                        "code_challenge_methods_supported" => {
                            found.code_challenge_methods_supported = Some(texts(value)?)
                        }
                        "claims_supported" => found.claims_supported = Some(texts(value)?),
                        "request_parameter_supported" => found.request_parameter_supported = Some(value.bool()?),
                        "require_pushed_authorization_requests" => {
                            found.require_pushed_authorization_requests = Some(value.bool()?);
                        }
                        _ => {}
                    }
                }
                found.issuer = issuer.ok_or(MISSING)?;
                found.authorization_endpoint = authorization.ok_or(MISSING)?;
                found.jwks_uri = jwks.ok_or(MISSING)?;
                found.response_types_supported = response_types.ok_or(MISSING)?;
                found.subject_types_supported = subject_types.ok_or(MISSING)?;
                found.id_token_signing_alg_values_supported = algs.ok_or(MISSING)?;
                Ok(found)
            },
        )
    }
}

pub(crate) struct A5;

impl Typed for A5 {
    type Of<'a> = Jwks<'a>;

    fn rscodec<'a>(document: &'a [u8], scratch: &mut [Slot]) -> Result<Jwks<'a>, json::Error> {
        json::read(
            document,
            &LIMITS,
            Profile::Rfc8259,
            scratch,
            #[inline(always)]
            |reader| {
                let mut keys = None;
                let mut buf = [0; 64];
                let mut members = reader.value().members()?;
                while let Some((key, value)) = members.next()? {
                    if name(key, &mut buf) == "keys" {
                        let mut set = Vec::new();
                        let mut elements = value.elements()?;
                        while let Some(value) = elements.next()? {
                            set.push(jwk(value)?);
                        }
                        keys = Some(set);
                    }
                }
                Ok(Jwks {
                    keys: keys.ok_or(MISSING)?,
                })
            },
        )
    }
}
