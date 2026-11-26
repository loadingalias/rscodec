//! The writer: compact JSON into a slice that the caller lends.

use super::{Error, ErrorKind, escape, float, integer};

#[cfg(test)]
mod tests;

/// Writes one JSON value into the start of `out` with `value`, and returns the length of the
/// text. The bytes of `out` after that length are unspecified.
///
/// The text is compact RFC 8259 JSON. The writer writes each `,`, `:`, quote, and escape, and it
/// allocates no memory. This closure writes the options of a WebAuthn registration ceremony. The
/// challenge is a secret during the ceremony, so the secret encoder writes it in the scope of
/// [`with_dit`](crate::with_dit):
///
/// ```
/// use rscodec::{
///     Dit,
///     json::{self, ErrorKind},
/// };
///
/// let (user_id, challenge) = (b"user-4711", [0x9f; 16]);
/// let options = |dit: &Dit, out: &mut [u8]| {
///     json::write(out, |options| {
///         options.object(|options| {
///             options.member("rp").object(|rp| {
///                 rp.member("id").str("example.com")?;
///                 rp.member("name").str("Example")
///             })?;
///             options.member("user").object(|user| {
///                 user.member("id").base64url(user_id)?;
///                 user.member("name").str("alice@example.com")?;
///                 user.member("displayName").str("Alice \"Al\" Smith")
///             })?;
///             options.member("challenge").base64url_secret(dit, &challenge)?;
///             options.member("pubKeyCredParams").array(|params| {
///                 // ES256 and RS256.
///                 for alg in [-7, -257] {
///                     params.item().object(|param| {
///                         param.member("type").str("public-key")?;
///                         param.member("alg").i64(alg)
///                     })?;
///                 }
///                 Ok(())
///             })?;
///             options.member("timeout").u64(60_000)?;
///             options.member("attestation").str("none")
///         })
///     })
/// };
/// let mut out = [0; 512];
/// let len = rscodec::with_dit(|dit| options(dit, &mut out))??;
/// assert_eq!(
///     &out[..len],
///     concat!(
///         r#"{"rp":{"id":"example.com","name":"Example"},"#,
///         r#""user":{"id":"dXNlci00NzEx","name":"alice@example.com","displayName":"Alice \"Al\" Smith"},"#,
///         r#""challenge":"n5-fn5-fn5-fn5-fn5-fnw","#,
///         r#""pubKeyCredParams":[{"type":"public-key","alg":-7},{"type":"public-key","alg":-257}],"#,
///         r#""timeout":60000,"attestation":"none"}"#,
///     )
///     .as_bytes(),
/// );
///
/// // A slice that is too short refuses, and the text that fits stays valid. Here the object does
/// // not fit, so nothing is written.
/// let error = rscodec::with_dit(|dit| options(dit, &mut out[..len - 1]))?.unwrap_err();
/// assert_eq!((error.kind(), error.offset()), (ErrorKind::Full, 0));
/// # Ok::<(), Box<dyn core::error::Error>>(())
/// ```
///
/// # Errors
///
/// The first error of `value`, which the writer gives as [`ErrorKind::Full`] when `out` ends
/// and as [`ErrorKind::NonFinite`] for NaN or an infinity. [`ErrorKind::Type`] at offset 0 if
/// `value` writes no value.
#[inline]
pub fn write(out: &mut [u8], value: impl FnOnce(Place<'_>) -> Result<(), Error>) -> Result<usize, Error> {
    let mut len = 0;
    value(Place {
        out,
        len: &mut len,
        key: None,
        first: 0,
    })?;
    if len == 0 {
        return Err(Error::new(ErrorKind::Type, 0));
    }
    Ok(len)
}

/// The place of one JSON value: the value of a [`write()`], an item of an [`Array`], or the value of
/// a member of an [`Object`]. Each method writes the value, after its `,` and its key, and takes
/// the place, so each place gets at most one value. A dropped place writes nothing.
///
/// Each method writes all of its bytes, or none: after an error, the written length is the length
/// before the call.
#[derive(Debug)]
pub struct Place<'w> {
    out: &'w mut [u8],
    len: &'w mut usize,
    key: Option<&'w str>,
    /// The index of the first value of the container: a value there has no `,` before it. The
    /// writer keeps it, so it reads no byte of `out` back.
    first: usize,
}

/// An open JSON object. [`Object::member`] gives the place of the value of each member.
#[derive(Debug)]
pub struct Object<'w> {
    out: &'w mut [u8],
    len: usize,
    first: usize,
}

/// An open JSON array. [`Array::item`] gives the place of each item.
#[derive(Debug)]
pub struct Array<'w> {
    out: &'w mut [u8],
    len: usize,
    first: usize,
}

impl Object<'_> {
    /// The place of the value of the member `key`. The writer escapes the key, and it does not
    /// check that the object has no other member `key`.
    #[inline]
    pub fn member<'k>(&'k mut self, key: &'k str) -> Place<'k> {
        Place {
            out: self.out,
            len: &mut self.len,
            key: Some(key),
            first: self.first,
        }
    }
}

impl Array<'_> {
    /// The place of the next item.
    #[inline]
    pub fn item(&mut self) -> Place<'_> {
        Place {
            out: self.out,
            len: &mut self.len,
            key: None,
            first: self.first,
        }
    }
}

impl Place<'_> {
    /// Writes the JSON string of `text`, with the escapes of RFC 8785: [`string_len`] gives its
    /// exact length.
    ///
    /// [`string_len`]: super::string_len
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn str(mut self, text: &str) -> Result<(), Error> {
        let full = Error::new(ErrorKind::Full, *self.len);
        let at = self.begin().ok_or(full)?;
        *self.len = escape::write(self.out, at, text).ok_or(full)?;
        Ok(())
    }

    /// Writes `value` in decimal: at most [`I64_BYTES_MAX`](super::I64_BYTES_MAX) bytes.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn i64(self, value: i64) -> Result<(), Error> {
        self.integer(value.unsigned_abs(), value < 0)
    }

    /// Writes `value` in decimal: at most [`U64_BYTES_MAX`](super::U64_BYTES_MAX) bytes.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn u64(self, value: u64) -> Result<(), Error> {
        self.integer(value, false)
    }

    /// Writes the shortest digits that read back as `value`, in the layout of `{:?}` of
    /// `core::fmt`, for example `1.0`, `-0.0`, `0.0001`, `1e16`, and `1.5e-7`: at most
    /// [`F64_BYTES_MAX`](super::F64_BYTES_MAX) bytes.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::NonFinite`] for NaN or an infinity, which JSON has no number for, and
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn f64(mut self, value: f64) -> Result<(), Error> {
        if !value.is_finite() {
            return Err(Error::new(ErrorKind::NonFinite, *self.len));
        }
        let full = Error::new(ErrorKind::Full, *self.len);
        let at = self.begin().ok_or(full)?;
        let mut spare = [0; float::SPAN];
        let len = float::write(value, stage(self.out, at, &mut spare));
        *self.len = keep(self.out, at, &spare, len).ok_or(full)?;
        Ok(())
    }

    /// Writes `true` or `false`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn bool(self, value: bool) -> Result<(), Error> {
        self.literal(if value { b"true" } else { b"false" })
    }

    /// Writes `null`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[inline]
    pub fn null(self) -> Result<(), Error> {
        self.literal(b"null")
    }

    /// Writes the bytes in base64url without padding (RFC 4648 section 5), as a JSON string. The
    /// alphabet needs no escape, so the writer encodes into the slice with no other pass. For a
    /// secret, for example a challenge, use [`Place::base64url_secret`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[cfg(feature = "base64")]
    #[inline]
    pub fn base64url(self, bytes: &[u8]) -> Result<(), Error> {
        self.base64url_by(bytes, |bytes, text| crate::base64::Profile::Url.encode(bytes, text))
    }

    /// Writes the text of [`Place::base64url`] for secret bytes, for example the challenge of a
    /// WebAuthn ceremony. It encodes with
    /// [`Profile::encode_secret`](crate::base64::Profile::encode_secret), in the scope of
    /// [`with_dit`](crate::with_dit) that the proof [`Dit`](crate::Dit) shows. The key, the
    /// structure, and each length are public.
    ///
    /// The member writes no character of the secret if the value does not fit. After a success,
    /// the slice holds the text, also when a container around it fails later and rolls back its
    /// length. The caller owns the slice, so the caller clears it.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Full`] if the slice ends.
    #[cfg(feature = "base64")]
    #[inline]
    pub fn base64url_secret(self, dit: &crate::Dit, bytes: &[u8]) -> Result<(), Error> {
        self.base64url_by(bytes, |bytes, text| {
            crate::base64::Profile::Url.encode_secret(dit, bytes, text)
        })
    }

    /// Writes an object with the members that `members` writes, and its `}` after them.
    ///
    /// # Errors
    ///
    /// The first error of `members`, or [`ErrorKind::Full`] if the slice ends. Either way the
    /// object writes no byte, and the offset of the error is the written length before it.
    #[inline]
    pub fn object(self, members: impl FnOnce(&mut Object<'_>) -> Result<(), Error>) -> Result<(), Error> {
        self.container(b'{', b'}', |out, len| {
            let mut object = Object { out, len, first: len };
            members(&mut object)?;
            Ok(object.len)
        })
    }

    /// Writes an array with the items that `items` writes, and its `]` after them.
    ///
    /// # Errors
    ///
    /// The first error of `items`, or [`ErrorKind::Full`] if the slice ends. Either way the
    /// array writes no byte, and the offset of the error is the written length before it.
    #[inline]
    pub fn array(self, items: impl FnOnce(&mut Array<'_>) -> Result<(), Error>) -> Result<(), Error> {
        self.container(b'[', b']', |out, len| {
            let mut array = Array { out, len, first: len };
            items(&mut array)?;
            Ok(array.len)
        })
    }

    /// Writes `open`, the content that `fill` writes, and `close`. `fill` takes the slice without
    /// its last byte, which stays free for `close`, and the length after `open`, and it gives the
    /// length after the content.
    #[inline(always)]
    fn container(
        mut self,
        open: u8,
        close: u8,
        fill: impl FnOnce(&mut [u8], usize) -> Result<usize, Error>,
    ) -> Result<(), Error> {
        let start = *self.len;
        let full = Error::new(ErrorKind::Full, start);
        let at = self.begin().ok_or(full)?;
        let last = self.out.len().checked_sub(1).filter(|&last| at < last).ok_or(full)?;
        self.out[at] = open;
        // The container writes nothing after an error, so the written length stays `start`.
        let end = fill(&mut self.out[..last], at + 1).map_err(|error| Error::new(error.kind(), start))?;
        self.out[end] = close;
        *self.len = end + 1;
        Ok(())
    }

    /// The index after the `,` and the key before the value, or `None` if the slice ends. A
    /// value after the `{` or the `[` of its container, or at the start, has no `,`.
    #[inline(always)]
    fn begin(&mut self) -> Option<usize> {
        let mut at = *self.len;
        if at > self.first {
            *self.out.get_mut(at)? = b',';
            at += 1;
        }
        if let Some(key) = self.key {
            at = escape::write(self.out, at, key)?;
            *self.out.get_mut(at)? = b':';
            at += 1;
        }
        Some(at)
    }

    /// Writes the JSON string of the base64url text of `bytes`. It tests the room for the whole
    /// value before `encode` writes a character into the slice that it takes, whose length is the
    /// length of the text.
    #[cfg(feature = "base64")]
    #[inline(always)]
    fn base64url_by(
        mut self,
        bytes: &[u8],
        encode: impl for<'t> FnOnce(&[u8], &'t mut [u8]) -> Result<&'t mut [u8], crate::base64::Error>,
    ) -> Result<(), Error> {
        let full = Error::new(ErrorKind::Full, *self.len);
        let at = self.begin().ok_or(full)?;
        // The `,`, the key, and `at` fit in the slice, so `at + 2` does not overflow.
        let end = crate::base64::Profile::Url
            .encoded_len(bytes.len())
            .and_then(|len| len.checked_add(at + 2))
            .ok_or(full)?;
        let Some([open, text @ .., close]) = self.out.get_mut(at..end) else {
            return Err(full);
        };
        *open = b'"';
        // `text` has the length of the text, so the encoder does not refuse.
        encode(bytes, text).map_err(|_| full)?;
        *close = b'"';
        *self.len = end;
        Ok(())
    }

    /// Writes `text`, a literal of JSON.
    #[inline(always)]
    fn literal(mut self, text: &'static [u8]) -> Result<(), Error> {
        let end = self.begin().and_then(|at| {
            self.out.get_mut(at..at + text.len())?.copy_from_slice(text);
            Some(at + text.len())
        });
        *self.len = end.ok_or_else(|| Error::new(ErrorKind::Full, *self.len))?;
        Ok(())
    }

    #[inline(always)]
    fn integer(mut self, value: u64, negative: bool) -> Result<(), Error> {
        let full = Error::new(ErrorKind::Full, *self.len);
        let at = self.begin().ok_or(full)?;
        let mut spare = [0; integer::SPAN];
        let len = integer::write(value, negative, stage(self.out, at, &mut spare));
        *self.len = keep(self.out, at, &spare, len).ok_or(full)?;
        Ok(())
    }
}

/// The `SPAN` bytes of `out` from `at` where a printer stores its text, or `spare` near the end of
/// `out`. A printer stores up to `SPAN` bytes and keeps the length that it returns.
#[inline(always)]
fn stage<'s, const SPAN: usize>(out: &'s mut [u8], at: usize, spare: &'s mut [u8; SPAN]) -> &'s mut [u8; SPAN] {
    out.get_mut(at..).and_then(<[u8]>::first_chunk_mut).unwrap_or(spare)
}

/// The index after the `len` bytes that a printer keeps from `at`, or `None` if they do not fit.
/// It copies them from `spare` when [`stage`] gave `spare`.
#[inline(always)]
fn keep<const SPAN: usize>(out: &mut [u8], at: usize, spare: &[u8; SPAN], len: usize) -> Option<usize> {
    let rest = out.get_mut(at..)?;
    if rest.len() < SPAN {
        rest.get_mut(..len)?.copy_from_slice(&spare[..len]);
    }
    Some(at + len)
}
