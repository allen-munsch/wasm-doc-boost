use alloc::string::String;
use alloc::vec::Vec;

use crate::luhn::{has_card_iin, luhn_ok};
use crate::{Hit, HitKind};

/// Scan for full magnetic-stripe track data.
///
/// Track 1: `%B<PAN>^<name>^<YYMM><service><discretionary>?`
/// Track 2: `;<PAN>=<YYMM><service><discretionary>?`
///
/// The PAN embedded in the track must pass Luhn + IIN, so a bare
/// `%B...?` string that does not contain a valid card number is not flagged.
pub fn scan_track(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Track 1 sentinel "%B" or Track 2 sentinel ";"
        let is_track1 = bytes[i] == b'%' && i + 1 < bytes.len() && bytes[i + 1] == b'B';
        let is_track2 = bytes[i] == b';';
        if !is_track1 && !is_track2 {
            i += 1;
            continue;
        }

        // The track body runs from the sentinel to the terminating '?'.
        let body_start = i;
        let body_end = match text[body_start..].find('?') {
            Some(rel) => body_start + rel + 1,
            None => break,
        };
        // Cap the body at a sane length to avoid scanning across the document.
        if body_end - body_start > 200 {
            i += 1;
            continue;
        }
        let body = &text[body_start..body_end];

        // Extract the digit run that should be the PAN: everything after the
        // sentinel up to the first non-digit (Track 2 '=' or Track 1 '^').
        let pan_start = body_start + if is_track1 { 2 } else { 1 };
        let mut pan_end = pan_start;
        while pan_end < body_end && bytes[pan_end].is_ascii_digit() {
            pan_end += 1;
        }
        let pan_len = pan_end - pan_start;
        if !(13..=19).contains(&pan_len) {
            i += 1;
            continue;
        }
        let pan = &bytes[pan_start..pan_end];

        // The next character must be the expected field separator.
        let expected_sep = if is_track1 { b'^' } else { b'=' };
        if pan_end >= body_end || bytes[pan_end] != expected_sep {
            i += 1;
            continue;
        }

        if luhn_ok(pan) && has_card_iin(pan) {
            hits.push(Hit {
                kind: HitKind::TrackData,
                text: String::from(body),
                start: body_start,
                end: body_end,
            });
        }

        i = body_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track1_valid() {
        let mut hits = Vec::new();
        // %B + 16-digit Visa (Luhn-valid) + ^name^ + YYMM + service + discretionary
        let text = "%B4111111111111111^DOE/JOHN^2512101000000000000000000?";
        scan_track(text, &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::TrackData);
        assert_eq!(hits[0].text, text);
    }

    #[test]
    fn test_track2_valid() {
        let mut hits = Vec::new();
        let text = ";4111111111111111=25121010000000000000?";
        scan_track(text, &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::TrackData);
    }

    #[test]
    fn test_track_rejects_non_luhn_pan() {
        let mut hits = Vec::new();
        // Same structure but the PAN fails Luhn — must not be flagged.
        let text = ";4111111111111112=25121010000000000000?";
        scan_track(text, &mut hits);
        assert!(hits.is_empty(), "non-Luhn PAN inside track must not match");
    }

    #[test]
    fn test_track_rejects_wrong_separator() {
        let mut hits = Vec::new();
        // Track 1 sentinel but Track 2 '=' separator.
        let text = "%B4111111111111111=25121010000000000000?";
        scan_track(text, &mut hits);
        assert!(hits.is_empty());
    }
}
