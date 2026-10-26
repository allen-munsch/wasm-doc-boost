use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// Scan for card PINs (4-6 isolated digits near a "PIN" keyword) and
/// ISO 9564 PIN blocks (16 hex chars near a "PIN block" keyword).
pub fn scan_pin(text: &str, hits: &mut Vec<Hit>) {
    scan_pin_digits(text, hits);
    scan_pin_block(text, hits);
}

/// "PIN", "PIN:", "PIN code" etc. followed within a short window by a
/// 4-6 digit run that is not part of a longer number.
fn scan_pin_digits(text: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    let bytes = text.as_bytes();

    for kw in ["pin:", "pin ", "pin=", "pin-"] {
        let mut from = 0;
        while let Some(rel) = lower[from..].find(kw) {
            let kw_start = from + rel;
            // Only when "PIN" is a standalone word, not inside e.g. "spin".
            if kw_start > 0 && bytes[kw_start - 1].is_ascii_alphanumeric() {
                from = kw_start + kw.len();
                continue;
            }
            let search_start = kw_start + kw.len();
            let search_end = (search_start + 12).min(text.len());
            let slice = &text[search_start..search_end];

            // Find the first 4-6 digit run in the window.
            let mut i = 0;
            while i < slice.len() {
                if !slice.as_bytes()[i].is_ascii_digit() {
                    i += 1;
                    continue;
                }
                let ds = i;
                while i < slice.len() && slice.as_bytes()[i].is_ascii_digit() {
                    i += 1;
                }
                let len = i - ds;
                if (4..=6).contains(&len) {
                    // Must be isolated from a longer digit or hex run, so a
                    // 4-digit prefix of a PIN block (e.g. "0412ABCDEF123456")
                    // is not separately flagged as a PIN.
                    let abs_start = search_start + ds;
                    let abs_end = search_start + i;
                    let leading_digit = abs_start > 0 && bytes[abs_start - 1].is_ascii_digit();
                    let trailing_digit = abs_end < bytes.len() && bytes[abs_end].is_ascii_digit();
                    let trailing_hex =
                        abs_end + 1 < bytes.len() && bytes[abs_end].is_ascii_hexdigit();
                    if !leading_digit && !trailing_digit && !trailing_hex {
                        hits.push(Hit {
                            kind: HitKind::Pin,
                            text: String::from(&text[abs_start..abs_end]),
                            start: abs_start,
                            end: abs_end,
                        });
                    }
                }
                break;
            }
            from = kw_start + kw.len();
        }
    }
}

/// ISO 9564 PIN block: a 16-character hex string near a "PIN block" keyword.
/// The first nibble encodes the format (0-3); here we require exactly 16
/// uppercase/lowercase hex chars adjacent to the keyword.
fn scan_pin_block(text: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    let bytes = text.as_bytes();

    for kw in ["pin block:", "pin block ", "pin block=", "pinblock:"] {
        let mut from = 0;
        while let Some(rel) = lower[from..].find(kw) {
            let kw_start = from + rel;
            let search_start = kw_start + kw.len();
            let search_end = (search_start + 24).min(text.len());
            let slice = &text[search_start..search_end];

            let mut i = 0;
            while i + 16 <= slice.len() {
                let candidate = &slice.as_bytes()[i..i + 16];
                let all_hex = candidate.iter().all(u8::is_ascii_hexdigit);
                if all_hex {
                    let abs_start = search_start + i;
                    // Require the block to be isolated from other hex/alnum chars.
                    let leading = abs_start > 0 && bytes[abs_start - 1].is_ascii_alphanumeric();
                    let trailing = abs_start + 16 < bytes.len()
                        && bytes[abs_start + 16].is_ascii_alphanumeric();
                    if !leading && !trailing {
                        hits.push(Hit {
                            kind: HitKind::PinBlock,
                            text: String::from(&text[abs_start..abs_start + 16]),
                            start: abs_start,
                            end: abs_start + 16,
                        });
                    }
                    break;
                }
                i += 1;
            }
            from = kw_start + kw.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_colon() {
        let mut hits = Vec::new();
        scan_pin("Card PIN: 4821 for this account", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Pin);
        assert_eq!(hits[0].text, "4821");
    }

    #[test]
    fn test_pin_6_digit() {
        let mut hits = Vec::new();
        scan_pin("Your PIN is 123456.", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Pin);
        assert_eq!(hits[0].text, "123456");
    }

    #[test]
    fn test_pin_rejects_embedded_in_word() {
        let mut hits = Vec::new();
        // "spinning" contains "pin " — must not match.
        scan_pin("The wheel is spinning fast", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_pin_rejects_too_long() {
        let mut hits = Vec::new();
        // 7 digits is not a PIN.
        scan_pin("PIN: 1234567", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_pin_not_flagged_inside_pin_block() {
        let mut hits = Vec::new();
        // The block starts with 4 digits; those must not also emit a PIN hit.
        scan_pin("PIN block: 0412ABCDEF123456", &mut hits);
        assert!(
            !hits.iter().any(|h| h.kind == HitKind::Pin),
            "4-digit prefix of a PIN block must not be flagged as a PIN"
        );
    }

    #[test]
    fn test_pin_block_valid() {
        let mut hits = Vec::new();
        let text = "PIN block: 04123456789ABCDE";
        scan_pin(text, &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::PinBlock);
        assert_eq!(hits[0].text, "04123456789ABCDE");
    }

    #[test]
    fn test_pin_block_rejects_non_hex() {
        let mut hits = Vec::new();
        scan_pin("PIN block: 04123456789ABCZZ", &mut hits);
        assert!(hits.is_empty());
    }
}
