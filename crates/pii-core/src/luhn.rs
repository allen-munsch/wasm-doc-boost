use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// IIN (Issuer Identification Number) prefix table.
/// (prefix_len, prefix_str, brand_name)
const IIN_PREFIXES: &[(usize, &str)] = &[
    (1, "4"),       // Visa
    (2, "51"),      // MasterCard
    (2, "52"),      // MasterCard
    (2, "53"),      // MasterCard
    (2, "54"),      // MasterCard
    (2, "55"),      // MasterCard
    (2, "34"),      // Amex
    (2, "37"),      // Amex
    (4, "6011"),    // Discover
    (2, "65"),      // Discover
    (3, "644"),     // Discover
    (2, "36"),      // Diners Club
    (2, "38"),      // Diners Club
    (3, "300"),     // Diners Club
    (3, "301"),     // Diners Club
    (3, "302"),     // Diners Club
    (3, "303"),     // Diners Club
    (3, "304"),     // Diners Club
    (3, "305"),     // Diners Club
    (2, "35"),      // JCB
];

/// Validate a digit string with the Luhn (mod-10) algorithm.
fn luhn_ok(digits: &[u8]) -> bool {
    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut n = (d - b'0') as u32;
        if double {
            n *= 2;
            if n > 9 {
                n -= 9;
            }
        }
        sum += n;
        double = !double;
    }
    sum % 10 == 0
}

/// Check if a digit string starts with a known card IIN prefix.
fn has_card_iin(digits: &[u8]) -> bool {
    for &(len, prefix) in IIN_PREFIXES {
        if digits.len() >= len && &digits[..len] == prefix.as_bytes() {
            return true;
        }
    }
    false
}

/// Scan for credit/debit card PANs (Primary Account Numbers).
/// These are 13-19 digit sequences that pass the Luhn checksum and
/// start with a known IIN prefix.
pub fn scan_pan(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let len = i - start;
        if !(13..=19).contains(&len) {
            continue;
        }
        let digits = &bytes[start..i];
        if luhn_ok(digits) && has_card_iin(digits) {
            hits.push(Hit {
                kind: HitKind::Pan,
                text: String::from(&text[start..i]),
                start,
                end: i,
            });
            // Scan nearby for CVV and expiry
            scan_context(text, start, i, hits);
        }
    }
}

/// After finding a PAN at [pan_start, pan_end), scan nearby text for
/// CVV (3-4 digit standalone number) and expiry (MM/YY or MM/YYYY).
fn scan_context(text: &str, _pan_start: usize, pan_end: usize, hits: &mut Vec<Hit>) {
    // Look in the rest of the current line and next line for CVV / expiry
    let rest = &text[pan_end..];
    let window = &rest[..rest.len().min(80)];

    // --- CVV: 3 or 4 digits isolated (not part of a longer number) ---
    for chunk in window.split(|b: char| !b.is_ascii_digit()) {
        let trimmed = chunk.trim();
        let n = trimmed.len();
        if n == 3 || n == 4 {
            let pos = pan_end + (trimmed.as_ptr() as usize - rest.as_ptr() as usize);
            // Verify it's not part of a larger number by checking boundaries
            let is_isolated = if pos > 0 {
                !text.as_bytes()[pos.saturating_sub(1)].is_ascii_digit()
            } else {
                true
            } && if pos + n < text.len() {
                !text.as_bytes()[pos + n].is_ascii_digit()
            } else {
                true
            };
            if is_isolated {
                hits.push(Hit {
                    kind: HitKind::Cvv,
                    text: String::from(&text[pos..pos + n]),
                    start: pos,
                    end: pos + n,
                });
            }
        }
    }

    // --- Expiry: MM/YY or MM/YYYY near PAN ---
    let exp_idx = window.find('/');
    if let Some(slash_pos) = exp_idx {
        let abs_slash = pan_end + slash_pos;
        if slash_pos >= 2 {
            let mm = &window[slash_pos - 2..slash_pos];
            let yy_part = if slash_pos + 3 < window.len() && window.as_bytes()[slash_pos + 1].is_ascii_digit() {
                if slash_pos + 5 < window.len() && window.as_bytes()[slash_pos + 3].is_ascii_digit() {
                    &window[slash_pos + 1..slash_pos + 5] // YYYY
                } else {
                    &window[slash_pos + 1..slash_pos + 3] // YY
                }
            } else {
                ""
            };
            let ok = mm.as_bytes().iter().all(u8::is_ascii_digit)
                && yy_part.as_bytes().iter().all(u8::is_ascii_digit)
                && mm.parse::<u8>().is_ok_and(|m| (1..=12).contains(&m));
            if ok && !yy_part.is_empty() {
                let end = abs_slash + 1 + yy_part.len();
                hits.push(Hit {
                    kind: HitKind::Expiry,
                    text: String::from(&text[abs_slash - 2..end]),
                    start: abs_slash - 2,
                    end,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luhn_valid_visa() {
        assert!(luhn_ok(b"4111111111111111"));
    }

    #[test]
    fn test_luhn_invalid() {
        assert!(!luhn_ok(b"4111111111111112"));
    }

    #[test]
    fn test_iin_visa() {
        assert!(has_card_iin(b"4111111111111111"));
        assert!(!has_card_iin(b"9111111111111111"));
    }

    #[test]
    fn test_scan_pan_in_text() {
        let mut hits = Vec::new();
        scan_pan("Card: 4111111111111111 expires 12/25", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Pan);
        assert_eq!(hits[0].text, "4111111111111111");
    }

    #[test]
    fn test_scan_pan_rejects_product_code() {
        let mut hits = Vec::new();
        // 16 digits that fails Luhn (a product code, not a card)
        scan_pan("Product: 3921837456218390", &mut hits);
        assert!(hits.is_empty(), "Should not flag non-Luhn numbers");
    }
}
