use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// Scan for phone numbers focusing on NANP (North American Numbering Plan) and
/// common international formats.
///
/// Recognized patterns (digit-only matching, formatting-agnostic):
/// - +1-XXX-XXX-XXXX / 1-XXX-XXX-XXXX: 11 digits starting with 1, area code 2-9
/// - XXX-XXX-XXXX: 10 digits, area code 2-9
/// - (XXX) XXX-XXXX: 10 digits with parens
/// - +(XXX)XXX-XXXX / +(XX)XXX-XXXX: international with country code
/// - Toll-free: 800/888/877/866/855/844/833 prefix
pub fn scan_phone(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    scan_formatted_phone(text, hits);
    scan_raw_digit_phone(bytes, text, hits);
    scan_international_phone(text, hits);
}

/// International phone: +XX or +XXX prefix, then parens area code + number.
/// Matches: +(066)815-9185, +(066) 815-9185, +44-20-7946-0958, +1-212-555-0123
fn scan_international_phone(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(6) {
        if bytes[i] != b'+' {
            continue;
        }
        // Skip country code: digits + possible parens for area code
        let mut j = i + 1;
        let mut paren_depth = 0i32;
        while j < bytes.len() {
            let b = bytes[j];
            if b == b'(' {
                paren_depth += 1;
            } else if b == b')' {
                paren_depth -= 1;
                if paren_depth < 0 { break; }
            } else if !b.is_ascii_digit() && b != b'-' && b != b' ' {
                break;
            }
            j += 1;
        }
        // Need at least 7 total digits
        if j - i < 8 {
            continue;
        }
        let candidate = &text[i..j];
        let digit_count = candidate.chars().filter(|c| c.is_ascii_digit()).count();
        if digit_count >= 7 && digit_count <= 15 {
            // Don't flag standalone "+NNNNNNN" as phone (must have formatting)
            if candidate.contains('-') || candidate.contains('(') || candidate.contains(' ') {
                hits.push(Hit {
                    kind: HitKind::Phone,
                    text: String::from(candidate),
                    start: i,
                    end: j,
                });
            }
        }
    }
}

/// Phone with formatting: (XXX) XXX-XXXX or XXX-XXX-XXXX
fn scan_formatted_phone(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    if bytes.len() < 12 {
        return;
    }

    for i in 0..bytes.len() - 11 {
        // Pattern: (XXX) XXX-XXXX  or  XXX-XXX-XXXX
        let parens = bytes[i] == b'(';

        if parens {
            // (XXX) XXX-XXXX  — 14 chars total
            if i + 13 > bytes.len() {
                continue;
            }
            // dash between exchange and subscriber is at offset 9 from '('
            if bytes[i + 9] != b'-' {
                continue;
            }
            let area: [u8; 3] = [bytes[i + 1], bytes[i + 2], bytes[i + 3]];
            let exch: [u8; 3] = [bytes[i + 6], bytes[i + 7], bytes[i + 8]];
            let subs: [u8; 4] = [bytes[i + 10], bytes[i + 11], bytes[i + 12], bytes[i + 13]];

            if valid_phone_parts(&area, &exch, &subs) {
                hits.push(Hit {
                    kind: HitKind::Phone,
                    text: String::from(&text[i..i + 14]),
                    start: i,
                    end: i + 14,
                });
            }
            continue;
        }

        // XXX-XXX-XXXX (10 digits, 2 dashes)
        if i + 12 > bytes.len() {
            continue;
        }
        if bytes[i + 3] == b'-' && bytes[i + 7] == b'-' {
            let area = [bytes[i], bytes[i + 1], bytes[i + 2]];
            let exch = [bytes[i + 4], bytes[i + 5], bytes[i + 6]];
            let subs = [bytes[i + 8], bytes[i + 9], bytes[i + 10], bytes[i + 11]];

            if valid_phone_parts(&area, &exch, &subs) {
                hits.push(Hit {
                    kind: HitKind::Phone,
                    text: String::from(&text[i..i + 12]),
                    start: i,
                    end: i + 12,
                });
            }
        }
    }
}

/// Raw digit runs: 10-digit (NANP) and 11-digit (with country code 1)
fn scan_raw_digit_phone(bytes: &[u8], text: &str, hits: &mut Vec<Hit>) {
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
        let digits = &bytes[start..i];

        let area_idx = match len {
            10 => 0_usize,     // NANP: area code first
            11 if digits[0] == b'1' => 1_usize, // Country code 1 + NANP
            _ => continue,
        };

        let area = &digits[area_idx..area_idx + 3];
        let exch = &digits[area_idx + 3..area_idx + 6];
        let subs = &digits[area_idx + 6..area_idx + 10];

        if valid_phone_parts(area, exch, subs) {
            hits.push(Hit {
                kind: HitKind::Phone,
                text: String::from(&text[start..i]),
                start,
                end: i,
            });
        }
    }
}

fn valid_phone_parts(area: &[u8], exch: &[u8], subs: &[u8]) -> bool {
    // area code: first digit 2-9, all digits
    area.len() == 3
        && area[0] >= b'2'
        && area[0] <= b'9'
        && area.iter().all(u8::is_ascii_digit)
    // exchange: first digit 2-9, all digits
        && exch.len() == 3
        && exch[0] >= b'2'
        && exch[0] <= b'9'
        && exch.iter().all(u8::is_ascii_digit)
    // subscriber: all digits
        && subs.len() == 4
        && subs.iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashed_phone() {
        let mut hits = Vec::new();
        scan_phone("Call 212-555-0123 today", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Phone);
        assert_eq!(hits[0].text, "212-555-0123");
    }

    #[test]
    fn test_parenthesized_phone() {
        let mut hits = Vec::new();
        scan_phone("(212) 555-0123", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "(212) 555-0123");
    }

    #[test]
    fn test_raw_10digit() {
        let mut hits = Vec::new();
        scan_phone("2125550123", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "2125550123");
    }

    #[test]
    fn test_rejects_invalid_area() {
        let mut hits = Vec::new();
        scan_phone("012-555-0123", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_rejects_invalid_exchange() {
        let mut hits = Vec::new();
        scan_phone("212-055-0123", &mut hits);
        assert!(hits.is_empty());
    }
}
