use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// Scan for US Social Security Numbers.
///
/// SSNs appear as 9 consecutive digits (with or without formatting:
/// `123-45-6789`, `123456789`, `123 45 6789`).
///
/// Validation rules:
/// - Area number: 001-899, not 666
/// - Group number: 01-99
/// - Serial number: 0001-9999
/// - Must NOT be a substring of a longer digit run (phone=10, PAN=13-19)
pub fn scan_ssn(text: &str, hits: &mut Vec<Hit>) {
    // First pass: find 9-digit runs with standard formatting (XXX-XX-XXXX)
    scan_formatted_ssn(text, hits);

    // Second pass: find unformatted 9-digit runs
    scan_raw_9digit_runs(text, hits);
}

/// SSN with dashes: XXX-XX-XXXX
fn scan_formatted_ssn(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    if bytes.len() < 11 {
        return;
    }
    for i in 0..bytes.len() - 10 {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3] == b'-'
            && bytes[i + 4].is_ascii_digit()
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6] == b'-'
            && (i + 7..i + 11).all(|j| bytes[j].is_ascii_digit())
            && !is_part_of_longer_run(bytes, i, i + 11)
        {
            let area = parse_3digits(&bytes[i..i + 3]);
            let group = parse_2digits(&bytes[i + 4..i + 6]);
            let serial = parse_4digits(&bytes[i + 7..i + 11]);
            if valid_ssn_parts(area, group, serial) {
                hits.push(Hit {
                    kind: HitKind::Ssn,
                    text: String::from(&text[i..i + 11]),
                    start: i,
                    end: i + 11,
                });
            }
        }
    }
}

/// Unformatted: exactly 9 digits, not part of a longer digit run.
fn scan_raw_9digit_runs(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();
    let slice: &[u8] = bytes;
    let mut i = 0;
    while i < slice.len() {
        if !slice[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < slice.len() && slice[i].is_ascii_digit() {
            i += 1;
        }
        let len = i - start;
        // Only exact 9-digit runs that aren't part of longer sequences
        if len == 9 && !is_part_of_longer_run(bytes, start, i) {
            let digits = &bytes[start..i];
            let area = parse_3digits(&digits[0..3]);
            let group = parse_2digits(&digits[3..5]);
            let serial = parse_4digits(&digits[5..9]);
            if valid_ssn_parts(area, group, serial) {
                hits.push(Hit {
                    kind: HitKind::Ssn,
                    text: String::from(&text[start..i]),
                    start,
                    end: i,
                });
            }
        }
    }
}

/// Check that the digit run at [start, end) is not a substring of a longer
/// digit sequence (prevents false positives on phone numbers, PANs, etc.).
fn is_part_of_longer_run(bytes: &[u8], start: usize, end: usize) -> bool {
    let leading_digit = start > 0 && bytes[start - 1].is_ascii_digit();
    let trailing_digit = end < bytes.len() && bytes[end].is_ascii_digit();
    leading_digit || trailing_digit
}

fn parse_3digits(bytes: &[u8]) -> u16 {
    ((bytes[0] - b'0') as u16) * 100 + ((bytes[1] - b'0') as u16) * 10 + ((bytes[2] - b'0') as u16)
}

fn parse_2digits(bytes: &[u8]) -> u16 {
    ((bytes[0] - b'0') as u16) * 10 + ((bytes[1] - b'0') as u16)
}

fn parse_4digits(bytes: &[u8]) -> u16 {
    ((bytes[0] - b'0') as u16) * 1000
        + ((bytes[1] - b'0') as u16) * 100
        + ((bytes[2] - b'0') as u16) * 10
        + ((bytes[3] - b'0') as u16)
}

/// Validate SSN component ranges.
fn valid_ssn_parts(area: u16, group: u16, serial: u16) -> bool {
    (1..=899).contains(&area)
        && area != 666
        && (1..=99).contains(&group)
        && (1..=9999).contains(&serial)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formatted_ssn() {
        let mut hits = Vec::new();
        scan_ssn("SSN: 123-45-6789", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Ssn);
        assert_eq!(hits[0].text, "123-45-6789");
    }

    #[test]
    fn test_raw_ssn() {
        let mut hits = Vec::new();
        scan_ssn("SSN: 123456789", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Ssn);
        assert_eq!(hits[0].text, "123456789");
    }

    #[test]
    fn test_rejects_phone_number() {
        let mut hits = Vec::new();
        // 10-digit phone number — should not flag the 9-digit substring
        scan_ssn("Call 1234567890 for help", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_rejects_invalid_area() {
        let mut hits = Vec::new();
        scan_ssn("SSN: 000-12-3456", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_rejects_666_area() {
        let mut hits = Vec::new();
        scan_ssn("SSN: 666-12-3456", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_rejects_invalid_group() {
        let mut hits = Vec::new();
        scan_ssn("SSN: 123-00-4567", &mut hits);
        assert!(hits.is_empty());
    }
}
