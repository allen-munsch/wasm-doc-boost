use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// Scan for email addresses using a simple state machine.
///
/// Matches: local@domain.tld
/// - local: alphanumeric + ._%+-
/// - domain: alphanumeric + .-
/// - tld: 2+ alphabetic chars
pub fn scan_email(text: &str, hits: &mut Vec<Hit>) {
    let bytes = text.as_bytes();

    for i in 0..bytes.len().saturating_sub(5) {
        // Find '@' not at the start or end
        if bytes[i] != b'@' {
            continue;
        }
        if i == 0 || i >= bytes.len() - 3 {
            continue;
        }

        // Walk backwards to find start of local part
        let local_start = {
            let mut j = i - 1;
            loop {
                let valid = bytes[j].is_ascii_alphanumeric()
                    || bytes[j] == b'.'
                    || bytes[j] == b'_'
                    || bytes[j] == b'%'
                    || bytes[j] == b'+'
                    || bytes[j] == b'-';
                if !valid {
                    break;
                }
                if j == 0 {
                    // Include j=0 if valid, then break
                    break;
                }
                j -= 1;
            }
            // j is at last valid character (or one before start)
            // The local part starts at j if j is valid, otherwise j+1
            if bytes[j].is_ascii_alphanumeric()
                || bytes[j] == b'.'
                || bytes[j] == b'_'
                || bytes[j] == b'%'
                || bytes[j] == b'+'
                || bytes[j] == b'-'
            {
                j
            } else {
                j + 1
            }
        };

        if local_start >= i {
            continue; // no valid local part
        }

        // Walk forward to find end of domain part
        let domain_end = {
            let mut j = i + 1;
            let mut saw_dot = false;
            while j < bytes.len() {
                let valid = bytes[j].is_ascii_alphanumeric() || bytes[j] == b'.' || bytes[j] == b'-';
                if !valid {
                    break;
                }
                if bytes[j] == b'.' {
                    saw_dot = true;
                }
                j += 1;
            }
            // Must have at least a dot-tld: domain.tld
            if !saw_dot || j - i < 4 {
                continue; // skip: too short or no dot
            }
            // TLD must be at least 2 alphabetic chars
            if j >= 3 && bytes[j - 2..j].iter().all(u8::is_ascii_alphabetic) {
                j
            } else if j >= 4
                && bytes[j - 3..j].iter().all(u8::is_ascii_alphabetic)
                && bytes[j - 4] == b'.'
            {
                j
            } else {
                continue;
            }
        };

        hits.push(Hit {
            kind: HitKind::Email,
            text: String::from(&text[local_start..domain_end]),
            start: local_start,
            end: domain_end,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_email() {
        let mut hits = Vec::new();
        scan_email("Contact: john@example.com for help", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Email);
        assert_eq!(hits[0].text, "john@example.com");
    }

    #[test]
    fn test_email_with_plus() {
        let mut hits = Vec::new();
        scan_email("user+tag@domain.org", &mut hits);
        assert_eq!(hits[0].text, "user+tag@domain.org");
    }

    #[test]
    fn test_no_at_sign() {
        let mut hits = Vec::new();
        scan_email("just a regular sentence", &mut hits);
        assert!(hits.is_empty());
    }
}
