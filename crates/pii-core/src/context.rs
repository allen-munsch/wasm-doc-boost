use alloc::string::String;
use alloc::vec::Vec;

use crate::{Hit, HitKind};

/// Scan for address-like patterns (ZIP codes, street suffixes, "Address:" prefix)
/// and name patterns ("Name:", "Buyer:") in OCR text.
///
/// These are heuristic — weaker than deterministic Luhn/SSN — but catch the
/// most common PII categories found on invoices and receipts.
pub fn scan_context(text: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();

    // ZIP code: 5 digits, isolated (not part of longer number)
    scan_zip(text, hits);

    // Address prefix: "Address:", "Addr:"
    if lower.contains("address:") || lower.contains("addr:") {
        address_hit(text, "Address", hits);
    }

    // Name prefix: "Name:", "Buyer:", "Customer:", "Contact:"
    // Note: "Buyer" without colon handled via bill_to_hit below (needs name/address split).
    for prefix in &["name:", "buyer:", "customer:", "contact:", "ship to:"] {
        if lower.contains(prefix) {
            name_hit(text, prefix, hits);
        }
    }

    // BILL_TO / Bill to / Buyer: extract person name (first 1-4 capitalized words)
    // and the remaining address text as ADDRESS hit.
    for marker in &["bill to:", "bill to ", "bill_to", "buyer "] {
        if lower.contains(marker) {
            bill_to_hit(text, marker, hits);
            break;
        }
    }

    // Address prefix: "Address:", "Addr:" — narrow, high-precision heuristic.
    // The CRF model handles broader street-address detection.
    // NOTE: removed the "street suffix + ZIP" heuristic — it was marking the
    // entire document as ADDRESS and producing massive FPs when substrings
    // like "st" in "East" triggered false matches.

    // Account numbers near financial keywords (GSTIN, Account, Bank, etc.)
    scan_account(text, hits);
}

/// 5-digit ZIP code (not part of a longer digit run).
fn scan_zip(text: &str, hits: &mut Vec<Hit>) {
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
        if len != 5 {
            continue;
        }
        // Must be isolated — surrounded by non-alphanumeric chars.
        // Prevents matching "24322" inside GSTIN codes like "T2ABCDEI24322E4".
        let leading = start > 0 && slice[start - 1].is_ascii_alphanumeric();
        let trailing = i < slice.len() && slice[i].is_ascii_alphanumeric();
        if leading || trailing {
            continue;
        }
        hits.push(Hit {
            kind: HitKind::Zip,
            text: String::from(&text[start..i]),
            start,
            end: i,
        });
    }
}

/// Scan for digit sequences (6-20 digits) near financial keywords.
/// Catches GSTIN, bank account, routing numbers the CRF may miss in short contexts.
fn scan_account(text: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    const KEYWORDS: &[&str] = &[
        "gstin", "gst", "account", "acct", "bank", "routing", "swift",
        "iban", "bsb", "sort code", "account no", "account number",
    ];

    for kw in KEYWORDS {
        if let Some(kw_pos) = lower.find(kw) {
            // Search within ±80 chars of the keyword for a 6-20 digit run
            let search_start = kw_pos.saturating_sub(80);
            let search_end = (kw_pos + kw.len() + 80).min(text.len());
            let slice = &text[search_start..search_end];

            let chars: Vec<char> = slice.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if !chars[i].is_ascii_digit() { i += 1; continue; }
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() { i += 1; }
                let len = i - start;
                if len < 6 || len > 20 { continue; }

                // Convert char offset to byte offset
                let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
                let byte_end: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                let abs_start = search_start + byte_start;
                let abs_end = search_start + byte_end;

                // Avoid duplicates: check if this span overlaps an existing ACCOUNT hit
                if hits.iter().any(|h| h.start < abs_end && h.end > abs_start) {
                    continue;
                }

                hits.push(Hit {
                    kind: HitKind::BankAccount,
                    text: String::from(&text[abs_start..abs_end]),
                    start: abs_start,
                    end: abs_end,
                });
            }
        }
    }
}

/// Find the position of an address marker and add a contextual hit.
fn address_hit(text: &str, _marker: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    if let Some(pos) = lower.find("address:") {
        let end = (pos + 100).min(text.len());
        hits.push(Hit {
            kind: HitKind::Address,
            text: String::from(&text[pos..end]),
            start: pos,
            end,
        });
    }
}

/// Find the position of a name marker and add a contextual hit.
fn name_hit(text: &str, marker: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    if let Some(pos) = lower.find(marker) {
        let start = pos + marker.len();
        // Find end: first newline, pipe, or comma after the name
        let remainder = &text[start..];
        let end_offset = remainder.find('\n')
            .or_else(|| remainder.find('|'))
            .or_else(|| remainder.find(','))
            .unwrap_or(remainder.len().min(50));
        let end = start + end_offset;
        let name_text = text[start..end].trim();
        if !name_text.is_empty() && name_text.len() > 2 {
            hits.push(Hit {
                kind: HitKind::Name,
                text: String::from(name_text),
                start,
                end,
            });
        }
    }
}

/// BILL_TO / Bill to: extract person name (first 1-4 capitalized words)
/// and the remaining address text as ADDRESS hit.
fn bill_to_hit(text: &str, marker: &str, hits: &mut Vec<Hit>) {
    let lower = text.to_lowercase();
    if let Some(pos) = lower.find(marker) {
        let after = &text[pos + marker.len()..];
        let after = after.trim_start();

        // Find end of block: "Tel", "Email", "Site", or comma
        let block_end = after.find("Tel ")
            .or_else(|| after.find("Email "))
            .or_else(|| after.find("Site "))
            .or_else(|| after.find(','))
            .unwrap_or(after.len().min(100));

        let block = &after[..block_end].trim();
        if block.is_empty() { return; }

        // Split into words and determine where the name ends
        let words: Vec<&str> = block.split_whitespace().collect();
        if words.is_empty() { return; }

        // The name is the first 2-4 alphabetic-heavy words before any digit starts
        let mut name_end_idx = 0usize;
        for (i, w) in words.iter().enumerate() {
            let has_digit = w.chars().any(|c| c.is_ascii_digit());
            let is_short = w.len() <= 1;
            if has_digit || (is_short && i > 1) {
                break;
            }
            name_end_idx = i + 1;
            if name_end_idx >= 4 { break; } // max 4 words for name
        }

        if name_end_idx == 0 { return; }

        let name_words = &words[..name_end_idx];
        let name_str = name_words.join(" ");
        if let Some(name_pos) = text.find(&name_str) {
            let name_end = name_pos + name_str.len();
            hits.push(Hit {
                kind: HitKind::Name,
                text: String::from(name_str),
                start: name_pos,
                end: name_end,
            });

            // Remaining text is the address
            let addr_start = name_end;
            let addr_text = text[addr_start..pos + marker.len() + block_end].trim();
            if !addr_text.is_empty() && addr_text.len() > 3 {
                hits.push(Hit {
                    kind: HitKind::Address,
                    text: String::from(addr_text),
                    start: addr_start,
                    end: pos + marker.len() + block_end,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip_code() {
        let mut hits = Vec::new();
        scan_zip("Austin, TX 78701", &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, HitKind::Zip);
        assert_eq!(hits[0].text, "78701");
    }

    #[test]
    fn test_zip_rejects_embedded() {
        let mut hits = Vec::new();
        // 5-digit sequence inside alphanumeric code (e.g., GSTIN) — must not match
        scan_zip("GSTIN T2ABCDEI24322E4", &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_address_detection() {
        let mut hits = Vec::new();
        scan_context("Address: 123 Main St, Austin, TX 78701", &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Address));
        assert!(hits.iter().any(|h| h.kind == HitKind::Zip));
    }

    #[test]
    fn test_name_detection() {
        let mut hits = Vec::new();
        scan_context("Buyer: John Smith, 123 Main St", &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Name));
    }

    #[test]
    fn test_bill_to_extracts_name_and_address() {
        let mut hits = Vec::new();
        let text = "BILL_TO Jason Roberts 645 Claudia Expressway Suite 600 South Anthonyland, OH 99813 US Tel +(414)326-1227";
        scan_context(text, &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Name && h.text == "Jason Roberts"),
            "Expected Name hit for 'Jason Roberts', got: {:?}", hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Address),
            "Expected Address hit, got: {:?}", hits);
    }

    #[test]
    fn test_bill_to_no_colon() {
        let mut hits = Vec::new();
        let text = "Bill to Cindy Banks 0668 Amanda Street Apt. 308 New Jeffrey, MO 07685 US Tel +(610)001-4758";
        scan_context(text, &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Name && h.text == "Cindy Banks"),
            "Expected Name hit for 'Cindy Banks', got: {:?}", hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Address),
            "Expected Address hit, got: {:?}", hits);
    }

    #[test]
    fn test_buyer_no_colon() {
        let mut hits = Vec::new();
        let text = "Buyer Christian Banks III 746 Anderson Shoal Apt. 968 South Sharontown, FM 28464 US Tel +(928)499-5656";
        scan_context(text, &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Name && h.text == "Christian Banks III"),
            "Expected Name hit for 'Christian Banks III', got: {:?}", hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Address),
            "Expected Address hit, got: {:?}", hits);
    }

    #[test]
    fn test_table_buyer() {
        // FATURA2 tag=10: "table Buyer <Name> <Address>..."
        let mut hits = Vec::new();
        let text = "table Buyer Alexander Williams 6479 Smith Causeway East Camerontown, AS 38212 US Tel +(330)644-2313";
        scan_context(text, &mut hits);
        assert!(hits.iter().any(|h| h.kind == HitKind::Name && h.text == "Alexander Williams"),
            "Expected Name hit for 'Alexander Williams', got: {:?}", hits);
    }
}
