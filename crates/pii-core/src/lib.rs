#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

mod context;
pub mod crf;
mod email;
mod luhn;
mod phone;
mod ssn;

/// Categories of personally identifiable / payment card information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HitKind {
    /// Credit/debit card Primary Account Number (validated by Luhn + IIN prefix).
    /// 13-19 digit number that passes the Luhn checksum.
    Pan = 0,
    /// Card Verification Value: 3-4 digits found near a validated PAN.
    Cvv = 1,
    /// Card expiry date: MM/YY or MM/YYYY near a validated PAN.
    Expiry = 2,
    /// US Social Security Number: 9 digits (with or without dashes/spaces),
    /// validated by area/group/serial rules.
    Ssn = 3,
    /// Email address: local@domain.tld.
    Email = 4,
    /// Phone number: NANP or E.164 digit patterns.
    Phone = 5,
    /// US bank routing number: 9 digits matching ABA prefix ranges.
    RoutingNumber = 6,
    /// Bank account number: 4-17 digits near a routing number.
    BankAccount = 7,
    /// Date of birth: DD/MM/YYYY, MM/DD/YYYY, or "Date of Birth" context.
    Dob = 8,
    /// Street address: "Address:" prefix, street suffix + ZIP pattern.
    Address = 9,
    /// Personal or business name: "Buyer:", "Name:", "Customer:" prefix.
    Name = 10,
    /// US 5-digit ZIP code (isolated, not part of a longer number).
    Zip = 11,
}

/// A detected PII/PCI hit within the scanned text.
#[derive(Debug, Clone)]
pub struct Hit {
    /// What kind of sensitive data was found.
    pub kind: HitKind,
    /// The matched text (characters from the original input).
    pub text: String,
    /// Byte offset where the match starts in the original text.
    pub start: usize,
    /// Byte offset where the match ends (exclusive) in the original text.
    pub end: usize,
}

/// Global CRF model, set once at initialization.
/// Not Mutex — WASM is single-threaded, and the model is set before any scan calls.
static mut CRF_MODEL: Option<crf::CrfModel> = None;

/// Set the CRF model for NER-based PII detection.
///
/// # Safety
///
/// Must be called once before any concurrent `scan` calls.
/// In WASM (single-threaded), this is safe as long as
/// `load_crf_model` is called before `scan_pii`.
pub fn set_crf_model(model: crf::CrfModel) {
    // SAFETY: caller guarantees single-threaded initialization
    unsafe {
        CRF_MODEL = Some(model);
    }
}

/// Scan text for all known categories of PII/PCI.
///
/// Returns every detected hit, ordered by start position.
/// SSN and PAN are mutually exclusive — a digit run is only classified
/// as SSN if it doesn't exceed 9 digits (PAN is 13-19).
///
/// If a CRF model was loaded via `set_crf_model`, its NER predictions
/// are ensemble-merged with the rule-based scanner results.
pub fn scan(text: &str) -> Vec<Hit> {
    let mut hits = Vec::new();

    // Run all scanners
    email::scan_email(text, &mut hits);
    phone::scan_phone(text, &mut hits);
    ssn::scan_ssn(text, &mut hits);
    luhn::scan_pan(text, &mut hits);
    context::scan_context(text, &mut hits);

    // CRF NER (if model loaded)
    // SAFETY: model is set once at startup, read-only thereafter
    let crf_model_ptr: *const Option<crf::CrfModel> = &raw const CRF_MODEL;
    let crf_hits = unsafe {
        if let Some(ref model) = *crf_model_ptr {
            let mut hits = crf::decode(model, text);
            // CRF handles ADDRESS, NAME, ACCOUNT — regex scanners handle PHONE, EMAIL, ZIP
            hits.retain(|h| matches!(h.kind, HitKind::Address | HitKind::Name | HitKind::BankAccount));
            hits
        } else {
            Vec::new()
        }
    };
    hits.extend(crf_hits);

    // Sort by start position
    hits.sort_by_key(|h| h.start);

    // Deduplicate overlapping hits of the SAME kind (keep the longer match).
    // Different kinds that overlap are kept — they detect different PII types
    // in the same region (e.g., ZIP code inside an address block).
    let mut deduped: Vec<Hit> = Vec::new();
    for hit in hits {
        if let Some(last) = deduped.last_mut() {
            if hit.start < last.end {
                // Same kind: keep the one with more text
                if hit.kind == last.kind {
                    if hit.text.len() > last.text.len() {
                        *last = hit;
                    }
                    continue;
                }
                // PAN vs SSN: PAN wins
                if (hit.kind == HitKind::Pan && last.kind == HitKind::Ssn)
                    || (last.kind == HitKind::Pan && hit.kind == HitKind::Ssn)
                {
                    if hit.kind == HitKind::Pan {
                        *last = hit;
                    }
                    continue;
                }
                // Different kinds: keep both (ZIP inside Address, Phone in Name block, etc.)
                deduped.push(hit);
                continue;
            }
        }
        deduped.push(hit);
    }

    deduped
}
