// Rust 1.95 introduced collapsible_match for `if` inside match arms.
// The content-stream parsers use this pattern extensively (match on operator
// name, then check `in_text_block && !op.operands.is_empty()`). Collapsing
// these into match guards would hurt readability. Allow crate-wide.
#![allow(clippy::collapsible_match)]

//! Smart PDF detection and text extraction using lopdf
//!
//! # Quick start
//!
//! ```no_run
//! // Full processing (detect + extract + markdown) with defaults
//! let result = pdf_inspector::process_pdf("document.pdf").unwrap();
//! println!("type: {:?}, pages: {}", result.pdf_type, result.page_count);
//! if let Some(md) = &result.markdown {
//!     println!("{md}");
//! }
//!
//! // Fast metadata-only detection (no text extraction)
//! let info = pdf_inspector::detect_pdf("document.pdf").unwrap();
//! println!("type: {:?}, pages: {}", info.pdf_type, info.page_count);
//!
//! // Custom options via builder
//! use pdf_inspector::{PdfOptions, ProcessMode};
//! let result = pdf_inspector::process_pdf_with_options(
//!     "document.pdf",
//!     PdfOptions::new().mode(ProcessMode::Analyze),
//! ).unwrap();
//! ```

pub mod adobe_korea1;
pub mod detector;
pub mod extractor;
pub mod glyph_names;
pub mod process_mode;
pub mod text_utils;
pub mod tounicode;
pub mod types;

pub use detector::{
    detect_pdf_type, detect_pdf_type_mem, detect_pdf_type_mem_with_config,
    detect_pdf_type_with_config, DetectionConfig, PdfType, PdfTypeResult, ScanStrategy,
};
pub use extractor::{
    extract_text, extract_text_with_positions, extract_text_with_positions_mem,
    extract_text_with_positions_pages, extract_text_with_positions_pages_with_password,
};
pub use process_mode::ProcessMode;
pub use types::{PdfLine, PdfRect, TextItem};

use lopdf::Document;
use log::debug;
use std::borrow::Cow;
use std::path::Path;


/// OCR reason emitted when the extracted text layer appears garbled due to
/// broken font decoding or mojibake.
pub const OCR_REASON_SUSPECTED_GARBLED_TEXT: &str = "suspected_garbled_text";

/// OCR reason: the page is a scanned image (a full-page raster / image-only
/// page) with no usable text layer.
pub const OCR_REASON_SCANNED: &str = "scanned";

/// OCR reason: the page has no extractable text and no image to OCR — blank,
/// or content the parser cannot reach.
pub const OCR_REASON_NO_TEXT: &str = "no_text";

/// OCR reason: the page's text is drawn as vector outlines (path operators)
/// rather than real text operators, so it cannot be extracted as characters.
pub const OCR_REASON_VECTOR_TEXT: &str = "vector_text";



// =============================================================================
// PdfClassification
// =============================================================================


/// Lightweight classification result for routing decisions.
#[derive(Debug)]
pub struct PdfClassification {
    /// The detected PDF type.
    pub pdf_type: PdfType,
    /// Total page count.
    pub page_count: u32,
    /// 0-indexed page numbers that need OCR (scanned/image pages).
    pub pages_needing_ocr: Vec<u32>,
    /// Detection confidence score (0.0–1.0).
    pub confidence: f32,
}

/// Classify a PDF from a memory buffer without extracting text.
/// Returns the PDF type and which pages need OCR (~10-50ms).
pub fn classify_pdf_mem(buffer: &[u8]) -> Result<PdfClassification, PdfError> {
    validate_pdf_bytes(buffer)?;
    let (doc, page_count) = load_document_from_mem(buffer)?;
    let detection = detector::detect_from_document(&doc, page_count, &DetectionConfig::default())?;
    Ok(PdfClassification {
        pdf_type: detection.pdf_type,
        page_count,
        // Convert from 1-indexed to 0-indexed for caller convenience
        pages_needing_ocr: detection.pages_needing_ocr.iter().map(|&p| p - 1).collect(),
        confidence: detection.confidence,
    })
}

// =============================================================================
// Document loading helpers
// =============================================================================

pub(crate) fn load_document_from_path<P: AsRef<Path>>(
    path: P,
) -> Result<(Document, u32), PdfError> {
    load_document_from_path_with_password(path, None)
}

/// Load a PDF file, decrypting with `password` if the file is encrypted.
pub(crate) fn load_document_from_path_with_password<P: AsRef<Path>>(
    path: P,
    password: Option<&str>,
) -> Result<(Document, u32), PdfError> {
    let buffer = std::fs::read(&path)?;
    load_document_from_mem_with_password(&buffer, password)
}

/// Load a PDF from a memory buffer.
pub(crate) fn load_document_from_mem(buffer: &[u8]) -> Result<(Document, u32), PdfError> {
    load_document_from_mem_with_password(buffer, None)
}

/// Load a PDF from a memory buffer, decrypting with `password` if encrypted.
pub(crate) fn load_document_from_mem_with_password(
    buffer: &[u8],
    password: Option<&str>,
) -> Result<(Document, u32), PdfError> {
    // Fix malformed struct element names before parsing. Some PDF generators
    // write bare names (/S Code) instead of proper PDF names (/S /Code), which
    // causes lopdf to silently drop the entire object.
    let fixed = fix_bare_struct_names(buffer);
    let buf = fixed.as_ref();

    let doc = match load_document_bytes(buf, password) {
        Ok(doc) => doc,
        Err(first_err) => {
            for repaired in repair_pdf_container_candidates(buf) {
                match load_document_bytes(&repaired, password) {
                    Ok(doc) => {
                        log::debug!("loaded PDF after repairing malformed container bytes");
                        let page_count = doc.get_pages().len() as u32;
                        return Ok((doc, page_count));
                    }
                    Err(e) => {
                        if is_encrypted_lopdf_error(&e) {
                            return Err(e.into());
                        }
                    }
                }
            }
            return Err(first_err.into());
        }
    };
    let page_count = doc.get_pages().len() as u32;
    Ok((doc, page_count))
}

fn load_document_bytes(buf: &[u8], password: Option<&str>) -> Result<Document, lopdf::Error> {
    match Document::load_mem(buf) {
        // Some encrypted PDFs load structurally but leave their streams
        // encrypted (`is_encrypted()` stays true); reading them yields garbage
        // until we re-load with a password. Others fail load_mem outright with
        // an encryption error. Handle both by re-loading with the password.
        Ok(doc) if doc.is_encrypted() => decrypt_document_bytes(buf, password),
        Ok(doc) => Ok(doc),
        Err(ref e) if is_encrypted_lopdf_error(e) => decrypt_document_bytes(buf, password),
        Err(e) => Err(e),
    }
}

/// Re-load an encrypted PDF, decrypting with `password`. Falls back to the
/// empty password (owner-only encryption, the common "protected" case) when a
/// non-empty password was supplied but rejected.
fn decrypt_document_bytes(buf: &[u8], password: Option<&str>) -> Result<Document, lopdf::Error> {
    let pw = password.unwrap_or("");
    match Document::load_mem_with_options(buf, lopdf::LoadOptions::with_password(pw)) {
        Ok(doc) => Ok(doc),
        Err(inner) if !pw.is_empty() => {
            Document::load_mem_with_options(buf, lopdf::LoadOptions::with_password(""))
                .map_err(|_| inner)
        }
        Err(inner) => Err(inner),
    }
}

fn repair_pdf_container_candidates(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut candidates = Vec::new();

    add_repair_candidate(&mut candidates, append_missing_eof_marker(buf), buf);
    add_repair_candidate(&mut candidates, recover_startxref_pointer(buf), buf);

    let stripped = strip_leading_pdf_container_bytes(buf);
    if let Some(stripped_buf) = stripped.as_deref() {
        add_repair_candidate(&mut candidates, Some(stripped_buf.to_vec()), buf);
        add_repair_candidate(
            &mut candidates,
            append_missing_eof_marker(stripped_buf),
            buf,
        );
        add_repair_candidate(
            &mut candidates,
            recover_startxref_pointer(stripped_buf),
            buf,
        );
    }

    candidates
}

/// Some PDF writers emit a `startxref` pointer that doesn't actually point
/// at the cross-reference table — a single corrupted byte in the offset is
/// enough. lopdf trusts that pointer outright and fails to load rather than
/// searching for the real table, unlike pypdf/pdfium which both recover by
/// locating it directly. This finds the real (classic, non-stream) `xref`
/// table by scanning for the keyword — validating that a plausible
/// subsection header follows, not just any standalone "xref" token, since
/// this crate processes untrusted input and a coincidental match inside
/// unrelated stream/string content must not get "repaired" against a bogus
/// offset (lopdf would then load successfully against garbage instead of
/// returning a clean error) — and appends a corrected trailing
/// `startxref`/`%%EOF` block. lopdf's own `get_xref_start` always uses the
/// *last* `%%EOF` in the final 512 bytes of the buffer, so ours
/// transparently supersedes the broken one without needing to touch
/// anything already in the file.
///
/// Doesn't cover cross-reference *streams* (`N 0 obj << /Type /XRef ...`,
/// used by some PDF 1.5+ writers instead of a classic table) — recovering
/// those needs the containing object's number, not just a byte offset.
fn recover_startxref_pointer(buf: &[u8]) -> Option<Vec<u8>> {
    let xref_pos = find_last_valid_xref_table_start(buf)?;

    let mut repaired = Vec::with_capacity(buf.len() + 32);
    repaired.extend_from_slice(buf);
    if !repaired.ends_with(b"\n") {
        repaired.push(b'\n');
    }
    repaired.extend_from_slice(format!("startxref\n{xref_pos}\n%%EOF\n").as_bytes());
    Some(repaired)
}

/// Finds the last standalone `xref` token in `buf` that is immediately
/// followed by a plausible classic cross-reference subsection header
/// (`<start-id> <count>`, e.g. "0 6") — the shape every real classic xref
/// table starts with. A single reverse byte scan: O(n) even on a
/// pathological buffer with many non-matching or non-standalone "xref"
/// occurrences, unlike repeatedly re-searching a shrinking prefix.
fn find_last_valid_xref_table_start(buf: &[u8]) -> Option<usize> {
    const KEYWORD: &[u8] = b"xref";
    if buf.len() < KEYWORD.len() {
        return None;
    }
    let mut pos = buf.len() - KEYWORD.len();
    loop {
        if &buf[pos..pos + KEYWORD.len()] == KEYWORD {
            let before_ok = pos == 0 || buf[pos - 1].is_ascii_whitespace();
            let after_ok = buf
                .get(pos + KEYWORD.len())
                .is_none_or(|c| c.is_ascii_whitespace());
            if before_ok && after_ok && looks_like_xref_subsection_header(buf, pos + KEYWORD.len())
            {
                return Some(pos);
            }
        }
        if pos == 0 {
            return None;
        }
        pos -= 1;
    }
}

/// Checks that `buf[pos..]` starts (after whitespace) with two
/// whitespace-separated runs of ASCII digits — `<start-id> <count>`, the
/// first subsection header of a classic PDF cross-reference table.
fn looks_like_xref_subsection_header(buf: &[u8], pos: usize) -> bool {
    fn skip_ws(buf: &[u8], mut pos: usize) -> usize {
        while buf.get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        pos
    }
    fn skip_digits(buf: &[u8], mut pos: usize) -> usize {
        while buf.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        pos
    }

    let pos = skip_ws(buf, pos);
    let after_first_digits = skip_digits(buf, pos);
    if after_first_digits == pos {
        return false; // no start-id
    }
    let sep = skip_ws(buf, after_first_digits);
    if sep == after_first_digits {
        return false; // start-id and count must be whitespace-separated
    }
    let after_count = skip_digits(buf, sep);
    if after_count == sep {
        return false; // no count
    }
    // The count run must end at whitespace/buffer-end, not run into trailing
    // garbage (e.g. a coincidental "xref\n0 6garbage" in stream content).
    buf.get(after_count).is_none_or(u8::is_ascii_whitespace)
}

fn add_repair_candidate(
    candidates: &mut Vec<Vec<u8>>,
    candidate: Option<Vec<u8>>,
    original: &[u8],
) {
    let Some(candidate) = candidate else {
        return;
    };
    if candidate.as_slice() == original {
        return;
    }
    if candidates.iter().any(|existing| existing == &candidate) {
        return;
    }
    candidates.push(candidate);
}

fn append_missing_eof_marker(buf: &[u8]) -> Option<Vec<u8>> {
    if contains_recent_eof_marker(buf) {
        return None;
    }

    let mut end = buf.len();
    while end > 0 && buf[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    if !buf[..end].ends_with(b"%%EO") {
        return None;
    }

    let mut repaired = Vec::with_capacity(end + 2);
    repaired.extend_from_slice(&buf[..end]);
    repaired.extend_from_slice(b"F\n");
    Some(repaired)
}

fn contains_recent_eof_marker(buf: &[u8]) -> bool {
    let start = buf.len().saturating_sub(1024);
    buf[start..].windows(b"%%EOF".len()).any(|w| w == b"%%EOF")
}

fn strip_leading_pdf_container_bytes(buf: &[u8]) -> Option<Vec<u8>> {
    let mut start = if buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    };

    while start < buf.len() && buf[start].is_ascii_whitespace() {
        start += 1;
    }

    if start > 0 && buf[start..].starts_with(b"%PDF-") {
        Some(buf[start..].to_vec())
    } else {
        None
    }
}

// =============================================================================
// fix_bare_struct_names (vendored from structure_tree.rs)
// =============================================================================

pub fn fix_bare_struct_names(buf: &[u8]) -> Cow<'_, [u8]> {
    // Quick check: if no StructTreeRoot, nothing to fix
    if !contains_bytes(buf, b"/StructTreeRoot") {
        return Cow::Borrowed(buf);
    }

    // Known struct type names that may appear as bare tokens.
    // We only fix names that are valid PDF structure types to avoid
    // false positives on arbitrary dictionary values.
    const KNOWN_NAMES: &[&[u8]] = &[
        b"Document",
        b"Part",
        b"Art",
        b"Sect",
        b"Div",
        b"BlockQuote",
        b"Caption",
        b"TOC",
        b"TOCI",
        b"Index",
        b"NonStruct",
        b"Private",
        b"H",
        b"H1",
        b"H2",
        b"H3",
        b"H4",
        b"H5",
        b"H6",
        b"P",
        b"L",
        b"LI",
        b"Lbl",
        b"LBody",
        b"Table",
        b"TR",
        b"TH",
        b"TD",
        b"THead",
        b"TBody",
        b"TFoot",
        b"Span",
        b"Quote",
        b"Note",
        b"Reference",
        b"BibEntry",
        b"Code",
        b"Link",
        b"Annot",
        b"Figure",
        b"Formula",
        b"Form",
        b"Ruby",
        b"RB",
        b"RT",
        b"RP",
        b"Warichu",
        b"WT",
        b"WP",
    ];

    let pattern = b"/S ";
    let mut result: Option<Vec<u8>> = None;
    let mut pos = 0;

    while pos + pattern.len() < buf.len() {
        let Some(idx) = find_bytes(&buf[pos..], pattern).map(|i| i + pos) else {
            break;
        };

        let after = idx + pattern.len();
        // Check if the next char is already '/' (correct name) or not
        if after < buf.len() && buf[after] == b'/' {
            pos = after;
            continue;
        }

        // Try to match a known bare struct name at this position
        let mut matched = false;
        for name in KNOWN_NAMES {
            let end = after + name.len();
            if end <= buf.len()
                && &buf[after..end] == *name
                // Must be followed by a delimiter (whitespace, newline, /, >)
                && (end >= buf.len() || matches!(buf[end], b'\n' | b'\r' | b' ' | b'/' | b'>'))
            {
                // Found a bare name — lazily allocate output buffer
                let out = result.get_or_insert_with(|| buf[..after].to_vec());
                // Append everything from last position up to the bare name
                if out.len() < after {
                    out.extend_from_slice(&buf[out.len()..after]);
                }
                out.push(b'/');
                out.extend_from_slice(name);
                pos = end;
                matched = true;
                debug!(
                    "fix_bare_struct_names: patched /S {} → /S /{}",
                    String::from_utf8_lossy(name),
                    String::from_utf8_lossy(name)
                );
                break;
            }
        }

        if !matched {
            pos = after;
        }
    }

    match result {
        Some(mut out) => {
            // Append remaining bytes
            if out.len() < buf.len() {
                out.extend_from_slice(&buf[out.len()..]);
            }
            Cow::Owned(out)
        }
        None => Cow::Borrowed(buf),
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    find_bytes(haystack, needle).is_some()
}


/// Detect markdown tables with suspicious structure that suggest the heuristic
/// missed/mangled rows or columns. Returns true when the caller should treat
/// the result as `needs_ocr` and fall back to GPU OCR.
///
/// Catches three failure modes observed in production:
///
/// 1. **Header row looks like a data row** — first row starts with a numeric
///    value (e.g. `|2|...`), suggesting we missed the actual header above it.
///    Real headers almost never start with a bare number.
///
/// 2. **Header has empty cells in a multi-column table** — e.g.
///    `|Position||Administration|Administration|` (3+ cols, ≥1 empty cell).
///    Indicates poor column boundary detection.
///

// =============================================================================
// PdfError
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF parsing error: {0}")]
    Parse(String),
    #[error("PDF is encrypted")]
    Encrypted,
    #[error("Invalid PDF structure")]
    InvalidStructure,
    #[error("Not a PDF: {0}")]
    NotAPdf(String),
}

impl From<lopdf::Error> for PdfError {
    fn from(e: lopdf::Error) -> Self {
        match e {
            lopdf::Error::IO(io_err) => PdfError::Io(io_err),
            lopdf::Error::Decryption(_)
            | lopdf::Error::InvalidPassword
            | lopdf::Error::AlreadyEncrypted
            | lopdf::Error::UnsupportedSecurityHandler(_) => PdfError::Encrypted,
            lopdf::Error::Unimplemented(msg) if msg.contains("encrypted") => PdfError::Encrypted,
            lopdf::Error::Parse(ref pe) if pe.to_string().contains("invalid file header") => {
                PdfError::NotAPdf("invalid PDF file header".to_string())
            }
            lopdf::Error::MissingXrefEntry
            | lopdf::Error::Xref(_)
            | lopdf::Error::IndirectObject { .. }
            | lopdf::Error::ObjectIdMismatch
            | lopdf::Error::InvalidObjectStream(_)
            | lopdf::Error::InvalidOffset(_) => PdfError::InvalidStructure,
            other => PdfError::Parse(other.to_string()),
        }
    }
}

/// Check whether a `lopdf::Error` represents an encryption-related failure.
pub(crate) fn is_encrypted_lopdf_error(e: &lopdf::Error) -> bool {
    matches!(
        e,
        lopdf::Error::Decryption(_)
            | lopdf::Error::InvalidPassword
            | lopdf::Error::AlreadyEncrypted
            | lopdf::Error::UnsupportedSecurityHandler(_)
    ) || matches!(e, lopdf::Error::Unimplemented(msg) if msg.contains("encrypted"))
}

// ---------------------------------------------------------------------------
// PDF validation helpers
// ---------------------------------------------------------------------------

/// Strip UTF-8 BOM and leading ASCII whitespace from a byte slice.
fn strip_bom_and_whitespace(bytes: &[u8]) -> &[u8] {
    let b = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    let start = b
        .iter()
        .position(|&c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    &b[start..]
}

/// Case-insensitive prefix check on byte slices.
fn starts_with_ci(haystack: &[u8], needle: &[u8]) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    haystack[..needle.len()]
        .iter()
        .zip(needle)
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Try to identify what kind of file the bytes represent.
fn detect_file_type_hint(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "file is empty".to_string();
    }

    let trimmed = strip_bom_and_whitespace(bytes);

    // HTML
    if starts_with_ci(trimmed, b"<!doctype html")
        || starts_with_ci(trimmed, b"<html")
        || starts_with_ci(trimmed, b"<head")
        || starts_with_ci(trimmed, b"<body")
    {
        return "file appears to be HTML".to_string();
    }

    // XML (but not HTML)
    if trimmed.starts_with(b"<?xml") || trimmed.starts_with(b"<") {
        if starts_with_ci(trimmed, b"<?xml") {
            return "file appears to be XML".to_string();
        }
        if trimmed.starts_with(b"<") && !trimmed.starts_with(b"<%") {
            return "file appears to be XML".to_string();
        }
    }

    // JSON
    if trimmed.starts_with(b"{") || trimmed.starts_with(b"[") {
        return "file appears to be JSON".to_string();
    }

    // PNG
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return "file appears to be a PNG image".to_string();
    }

    // JPEG
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "file appears to be a JPEG image".to_string();
    }

    // ZIP / Office documents
    if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return "file appears to be a ZIP archive (possibly an Office document)".to_string();
    }

    // If it looks like mostly printable ASCII/UTF-8, call it plain text
    let sample = &bytes[..bytes.len().min(512)];
    let printable = sample
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count();
    if printable > sample.len() * 3 / 4 {
        return "file appears to be plain text".to_string();
    }

    "file is not a PDF".to_string()
}

/// Validate that a byte buffer looks like a PDF (has `%PDF-` magic).
///
/// Scans the first 1024 bytes, allowing for a UTF-8 BOM and leading whitespace.
pub(crate) fn validate_pdf_bytes(buffer: &[u8]) -> Result<(), PdfError> {
    if buffer.is_empty() {
        return Err(PdfError::NotAPdf(detect_file_type_hint(buffer)));
    }

    let header = &buffer[..buffer.len().min(1024)];
    let trimmed = strip_bom_and_whitespace(header);

    if trimmed.starts_with(b"%PDF-") {
        Ok(())
    } else {
        Err(PdfError::NotAPdf(detect_file_type_hint(buffer)))
    }
}

/// Validate that a file on disk looks like a PDF.
///
/// Reads only the first 1024 bytes and delegates to [`validate_pdf_bytes`].
pub(crate) fn validate_pdf_file<P: AsRef<Path>>(path: P) -> Result<(), PdfError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 1024];
    let n = file.read(&mut buf)?;
    validate_pdf_bytes(&buf[..n])
}
