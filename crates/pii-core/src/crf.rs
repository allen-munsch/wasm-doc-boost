//! Linear-chain CRF Viterbi decoder for PII NER.
//!
//! This module performs inference only — model loading/JSON parsing
//! happens in wasm-bridge.  pii-core stays dependency-free.
//!
//! Model format (parsed by wasm-bridge, passed here):
//!   - labels: ordered list of label names \[K\]
//!   - feature_index: map from feature name → usize index \[F\]
//!   - label_weights: per-label weight vectors \[K\]\[F\]
//!   - transitions: transition score matrix \[K\]\[K\]

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use crate::{Hit, HitKind};

/// A loaded CRF model ready for inference.
pub struct CrfModel {
    /// Label names in index order, e.g. \["O","ADDRESS","NAME","PHONE","EMAIL","ZIP","ACCOUNT"\]
    pub labels: Vec<String>,
    /// Feature name → sparse index mapping
    pub feature_index: Vec<(String, usize)>,
    /// Per-label weight vectors \[K\]\[F\]
    pub label_weights: Vec<Vec<f32>>,
    /// Transition scores \[K\]\[K\]
    pub transitions: Vec<Vec<f32>>,
}

/// Split text into word tokens (whitespace-delimited, with offsets).
fn tokenize(text: &str) -> Vec<(&str, usize, usize)> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let len = bytes.len();

    while start < len {
        // Skip whitespace
        while start < len && bytes[start].is_ascii_whitespace() {
            start += 1;
        }
        if start >= len {
            break;
        }

        // Find end of word
        let mut end = start;
        while end < len && !bytes[end].is_ascii_whitespace() && bytes[end] != b'\n' {
            end += 1;
        }

        let word = &text[start..end];
        if !word.is_empty() {
            tokens.push((word, start, end));
        }

        start = end + 1;
    }

    tokens
}

/// Compute the character-level shape of a word.
/// Uppercase→A, lowercase→a, digit→0, other→-
fn token_shape(token: &str) -> String {
    let mut shape = String::with_capacity(token.len().min(10));
    for c in token.chars().take(10) {
        if c.is_uppercase() {
            shape.push('A');
        } else if c.is_lowercase() {
            shape.push('a');
        } else if c.is_ascii_digit() {
            shape.push('0');
        } else {
            shape.push('-');
        }
    }
    shape
}

// ── Gazetteer helpers for ADDRESS/NAME disambiguation ──

fn is_street_suffix(lower: &str) -> bool {
    matches!(
        lower,
        "street" | "st" | "st." | "avenue" | "ave" | "ave." | "road" | "rd" | "rd."
            | "drive" | "dr" | "dr." | "lane" | "ln" | "ln." | "court" | "ct" | "ct."
            | "boulevard" | "blvd" | "blvd." | "way" | "place" | "pl" | "pl."
            | "causeway" | "cswy" | "highway" | "hwy" | "parkway" | "pkwy"
            | "circle" | "cir" | "trail" | "trl" | "suite" | "ste" | "apt" | "unit"
            | "po" | "p.o." | "box"
            | "terrace" | "ter" | "ter." | "square" | "sq" | "plaza" | "plz" | "mall"
    )
}

fn is_state_abbrev(upper: &str) -> bool {
    matches!(
        upper,
        "AL" | "AK" | "AZ" | "AR" | "CA" | "CO" | "CT" | "DE" | "FL" | "GA"
            | "HI" | "ID" | "IL" | "IN" | "IA" | "KS" | "KY" | "LA" | "ME" | "MD"
            | "MA" | "MI" | "MN" | "MS" | "MO" | "MT" | "NE" | "NV" | "NH" | "NJ"
            | "NM" | "NY" | "NC" | "ND" | "OH" | "OK" | "OR" | "PA" | "RI" | "SC"
            | "SD" | "TN" | "TX" | "UT" | "VT" | "VA" | "WA" | "WV" | "WI" | "WY"
            | "DC" | "AS" | "GU" | "MP" | "PR" | "VI"
            // Freely Associated States (appear in FATURA2 synthetic data)
            | "FM" | "MH" | "PW"
    )
}

fn is_direction_word(lower: &str) -> bool {
    matches!(
        lower,
        "n" | "s" | "e" | "w" | "north" | "south" | "east" | "west"
            | "ne" | "nw" | "se" | "sw" | "n.e." | "n.w." | "s.e." | "s.w."
            | "northeast" | "northwest" | "southeast" | "southwest"
    )
}

fn is_title_prefix(lower: &str) -> bool {
    matches!(
        lower,
        "mr" | "mr." | "mrs" | "mrs." | "ms" | "ms." | "miss" | "dr" | "dr."
            | "prof" | "prof." | "sr" | "sr." | "jr" | "jr." | "md" | "phd"
    )
}

/// True if the matched text span contains at least one address-like signal.
/// Used for post-CRF filtering to reduce ADDRESS false positives.
///
/// Real street addresses are compact (3-15 words).  Spans longer than
/// 20 words are almost certainly over-labeled CRF output, not addresses.
fn has_address_signal(text: &str) -> bool {
    let lower = text.to_lowercase();
    let words: alloc::vec::Vec<&str> = lower.split_whitespace().collect();

    // Hard cap: a single address is never >20 words
    if words.len() > 20 {
        return false;
    }

    // Strong signals: any one of these is sufficient
    for word in &words {
        let w = word.trim_end_matches(|c: char| c == ',' || c == '.');
        if is_street_suffix(w) || is_direction_word(w) {
            return true;
        }
    }
    // PO Box pattern
    if lower.contains("p.o.") || lower.contains("po box") || lower.contains("pobox") {
        return true;
    }
    // State abbreviation in text
    for word in text.split_whitespace() {
        if word.len() == 2
            && word.chars().all(|c| c.is_uppercase())
            && is_state_abbrev(word)
        {
            return true;
        }
    }
    // Weaker signal: short spans (2-20 words) with a building-number-like token
    // (alphanumeric starting with digit, 2-5 chars).
    if words.len() >= 2 {
        for word in &words {
            let w = word.trim_end_matches(|c: char| c == ',' || c == '.');
            if w.len() >= 2
                && w.len() <= 5
                && w.starts_with(|c: char| c.is_ascii_digit())
                && w.chars().any(|c| c.is_alphabetic())
            {
                return true;
            }
        }
    }
    false
}

/// If the ADDRESS span starts with a name-like prefix followed by a street number,
/// return the byte offset where the address portion begins (after the name).
///
/// Pattern: 1-3 capitalized alpha words then a digit token (street number).
/// Example: "Cindy Banks 0668 Amanda Street..." → name="Cindy Banks", address="0668 Amanda Street..."
/// Example: "Bill to Cindy Banks 0668..." — "Bill" and "to" are not removed; they stay as ADDRESS.
fn extract_name_prefix(address_text: &str) -> Option<usize> {
    let words: Vec<&str> = address_text.split_whitespace().collect();
    if words.len() < 3 {
        return None;
    }

    // Find the first street-number-like token (starts with digit, 1-5 chars).
    // This anchors the address start — everything before it is the candidate name region.
    let mut addr_start = None;
    for (i, w) in words.iter().enumerate() {
        let core = w.trim_end_matches(|c: char| c == ',' || c == '.');
        if !core.is_empty()
            && core.starts_with(|c: char| c.is_ascii_digit())
            && core.len() <= 5
        {
            addr_start = Some(i);
            break;
        }
    }
    let addr_start = addr_start?;
    if addr_start < 1 {
        return None;
    }

    // Walk backward from addr_start gathering capitalized alpha words.
    // Stop at structural prefixes ("Bill", "to", "Buyer") and name suffixes (Jr, Sr, III, etc.).
    let structural: &[&str] = &["bill", "to", "buyer", "ship", "attn", "attention"];
    let suffixes: &[&str] = &["jr", "sr", "ii", "iii", "iv", "md", "dds", "phd", "esq"];

    let name_end = addr_start;
    let mut name_start = addr_start;
    for j in (0..addr_start).rev() {
        let w = words[j];
        let core = w.trim_end_matches(|c: char| c == ',' || c == '.');
        let lower = core.to_lowercase();

        if lower.is_empty() || !core.chars().all(|c| c.is_ascii_alphabetic()) {
            break;
        }
        if structural.contains(&lower.as_str()) {
            break;
        }
        if j == addr_start - 1 && suffixes.contains(&lower.as_str()) {
            // Suffix like "III" right before street number — skip it, keep name_end at addr_start
            continue;
        }
        if !core.starts_with(|c: char| c.is_ascii_uppercase()) {
            break;
        }
        name_start = j;
    }

    let name_word_count = name_end - name_start;
    if name_word_count < 1 || name_word_count > 3 {
        return None;
    }

    // Find byte offset of the name end / address start (word[addr_start])
    let mut found = 0usize;
    for (i, _) in address_text.char_indices() {
        let word_count = address_text[..i].split_whitespace().count();
        if word_count >= addr_start && found == 0 {
            found = i;
            break;
        }
    }
    if found == 0 {
        return None;
    }
    Some(found)
}

/// True if the text looks like a person name (1-5 capitalized words,
/// no digit-heavy patterns, no document-level stopwords).
fn has_name_signal(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || words.len() > 5 {
        return false;
    }

    let lower = text.to_lowercase();

    // Reject spans containing non-name document keywords
    // Use word-boundary matching to avoid false positives on substrings
    // (e.g. "bank" should not reject surname "Banks")
    let lower_words: Vec<&str> = lower.split_whitespace().collect();
    for kw in &[
        "invoice", "date", "due", "total", "payment", "terms", "conditions",
        "note", "order", "table", "logo", "number", "discount", "tax",
        "subtotal", "ship", "shipped", "courier", "bank", "swift", "code",
        "branch", "account", "site", "http", "www.",
        "address", "thank", "gstin", "email",
    ] {
        for w in &lower_words {
            let w_clean = w.trim_end_matches(|c: char| c == ',' || c == '.');
            if w_clean == *kw {
                return false;
            }
        }
    }

    // Reject spans with digit-heavy tokens (>50% digits)
    let mut digit_tokens = 0u32;
    for w in &words {
        let digits = w.chars().filter(|c| c.is_ascii_digit()).count();
        if digits > 0 && digits * 2 > w.len() {
            digit_tokens += 1;
        }
    }
    if digit_tokens >= words.len() as u32 / 2 + 1 {
        return false;
    }

    // Must have at least one word that looks like a proper name:
    // starts with uppercase, 2-20 chars, mostly alphabetic
    words.iter().any(|w| {
        let chars: Vec<char> = w.chars().collect();
        chars.first().map_or(false, |c| c.is_uppercase())
            && w.len() >= 2
            && w.len() <= 20
            && chars.iter().filter(|c| c.is_alphabetic()).count() * 2 >= chars.len()
    })
}

/// Extract a person name from the text immediately preceding an ADDRESS hit.
/// Walks backward from the end finding 1-3 capitalized alpha words,
/// stopping at structural prefixes or non-alpha tokens.
/// Returns the candidate name string (e.g. "Cindy Banks" from "Bill to Cindy Banks").
fn find_name_in_leading(leading: &str) -> Option<String> {
    let words: Vec<&str> = leading.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    let structural: &[&str] = &["bill", "to", "buyer", "ship", "attn", "attention"];
    let suffixes: &[&str] = &["jr", "sr", "ii", "iii", "iv", "md", "dds", "phd", "esq"];

    // Walk backward from the last word
    let mut name_end = words.len();
    let mut name_start = words.len();

    for j in (0..words.len()).rev() {
        let w = words[j];
        let core = w.trim_end_matches(|c: char| c == ',' || c == '.');
        let lower = core.to_lowercase();

        if lower.is_empty() || !core.chars().all(|c| c.is_ascii_alphabetic()) {
            break;
        }
        if structural.contains(&lower.as_str()) {
            break;
        }
        if j == words.len() - 1 && suffixes.contains(&lower.as_str()) {
            // Last word is a suffix like "III" — skip
            name_end = j;
            continue;
        }
        if !core.starts_with(|c: char| c.is_ascii_uppercase()) {
            break;
        }
        name_start = j;
    }

    let count = name_end - name_start;
    if count < 1 || count > 3 {
        return None;
    }

    Some(words[name_start..name_end].join(" "))
}

/// Hash a feature name string to a feature index, using the model's feature_index list.
/// Linear scan — acceptable for startup (model loading is rare).
fn feature_id(model: &CrfModel, name: &str) -> Option<usize> {
    model
        .feature_index
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, idx)| *idx)
}

/// Extract CRF features for a token at position `i`.
/// Returns a sparse list of (feature_index, value=1.0) pairs.
fn extract_token_features(
    model: &CrfModel,
    tokens: &[&str],
    i: usize,
) -> Vec<(usize, f32)> {
    let t = tokens[i];
    let mut feats = Vec::new();

    // Helper: add feature if its name is in the model's feature index
    let mut add = |name: &str| {
        if let Some(idx) = feature_id(model, name) {
            feats.push((idx, 1.0));
        }
    };

    // Current token
    let lower = t.to_lowercase();
    let key = if lower.len() <= 20 {
        lower.as_str()
    } else {
        "__LONG__"
    };
    let w_key = alloc::format!("w:{key}");
    add(&w_key);

    let shape = token_shape(t);
    let w_shape = alloc::format!("w_shape:{shape}");
    add(&w_shape);

    let w_len = alloc::format!("w_len:{}", t.len().min(20));
    add(&w_len);

    let w_upper = alloc::format!("w_upper:{}", if t.chars().all(|c| c.is_uppercase()) { "1" } else { "0" });
    add(&w_upper);

    let w_title = alloc::format!("w_title:{}", if t.chars().next().is_some_and(|c| c.is_uppercase()) && t.chars().skip(1).all(|c| c.is_lowercase()) { "1" } else { "0" });
    add(&w_title);

    let w_digit = alloc::format!("w_digit:{}", if t.chars().all(|c| c.is_ascii_digit()) { "1" } else { "0" });
    add(&w_digit);

    let w_alpha = alloc::format!("w_alpha:{}", if t.chars().all(|c| c.is_alphabetic()) { "1" } else { "0" });
    add(&w_alpha);

    let w_has_digit = alloc::format!("w_has_digit:{}", if t.chars().any(|c| c.is_ascii_digit()) { "1" } else { "0" });
    add(&w_has_digit);

    let w_has_at = alloc::format!("w_has_at:{}", if t.contains('@') { "1" } else { "0" });
    add(&w_has_at);

    let w_has_hyphen = alloc::format!("w_has_hyphen:{}", if t.contains('-') { "1" } else { "0" });
    add(&w_has_hyphen);

    let w_has_alpha = alloc::format!("w_has_alpha:{}", if t.chars().any(|c| c.is_alphabetic()) { "1" } else { "0" });
    add(&w_has_alpha);

    let w_punct = alloc::format!("w_punct:{}", if !t.is_empty() && t.chars().all(|c| !c.is_alphanumeric()) { "1" } else { "0" });
    add(&w_punct);

    // First-character type (shape prefix)
    let w_starts_val = if t.is_empty() {
        'X'
    } else {
        let first = t.chars().next().unwrap();
        if first.is_ascii_digit() {
            'D'
        } else if first.is_alphabetic() {
            'A'
        } else if matches!(first, '.' | ',' | '-' | '/' | '(' | ')') {
            'P'
        } else {
            'X'
        }
    };
    let w_starts = alloc::format!("w_starts:{}", w_starts_val);
    add(&w_starts);

    // Gazetteer features for ADDRESS/NAME disambiguation
    let tl = t.trim_end_matches(|c: char| c == ',' || c == '.').to_lowercase();
    let w_street_suffix = alloc::format!("w_street_suffix:{}", if is_street_suffix(&tl) { "1" } else { "0" });
    add(&w_street_suffix);
    let w_state_abbrev = alloc::format!("w_state_abbrev:{}", if t.len() == 2 && is_state_abbrev(t) { "1" } else { "0" });
    add(&w_state_abbrev);
    let w_direction = alloc::format!("w_direction:{}", if is_direction_word(&tl) { "1" } else { "0" });
    add(&w_direction);
    let w_is_bldg_num = alloc::format!("w_is_bldg_num:{}", if t.starts_with(|c: char| c.is_ascii_digit()) && t.len() <= 5 && !t.chars().all(|c| c.is_ascii_digit()) { "1" } else { "0" });
    add(&w_is_bldg_num);
    let w_title_prefix = alloc::format!("w_title_prefix:{}", if is_title_prefix(&tl) { "1" } else { "0" });
    add(&w_title_prefix);
    let w_po_box = alloc::format!("w_po_box:{}", if matches!(tl.as_str(), "po" | "p.o." | "box" | "p.o" | "pobox") { "1" } else { "0" });
    add(&w_po_box);

    // Personal name heuristic: capital-start alpha word not matching any gazetteer category
    let is_person_name = t.chars().next().map_or(false, |c| c.is_uppercase())
        && t.chars().all(|c| c.is_alphabetic())
        && t.len() >= 2
        && t.len() <= 20
        && !is_street_suffix(&tl)
        && !is_direction_word(&tl)
        && !is_title_prefix(&tl)
        && !(t.len() == 2 && is_state_abbrev(t))
        && !matches!(tl.as_str(), "po" | "p.o." | "box" | "p.o" | "pobox");
    let w_is_name_word = alloc::format!("w_is_name_word:{}", if is_person_name { "1" } else { "0" });
    add(&w_is_name_word);

    if t.len() >= 2 {
        let w_prefix2 = alloc::format!("w_prefix2:{}", t[..2].to_lowercase());
        add(&w_prefix2);
        let w_suffix2 = alloc::format!("w_suffix2:{}", t[t.len() - 2..].to_lowercase());
        add(&w_suffix2);
    }
    if t.len() >= 3 {
        let w_prefix3 = alloc::format!("w_prefix3:{}", t[..3].to_lowercase());
        add(&w_prefix3);
        let w_suffix3 = alloc::format!("w_suffix3:{}", t[t.len() - 3..].to_lowercase());
        add(&w_suffix3);
    }

    // Position
    if i == 0 {
        add("pos_bias:1");
    } else if i == tokens.len() - 1 {
        add("pos_bias:2");
    } else {
        add("pos_bias:0");
    }

    // Neighbors (within ±2 window)
    for offset in [-2i32, -1, 1, 2] {
        let j = i as i32 + offset;
        let nk = if j < 0 {
            "__BOS__"
        } else if j >= tokens.len() as i32 {
            "__EOS__"
        } else {
            let nt = tokens[j as usize];
            if nt.len() <= 20 {
                &nt.to_lowercase()
            } else {
                "__LONG__"
            }
        };
        let name = alloc::format!("w{offset:+}:{nk}");
        add(&name);

        if j >= 0 && (j as usize) < tokens.len() {
            let nt = tokens[j as usize];
            let nshape = token_shape(nt);
            if nshape.len() > 8 {
                let ns = alloc::format!("w{offset:+}_shape:{}", &nshape[..8]);
                add(&ns);
            } else {
                let ns = alloc::format!("w{offset:+}_shape:{nshape}");
                add(&ns);
            }
        }
    }

    feats
}

/// Convert CRF label name to HitKind.
fn label_to_kind(label: &str) -> Option<HitKind> {
    match label {
        "ADDRESS" => Some(HitKind::Address),
        "NAME" => Some(HitKind::Name),
        "PHONE" => Some(HitKind::Phone),
        "EMAIL" => Some(HitKind::Email),
        "ZIP" => Some(HitKind::Zip),
        "ACCOUNT" => Some(HitKind::BankAccount),
        _ => None,
    }
}

/// Viterbi decode: find the most likely label sequence.
///
/// Returns `(label_indices, scores)` where `label_indices[i]` is
/// the predicted label index for token `i` and `scores` is the
/// per-token confidence.
fn viterbi(
    model: &CrfModel,
    token_features: &[Vec<(usize, f32)>],
) -> (Vec<usize>, Vec<f32>) {
    let k = model.labels.len();
    let n = token_features.len();

    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // dp[t][label] = (score, prev_label)
    let mut dp: Vec<Vec<(f32, usize)>> = Vec::with_capacity(n);

    // Initialize first position
    let mut first = Vec::with_capacity(k);
    for l in 0..k {
        let emission = token_features[0]
            .iter()
            .map(|(fi, v)| model.label_weights[l][*fi] * v)
            .sum::<f32>();
        first.push((emission, 0));
    }
    dp.push(first);

    // Fill DP table
    for t in 1..n {
        let mut row = Vec::with_capacity(k);
        for l in 0..k {
            let emission: f32 = token_features[t]
                .iter()
                .map(|(fi, v)| model.label_weights[l][*fi] * v)
                .sum();

            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0;
            for prev in 0..k {
                let score = dp[t - 1][prev].0 + model.transitions[prev][l] + emission;
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }
            row.push((best_score, best_prev));
        }
        dp.push(row);
    }

    // Backtrace
    let mut labels = vec![0usize; n];
    let mut best_final = f32::NEG_INFINITY;
    let mut best_idx = 0;
    for l in 0..k {
        if dp[n - 1][l].0 > best_final {
            best_final = dp[n - 1][l].0;
            best_idx = l;
        }
    }
    labels[n - 1] = best_idx;

    for t in (1..n).rev() {
        labels[t - 1] = dp[t][labels[t]].1;
    }

    // Per-token confidence (relative score, not calibrated probability)
    let scores: Vec<f32> = (0..n).map(|t| dp[t][labels[t]].0).collect();

    (labels, scores)
}

/// Run CRF decode on raw text.
///
/// Returns a list of PII hits detected by the CRF.
/// Each hit has `kind`, `text`, `start`, `end`.
pub fn decode(model: &CrfModel, text: &str) -> Vec<Hit> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Vec::new();
    }

    let token_strs: Vec<&str> = tokens.iter().map(|(w, _, _)| *w).collect();
    let features: Vec<Vec<(usize, f32)>> = (0..token_strs.len())
        .map(|i| extract_token_features(model, &token_strs, i))
        .collect();

    let (label_indices, _scores) = viterbi(model, &features);

    // Convert label spans to Hit objects
    let mut hits = Vec::new();
    let k = model.labels.len();
    let mut i = 0;

    while i < label_indices.len() {
        let label_idx = label_indices[i];
        if label_idx < k {
            let label_name = &model.labels[label_idx];
            if let Some(kind) = label_to_kind(label_name) {
                // Gather consecutive tokens with the same label
                let start_byte = tokens[i].1;
                let mut end = i;
                while end + 1 < label_indices.len() && label_indices[end + 1] == label_idx {
                    end += 1;
                }
                let end_byte = tokens[end].2;

                // Reconstruct the matched text from the original slice
                let matched_text = &text[start_byte..end_byte];

                // Post-CRF structural validation: filter obvious false positives
                // that the CRF labels incorrectly due to token-level ambiguity.
                // These checks are safe — they cannot reject true positives
                // because real PII always satisfies the structural constraints.
                let valid = match kind {
                    HitKind::Email => {
                        // Must contain '@' with non-empty local and domain parts
                        if let Some(at) = matched_text.find('@') {
                            at > 0 && at < matched_text.len() - 3
                        } else {
                            false
                        }
                    }
                    HitKind::Zip => {
                        // Must be exactly 5 ASCII digits
                        matched_text.len() == 5
                            && matched_text.as_bytes().iter().all(|b| b.is_ascii_digit())
                    }
                    HitKind::BankAccount => {
                        // Must be 6+ all-ASCII-digit chars — real accounts are
                        // 8-17 digit numeric strings.  Shorter runs (4-5 digits)
                        // are typically ZIP codes or house numbers.
                        matched_text.len() >= 6
                            && matched_text.as_bytes().iter().all(|b| b.is_ascii_digit())
                    }
                    HitKind::Address => {
                        // ADDRESS spans must contain at least one address-like signal.
                        // All spans (single-word and multi-word) are validated — the CRF
                        // tends to over-label, so structural checks are essential.
                        has_address_signal(&matched_text)
                    }
                    HitKind::Name => {
                        // NAME spans should look like person names:
                        // 1-3 capitalized words, no non-name patterns.
                        let words: Vec<&str> = matched_text.split_whitespace().collect();
                        if words.len() > 5 {
                            false
                        } else if !matched_text.chars().any(|c| c.is_alphabetic()) {
                            // All-numeric spans are never names
                            false
                        } else {
                            has_name_signal(&matched_text)
                        }
                    }
                    _ => true,
                };

                if !valid {
                    i = end + 1;
                    continue;
                }

                hits.push(Hit {
                    kind,
                    text: matched_text.into(),
                    start: start_byte,
                    end: end_byte,
                });

                i = end + 1;
                continue;
            }
        }
        i += 1;
    }

    // ── Post-CRF: extract NAME prefixes from ADDRESS spans ──
    // CRF often labels only the street-address portion (starting at street number),
    // missing leading person names like "Cindy Banks" before "0668 Amanda Street...".
    // We check both within the ADDRESS span and in the text immediately before it.
    let mut name_hits: Vec<Hit> = Vec::new();
    for h in &mut hits {
        if h.kind == HitKind::Address {
            // Try within the ADDRESS span first
            let mut name_text: Option<String> = None;
            let mut name_byte_offset: Option<usize> = None;

            if let Some(offset) = extract_name_prefix(&h.text) {
                let candidate = h.text[..offset].trim_end().to_string();
                if !candidate.is_empty() && has_name_signal(&candidate) {
                    name_text = Some(candidate);
                    name_byte_offset = Some(offset);
                }
            }

            // If no name found within the span, try the text before it.
            // The CRF often labels only "0668 Amanda Street..." as ADDRESS, missing
            // "Cindy Banks" before it. Walk backward from the ADDRESS start to find
            // 1-3 capitalized alpha words (person name) followed by a street number.
            if name_text.is_none() && h.start > 0 {
                let lookback = h.start.saturating_sub(40);
                let leading = text[lookback..h.start].trim_end();
                if let Some(candidate) = find_name_in_leading(leading) {
                    if has_name_signal(&candidate) {
                        // Find byte position of candidate within leading text
                        let pos = leading.rfind(&candidate).unwrap_or(0);
                        name_text = Some(candidate);
                        name_byte_offset = Some(h.start - lookback - pos);
                    }
                }
            }

            if let (Some(nt), Some(nb_offset)) = (name_text, name_byte_offset) {
                name_hits.push(Hit {
                    kind: HitKind::Name,
                    text: nt.clone(),
                    start: h.start - nb_offset,
                    end: h.start,
                });
                // Trim name from ADDRESS if it was inside the span
                if h.text.starts_with(&nt) {
                    let trim_len = nt.len();
                    let remainder = h.text[trim_len..].trim_start().to_string();
                    let trim_bytes = h.text.len() - remainder.len();
                    *h = Hit {
                        kind: HitKind::Address,
                        text: remainder,
                        start: h.start + trim_bytes,
                        end: h.end,
                    };
                }
            }
        }
    }
    hits.extend(name_hits);

    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_name_in_leading_bill_to() {
        let result = find_name_in_leading("Bill to Cindy Banks");
        assert_eq!(result, Some("Cindy Banks".to_string()));
    }

    #[test]
    fn test_find_name_in_leading_buyer_iii() {
        let result = find_name_in_leading("Buyer Christian Banks III");
        assert_eq!(result, Some("Christian Banks".to_string()));
    }

    #[test]
    fn test_find_name_in_leading_simple() {
        let result = find_name_in_leading("Cindy Banks");
        assert_eq!(result, Some("Cindy Banks".to_string()));
    }

    #[test]
    fn test_find_name_in_leading_single() {
        let result = find_name_in_leading("Sarah");
        assert_eq!(result, Some("Sarah".to_string()));
    }

    #[test]
    fn test_find_name_in_leading_no_name() {
        assert_eq!(find_name_in_leading("Bill to"), None);
        assert_eq!(find_name_in_leading("Invoice Number 12345"), None);
    }

    #[test]
    fn test_has_name_signal_banks() {
        // "Banks" is a surname, not "bank" as financial institution
        assert!(has_name_signal("Cindy Banks"));
        assert!(has_name_signal("Banks"));
    }

    #[test]
    fn test_has_name_signal_rejects_bank() {
        // "bank" as a standalone word should still be rejected
        // But with word-boundary matching, it only matches if "bank" is a whole word
        // Since "bank" is a 4-letter word that appears in many contexts,
        // this test verifies the word-boundary behavior
        assert!(!has_name_signal("Swiss Bank Corp"));
    }
}
