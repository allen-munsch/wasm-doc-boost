//! gbdt-core — the model forest + rotation combine shared by wasm-bridge
//! (browser) and py-features (backend). One implementation, bit-identical
//! predictions on both sides.
//!
//! The crate has no image or platform dependencies: it only scores feature
//! vectors. Feature extraction stays in `features-core`, and each binding
//! (wasm/py) does the decode -> resize -> extract -> predict -> combine
//! pipeline itself.

use serde::de::Error as _;
use serde::Deserialize;

/// Per-degree rework (v3) labels: 5 doc traits + flip + trust.
pub const LABELS: [&str; 7] = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "flip",
    "trust",
];

/// Index of the deterministic radon axis (radians, [0, pi)) in extract_all_v2.
pub const RADON_AXIS_INDEX: usize = 100;

/// Textline flip-evidence feature indices in extract_all_v2. The flip head can
/// be wrong with high confidence, but these three are deterministic and nearly
/// perfect on text with enough lines: a positive flip_score means upright,
/// negative means 180-flipped.
pub const FLIP_SCORE_INDEX: usize = 104;
pub const VOTE_FRAC_INDEX: usize = 105;
pub const N_LINES_INDEX: usize = 106;
pub const BODY_FRAC_INDEX: usize = 109;

/// Minimum number of valid text lines and minimum |flip_score| before the
/// deterministic textline evidence overrides the GBDT flip head at a cardinal
/// (0/90/180/270) axis. FATURA2 upright runs ~16 lines and |score| ~0.35-0.4.
pub const FLIP_NLINES_MIN: f64 = 5.0;
pub const FLIP_SCORE_MIN: f64 = 0.05;
/// Half-width of the vote_frac tie band: inside |vote_frac - 0.5| <= this the
/// line-vote evidence is ambiguous and the gate defers to the GBDT flip head.
pub const VOTE_TIE_EPS: f64 = 0.05;

/// Line-vote margin -> flip confidence, monotonic in |vote_frac - 0.5|.
/// A unanimous vote (margin 0.5) is 1.0; the tie-band edge (margin =
/// VOTE_TIE_EPS) is 0.0. Recalibrated on the calibration split (step 5).
#[inline]
fn vote_conf(margin: f64) -> f64 {
    ((margin - VOTE_TIE_EPS) / (0.5 - VOTE_TIE_EPS)).clamp(0.0, 1.0)
}

/// Per-glyph flip evidence passed from features-core into the combine step.
/// `sum_logodds` is the sum of per-glyph logistic log-odds (positive =
/// flipped, negative = upright); `count` is the number of glyphs that voted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphEvidence {
    pub sum_logodds: f64,
    pub count: f64,
}

/// Minimum glyph count before the per-glyph shape evidence overrides the
/// line-vote flip gate at a cardinal axis. Below this the summed log-odds is
/// too noisy to trust.
pub const GLYPH_COUNT_MIN: f64 = 8.0;

/// Page-level calibration logistic: P(flipped) = sigmoid(b0 + b1*L + b2*N)
/// with L = sum of per-glyph log-odds and N = glyph count. Fitted on the
/// fatura2 calibration split by scripts/train_glyph_classifier.py.
pub const GLYPH_CAL_B0: f64 = -0.00890701834;
pub const GLYPH_CAL_B1: f64 = 0.374395665;
pub const GLYPH_CAL_B2: f64 = 0.0126013738;

/// Fallback operating point if the manifest does not carry one. The shipped
/// value is pinned in model.json's manifest by tau calibration.
pub const DEFAULT_TAU: f64 = 0.90;

/// Combine the deterministic radon axis with the GBDT flip/trust heads.
///
/// Mirrors `train_model_v2.py` `rotation_metrics` exactly:
///   axis_deg       = degrees(features[100]) mod 180
///   rotation_deg   = (180 - axis_deg) mod 180 + 180 * (flip >= 0.5)
///   confidence     = trust * max(flip, 1 - flip)
///   needs_review   = confidence < tau  OR  not is_document
///
/// Fold-snap: the radon axis is mod-180 blind, so for exactly-horizontal text
/// the argmax lands on bin 0 (~0.01 deg) or bin 59 (~179.99 deg). Both are the
/// SAME horizontal text; snap the fold region to 0 before combining, matching
/// features-core/src/textline.rs.
///
/// The decision also carries the aggregation's counts and agreements so the
/// caller can see how much evidence voted and how concentrated it was:
///   flip_vote_count     = n_lines (text lines that voted on flip)
///   flip_vote_agreement = winning-side line-vote fraction, [0.5, 1.0]
///   axis_vote_count     = independent axis estimates available
///   axis_agreement      = fraction of those estimates agreeing with the chosen
///                         radon axis (boundary long-edge when visible; docstrum
///                         link direction rotated 90deg when concentrated)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationDecision {
    pub rotation_deg: f64,
    pub confidence: f64,
    pub needs_review: bool,
    pub flip_vote_count: f64,
    pub flip_vote_agreement: f64,
    pub axis_vote_count: f64,
    pub axis_agreement: f64,
    pub glyph_count: f64,
    pub glyph_logodds: f64,
}

/// Feature indices for the axis-agreement diagnostic (see combine_rotation).
pub const BOUNDARY_LONG_EDGE_INDEX: usize = 110;
pub const BOUNDARY_VISIBLE_INDEX: usize = 113;
pub const DOCSTRUM_MEAN_INDEX: usize = 114;
pub const DOCSTRUM_CONC_INDEX: usize = 115;

/// Axis estimates within this many radians of the chosen axis count as
/// agreeing (10 degrees).
pub const AXIS_AGREE_TOLERANCE_RAD: f64 = 0.174_532_925;

/// Minimum docstrum concentration for its mean angle to be a usable axis
/// estimate.
pub const DOCSTRUM_CONCENTRATION_MIN: f64 = 0.5;

/// Circular distance between two mod-pi axis angles, in [0, pi/2].
#[inline]
fn axis_err_rad(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(core::f64::consts::PI);
    if d > core::f64::consts::FRAC_PI_2 {
        core::f64::consts::PI - d
    } else {
        d
    }
}

/// Number of points in the confidence calibration lookup table (21 bins on
/// the [0,1] grid: index 0 = confidence 0.0, index 20 = confidence 1.0).
pub const CALIBRATION_GRID: usize = 21;

/// Apply an isotonic calibration curve (21 values on the [0,1] grid) to a raw
/// confidence with linear interpolation. Identity when the curve is absent or
/// malformed.
pub fn calibrate_confidence(raw: f64, curve: &[f64]) -> f64 {
    if curve.len() != CALIBRATION_GRID || raw.is_nan() {
        return raw;
    }
    let x = raw.clamp(0.0, 1.0) * (CALIBRATION_GRID - 1) as f64;
    let i = x.floor() as usize;
    let j = (i + 1).min(CALIBRATION_GRID - 1);
    let t = x - i as f64;
    curve[i] * (1.0 - t) + curve[j] * t
}

/// Line-vote flip gate (v5): deterministic asc/desc line votes at a cardinal
/// axis. Kept as a helper so the glyph gate can fall back to it when too few
/// glyphs voted.
fn vote_flip_gate(features: &[f64], flip_p: f64) -> (f64, f64) {
    let n_lines = features.get(N_LINES_INDEX).copied().unwrap_or(0.0);
    let vote_frac = features.get(VOTE_FRAC_INDEX).copied().unwrap_or(0.5);
    if n_lines >= FLIP_NLINES_MIN {
        let margin = (vote_frac - 0.5).abs();
        if margin > VOTE_TIE_EPS {
            let flip = if vote_frac < 0.5 { 1.0 } else { 0.0 };
            (flip, vote_conf(margin))
        } else {
            // Tie band: line votes are ambiguous; defer to the (retrained)
            // flip head and report its own confidence.
            let f = if flip_p >= 0.5 { 1.0 } else { 0.0 };
            (f, flip_p.max(1.0 - flip_p))
        }
    } else {
        let f = if flip_p >= 0.5 { 1.0 } else { 0.0 };
        (f, flip_p.max(1.0 - flip_p))
    }
}

/// Per-glyph flip gate: sum of per-glyph logistic log-odds, calibrated to a
/// page-level P(flipped). Confidence is the probability the flip decision is
/// correct, max(P, 1-P) in [0.5, 1] — already calibrated, so the generic
/// isotonic confidence curve is NOT applied on top of it.
fn glyph_flip_gate(ev: &GlyphEvidence) -> (f64, f64) {
    let z = GLYPH_CAL_B0 + GLYPH_CAL_B1 * ev.sum_logodds + GLYPH_CAL_B2 * ev.count;
    let p_flip = sigmoid(z);
    let flip = if p_flip >= 0.5 { 1.0 } else { 0.0 };
    let conf = p_flip.max(1.0 - p_flip);
    (flip, conf)
}

pub fn combine_rotation(features: &[f64], preds: &[f64], tau: f64) -> RotationDecision {
    combine_rotation_calibrated(features, preds, tau, None, None)
}

pub fn combine_rotation_calibrated(
    features: &[f64],
    preds: &[f64],
    tau: f64,
    calibration: Option<&[f64]>,
    glyph: Option<GlyphEvidence>,
) -> RotationDecision {
    let boundary_visible = features.get(BOUNDARY_VISIBLE_INDEX).copied().unwrap_or(0.0);
    let boundary_axis = features
        .get(BOUNDARY_LONG_EDGE_INDEX)
        .copied()
        .unwrap_or_else(|| features[RADON_AXIS_INDEX]);
    let docstrum_conc = features.get(DOCSTRUM_CONC_INDEX).copied().unwrap_or(0.0);
    let docstrum_axis = features
        .get(DOCSTRUM_MEAN_INDEX)
        .map(|m| (m + core::f64::consts::FRAC_PI_2).rem_euclid(core::f64::consts::PI))
        .unwrap_or_else(|| features[RADON_AXIS_INDEX]);

    let mut axis_rad = features[RADON_AXIS_INDEX];
    // Independent axis consensus: the boundary minAreaRect and the docstrum
    // inter-row direction are near-perfect on bounded invoices, while the
    // radon response can lock onto a diagonal graphic (~50 deg off) or carry
    // a few degrees of fold bias. When both independent votes agree, use them
    // to correct the radon axis (previously they were only reported as an
    // agreement count, never used to fix the answer).
    if boundary_visible >= 0.5
        && docstrum_conc >= DOCSTRUM_CONCENTRATION_MIN
        && axis_err_rad(boundary_axis, docstrum_axis) <= AXIS_AGREE_TOLERANCE_RAD
    {
        let consensus = boundary_axis;
        if axis_err_rad(consensus, axis_rad) > AXIS_AGREE_TOLERANCE_RAD {
            axis_rad = consensus;
        } else {
            let cdeg = consensus.to_degrees().rem_euclid(180.0);
            if cdeg < 1.0 || cdeg > 179.0 || (cdeg - 90.0).abs() < 1.0 {
                axis_rad = consensus;
            }
        }
    }

    let mut axis_deg = axis_rad.to_degrees().rem_euclid(180.0);
    if axis_deg < 1.0 || axis_deg > 179.0 {
        axis_deg = 0.0;
    }
    let flip_p = preds[5];
    let trust_p = preds[6];

    // v5 flip gate: at a cardinal axis the deterministic textline evidence is
    // primary — it is the one signal independent of the GBDT flip head. The
    // per-glyph shape evidence overrides the line-vote gate when enough
    // glyphs voted: it is the only channel that works for all-caps/digit
    // invoices whose asc/desc ink mass is ~0 (vote_frac is a coin flip there).
    // Both gates derive confidence from evidence strength, never a constant
    // 1.0, so weak evidence lowers confidence and needs_review fires instead
    // of reporting a confident wrong answer.
    let cardinal = axis_deg == 0.0 || (axis_deg - 90.0).abs() < 1.0;
    let glyph_gate_active = cardinal && matches!(glyph, Some(ev) if ev.count >= GLYPH_COUNT_MIN);
    let (flip, flip_conf) = if cardinal {
        match glyph {
            Some(ev) if ev.count >= GLYPH_COUNT_MIN => glyph_flip_gate(&ev),
            _ => vote_flip_gate(features, flip_p),
        }
    } else {
        let f = if flip_p >= 0.5 { 1.0 } else { 0.0 };
        (f, flip_p.max(1.0 - flip_p))
    };

    let rotation_deg = (180.0 - axis_deg).rem_euclid(180.0) + 180.0 * flip;
    let raw_confidence = trust_p * flip_conf;
    let confidence = match calibration {
        Some(curve) => calibrate_confidence(raw_confidence, curve),
        None => raw_confidence,
    };
    // Cross-check: the per-glyph shape gate and the GBDT flip head are
    // independent flip signals. On the fatura2 gold split they agree 999/1000;
    // the single disagreement is the single glyph-gate miss, which the GBDT
    // head gets right. Route any glyph/GBDT disagreement to review rather than
    // trusting the weaker per-glyph signal alone.
    let gbdt_flip = if flip_p >= 0.5 { 1.0 } else { 0.0 };
    let glyph_gbdt_disagrees = glyph_gate_active && flip != gbdt_flip;
    let needs_review = confidence < tau || preds[0] < 0.5 || glyph_gbdt_disagrees;

    // Flip-vote counts and agreement: n_lines is the vote count; the winning
    // side's fraction is the agreement, so a tie is 0.5 and unanimity is 1.0.
    let flip_vote_count = features.get(N_LINES_INDEX).copied().unwrap_or(0.0);
    let vote_frac = features.get(VOTE_FRAC_INDEX).copied().unwrap_or(0.5);
    let flip_vote_agreement = vote_frac.max(1.0 - vote_frac);

    // Axis agreement: count the independent axis estimates available and how
    // many agree with the chosen radon axis. Radon itself always counts; the
    // boundary long-edge is a second estimate when the page boundary is
    // visible; the docstrum inter-row link direction (rotated 90deg to the
    // text axis) is a third when it is concentrated enough to be meaningful.
    let mut axis_vote_count = 1.0f64;
    let mut axis_agreement = 1.0f64;
    if boundary_visible >= 0.5 {
        axis_vote_count += 1.0;
        if axis_err_rad(boundary_axis, axis_rad) <= AXIS_AGREE_TOLERANCE_RAD {
            axis_agreement += 1.0;
        }
    }
    if docstrum_conc >= DOCSTRUM_CONCENTRATION_MIN {
        axis_vote_count += 1.0;
        if axis_err_rad(docstrum_axis, axis_rad) <= AXIS_AGREE_TOLERANCE_RAD {
            axis_agreement += 1.0;
        }
    }

    let (glyph_count, glyph_logodds) = match glyph {
        Some(ev) => (ev.count, ev.sum_logodds),
        None => (0.0, 0.0),
    };

    RotationDecision {
        rotation_deg,
        confidence,
        needs_review,
        flip_vote_count,
        flip_vote_agreement,
        axis_vote_count,
        axis_agreement: axis_agreement / axis_vote_count,
        glyph_count,
        glyph_logodds,
    }
}

/// A single node in a GBDT tree (XGBoost JSON dump format).
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum RawNode {
    Split(Box<RawSplitNode>),
    Leaf(RawLeafNode),
}

#[derive(Debug, Deserialize, Clone)]
struct RawSplitNode {
    split: String, // e.g. "f0"
    split_condition: f64,
    #[serde(default)]
    children: Vec<RawNode>,
}

#[derive(Debug, Deserialize, Clone)]
struct RawLeafNode {
    leaf: f64,
}

/// Internal flat representation of a tree. Nodes are stored breadth-first-ish
/// but traversed via indices.
#[derive(Clone, Debug)]
struct Tree {
    /// For each node: if Some(feature_index, threshold), it's a split node
    /// with left = 2*idx+1, right = 2*idx+2. If None, it's a leaf with the given value.
    nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
enum Node {
    Split { feature: usize, threshold: f64 },
    Leaf { value: f64 },
}

/// A trained model: `trees_per_label[i]` is the list of trees for label `i`.
#[derive(Clone, Debug)]
pub struct Model {
    trees_per_label: Vec<Vec<Tree>>,
}

/// A model manifest: label names + the pinned operating point + the forest.
///
/// This is the v3 `model.json` shape written by `train_model_v2.py`:
/// `{"labels": [...], "tau": 0.9, "version": "0.4.0", "trees": [[...], ...], "trees_per_label": [...]}`.
/// `version` is the release tag this model was built for (a semver string),
/// not an internal format counter. The legacy flat tree array is also
/// accepted for backward compatibility.
#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub labels: Vec<String>,
    pub tau: f64,
    pub version: Option<String>,
    pub confidence_calibration: Option<Vec<f64>>,
    /// Per-label decision thresholds for the 5 doc-trait heads, in label
    /// order. Selected on the calibration split to maximize F1 (the rare
    /// crumpled/shadow heads over-predict at a blanket 0.5). Defaults to 0.5
    /// per label when absent.
    pub label_thresholds: Option<Vec<f64>>,
    pub model: Model,
}

fn default_tau() -> f64 {
    0.9
}

/// Untagged input to `ModelManifest::from_json`: manifest object or flat array.
#[derive(Deserialize)]
#[serde(untagged)]
enum ModelInput {
    Manifest {
        labels: Vec<String>,
        #[serde(default = "default_tau")]
        tau: f64,
        trees: Vec<Vec<RawNode>>,
        /// Exact tree count per label. Absent on older manifests, which split
        /// `trees` evenly (valid only when every head uses the same
        /// `n_estimators`). `train_model_v2.py` always writes it because its
        /// seven heads use 200/200/200/80/80/160/160 trees.
        #[serde(default)]
        trees_per_label: Option<Vec<usize>>,
        #[serde(default)]
        version: Option<String>,
        /// 21-value isotonic lookup table mapping raw confidence to the
        /// empirical probability the rotation is correct. Absent on older
        /// manifests (identity calibration).
        #[serde(default)]
        confidence_calibration: Option<Vec<f64>>,
        /// Per-label doc-trait decision thresholds (index-aligned with labels).
        /// Absent on older manifests (0.5 per label).
        #[serde(default)]
        label_thresholds: Option<Vec<f64>>,
    },
    Flat(Vec<Vec<RawNode>>),
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Model {
    /// Build a model from per-label tree lists, splitting `raw_trees` evenly
    /// across `num_labels` labels (the remainder goes to the first labels).
    fn from_raw_trees(
        raw_trees: Vec<Vec<RawNode>>,
        num_labels: usize,
        trees_per_label: Option<&[usize]>,
    ) -> Result<Self, String> {
        if num_labels == 0 {
            return Err("num_labels must be > 0".to_string());
        }
        let total_trees = raw_trees.len();

        // Exact per-label counts when the manifest carries them; otherwise
        // split evenly (legacy behavior, valid only for same-n_estimators
        // heads). train_model_v2.py uses uneven heads, so its manifest must
        // carry trees_per_label or trees would land on the wrong labels.
        let counts: Vec<usize> = match trees_per_label {
            Some(counts) => {
                if counts.len() != num_labels {
                    return Err(format!(
                        "trees_per_label has {} entries, expected {}",
                        counts.len(),
                        num_labels
                    ));
                }
                let sum: usize = counts.iter().sum();
                if sum != total_trees {
                    return Err(format!(
                        "trees_per_label sums to {}, but {} trees were provided",
                        sum, total_trees
                    ));
                }
                counts.to_vec()
            }
            None => {
                let base = total_trees / num_labels;
                let remainder = total_trees % num_labels;
                (0..num_labels)
                    .map(|i| base + if i < remainder { 1 } else { 0 })
                    .collect()
            }
        };

        let all_trees: Vec<Tree> = raw_trees
            .into_iter()
            .map(|nodes| Tree::from_raw(&nodes))
            .collect();

        let mut split = Vec::with_capacity(num_labels);
        let mut offset = 0;
        for count in counts {
            split.push(all_trees[offset..offset + count].to_vec());
            offset += count;
        }

        Ok(Self {
            trees_per_label: split,
        })
    }

    /// Parse from XGBoost JSON dump format: `[{...per-label trees...}]`.
    /// Each label's trees are dumped consecutively. `num_labels` tells how many
    /// labels there are; trees are split evenly.
    pub fn from_xgboost_json(json: &str, num_labels: usize) -> Result<Self, serde_json::Error> {
        let raw_trees: Vec<Vec<RawNode>> = serde_json::from_str(json)?;
        Self::from_raw_trees(raw_trees, num_labels, None).map_err(serde_json::Error::custom)
    }

    /// Evaluate all trees on a feature vector. Returns a prediction for each label
    /// (sum of tree logits passed through sigmoid).
    pub fn predict(&self, features: &[f64]) -> Vec<f64> {
        self.trees_per_label
            .iter()
            .map(|trees| {
                let sum: f64 = trees.iter().map(|t| t.evaluate(features)).sum();
                sigmoid(sum)
            })
            .collect()
    }

    /// Return raw logits (before sigmoid) for each label.
    pub fn predict_logits(&self, features: &[f64]) -> Vec<f64> {
        self.trees_per_label
            .iter()
            .map(|trees| trees.iter().map(|t| t.evaluate(features)).sum())
            .collect()
    }
}

impl ModelManifest {
    /// Parse the v3 manifest `{labels, tau, version, trees, trees_per_label}`
    /// (where `version` is a release-tag string) or a legacy flat tree array.
    ///
    /// The legacy flat array carries no label names or tau; those callers must
    /// keep using `Model::from_xgboost_json(json, num_labels)` instead.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let input: ModelInput = serde_json::from_str(json)?;
        let (labels, tau, raw_trees, trees_per_label, version, confidence_calibration, label_thresholds) =
            match input {
                ModelInput::Manifest {
                    labels,
                    tau,
                    trees,
                    trees_per_label,
                    version,
                    confidence_calibration,
                    label_thresholds,
                } => (labels, tau, trees, trees_per_label, version, confidence_calibration, label_thresholds),
                ModelInput::Flat(trees) => (Vec::new(), default_tau(), trees, None, None, None, None),
            };
        if labels.is_empty() {
            return Err(serde_json::Error::custom(
                "manifest has empty labels; use from_xgboost_json(json, num_labels) for legacy flat arrays",
            ));
        }
        let model = Model::from_raw_trees(raw_trees, labels.len(), trees_per_label.as_deref())
            .map_err(serde_json::Error::custom)?;
        Ok(Self {
            labels,
            tau,
            version,
            confidence_calibration,
            label_thresholds,
            model,
        })
    }

    /// Binarize the 5 doc-trait heads at their per-label thresholds (default
    /// 0.5). Returns booleans index-aligned with the first 5 labels.
    pub fn trait_labels(&self, features: &[f64]) -> Vec<bool> {
        let preds = self.model.predict(features);
        preds
            .iter()
            .enumerate()
            .take(5)
            .map(|(i, &p)| {
                let t = self
                    .label_thresholds
                    .as_ref()
                    .and_then(|v| v.get(i).copied())
                    .unwrap_or(0.5);
                p >= t
            })
            .collect()
    }
}

impl Tree {
    fn from_raw(raw_nodes: &[RawNode]) -> Self {
        let mut nodes = Vec::new();
        build_tree(&mut nodes, raw_nodes, 0);
        Tree { nodes }
    }

    fn evaluate(&self, features: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            match &self.nodes[idx] {
                Node::Split { feature, threshold } => {
                    idx = if features[*feature] < *threshold {
                        2 * idx + 1
                    } else {
                        2 * idx + 2
                    };
                }
                Node::Leaf { value } => return *value,
            }
        }
    }
}

/// Walk a raw node recursively, emitting nodes breadth-first (by index).
/// Does NOT fill gaps for missing children (leaf-only branches stop).
/// Instead, we fill the array breadth-first by pre-allocating based on the
/// maximum nodeid.
fn build_tree(flat: &mut Vec<Node>, raw_nodes: &[RawNode], idx: usize) {
    // Ensure the vec is large enough
    if idx >= flat.len() {
        flat.resize_with(idx + 1, || Node::Leaf { value: 0.0 });
    }

    if raw_nodes.is_empty() {
        flat[idx] = Node::Leaf { value: 0.0 };
        return;
    }

    match &raw_nodes[0] {
        RawNode::Leaf(leaf) => {
            flat[idx] = Node::Leaf { value: leaf.leaf };
        }
        RawNode::Split(split) => {
            let feature_idx = split
                .split
                .strip_prefix('f')
                .and_then(|s| s.parse::<usize>().ok())
                .expect("split field must be 'f<index>'");
            flat[idx] = Node::Split {
                feature: feature_idx,
                threshold: split.split_condition,
            };

            if !split.children.is_empty() {
                build_tree(flat, &[split.children[0].clone()], 2 * idx + 1);
            }
            if split.children.len() >= 2 {
                build_tree(flat, &[split.children[1].clone()], 2 * idx + 2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_tree_two_labels() {
        // Two labels, two trees each.
        let json = r#"[
            [
                {"nodeid": 0, "depth": 0, "split": "f0", "split_condition": 0.5, "yes": 1, "no": 2, "children": [
                    {"nodeid": 1, "depth": 1, "leaf": -1.0},
                    {"nodeid": 2, "depth": 1, "leaf": 1.0}
                ]}
            ],
            [
                {"nodeid": 0, "depth": 0, "split": "f1", "split_condition": 10.0, "yes": 1, "no": 2, "children": [
                    {"nodeid": 1, "depth": 1, "leaf": 0.5},
                    {"nodeid": 2, "depth": 1, "leaf": -0.5}
                ]}
            ],
            [
                {"nodeid": 0, "depth": 0, "leaf": 2.0}
            ],
            [
                {"nodeid": 0, "depth": 0, "split": "f0", "split_condition": 0.5, "yes": 1, "no": 2, "children": [
                    {"nodeid": 1, "depth": 1, "leaf": 0.0},
                    {"nodeid": 2, "depth": 1, "leaf": -1.0}
                ]}
            ]
        ]"#;

        let model = Model::from_xgboost_json(json, 2).unwrap();

        let feats = [0.3, 5.0];
        let preds = model.predict(&feats);
        assert_eq!(preds.len(), 2);
        let eps = 1e-4;
        assert!((preds[0] - 0.3775).abs() < eps, "got {}", preds[0]);
        assert!((preds[1] - 0.8808).abs() < eps, "got {}", preds[1]);

        let feats2 = [0.7, 15.0];
        let preds2 = model.predict(&feats2);
        assert!((preds2[0] - 0.6225).abs() < eps, "got {}", preds2[0]);
        assert!((preds2[1] - 0.7311).abs() < eps, "got {}", preds2[1]);
    }

    #[test]
    fn test_sigmoid_zero_logit() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_single_label_deeper_tree() {
        let json = r#"[
            [
                {"nodeid": 0, "depth": 0, "split": "f0", "split_condition": 0.5, "yes": 1, "no": 2, "children": [
                    {"nodeid": 1, "depth": 1, "split": "f1", "split_condition": 10.0, "yes": 3, "no": 4, "children": [
                        {"nodeid": 3, "depth": 2, "leaf": 1.0},
                        {"nodeid": 4, "depth": 2, "leaf": -1.0}
                    ]},
                    {"nodeid": 2, "depth": 1, "leaf": 2.0}
                ]}
            ]
        ]"#;

        let model = Model::from_xgboost_json(json, 1).unwrap();

        let p = model.predict(&[0.3, 5.0]);
        assert!((p[0] - 0.7311).abs() < 1e-4, "got {}", p[0]);

        let p = model.predict(&[0.3, 15.0]);
        assert!((p[0] - 0.2689).abs() < 1e-4, "got {}", p[0]);

        let p = model.predict(&[0.7, 0.0]);
        assert!((p[0] - 0.8808).abs() < 1e-4, "got {}", p[0]);
    }

    #[test]
    fn test_manifest_from_json() {
        let json = r#"{
            "labels": ["a", "b", "c"],
            "tau": 0.75,
            "trees": [
                [{"nodeid": 0, "depth": 0, "leaf": 1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": -1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 2.0}],
                [{"nodeid": 0, "depth": 0, "leaf": -2.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 0.5}]
            ]
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert_eq!(
            m.labels,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert!((m.tau - 0.75).abs() < 1e-12);
        assert_eq!(m.model.trees_per_label.len(), 3);
        assert_eq!(m.model.trees_per_label[0].len(), 2);
        assert_eq!(m.model.trees_per_label[2].len(), 2);

        let p = m.model.predict(&[]);
        assert!((p[0] - sigmoid(0.0)).abs() < 1e-12);
        assert!((p[1] - sigmoid(2.0)).abs() < 1e-12);
        assert!((p[2] - sigmoid(-1.5)).abs() < 1e-12);

        let flat = r#"[[{"nodeid":0,"depth":0,"leaf":0.0}]]"#;
        let err = ModelManifest::from_json(flat).unwrap_err();
        assert!(err.to_string().contains("empty labels"), "got: {err}");
    }

    #[test]
    fn test_manifest_uneven_trees_per_label() {
        let json = r#"{
            "labels": ["a", "b", "c"],
            "tau": 0.8,
            "trees_per_label": [3, 1, 2],
            "trees": [
                [{"nodeid": 0, "depth": 0, "leaf": 1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": -1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 2.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 0.5}],
                [{"nodeid": 0, "depth": 0, "leaf": -0.5}]
            ]
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert_eq!(m.model.trees_per_label.len(), 3);
        let counts: Vec<usize> = m.model.trees_per_label.iter().map(Vec::len).collect();
        assert_eq!(counts, vec![3, 1, 2]);

        let p = m.model.predict(&[]);
        assert!((p[0] - sigmoid(1.0)).abs() < 1e-12);
        assert!((p[1] - sigmoid(2.0)).abs() < 1e-12);
        assert!((p[2] - sigmoid(0.0)).abs() < 1e-12);

        let bad = r#"{
            "labels": ["a", "b", "c"],
            "tau": 0.8,
            "trees_per_label": [3, 1, 1],
            "trees": [
                [{"nodeid": 0, "depth": 0, "leaf": 1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": -1.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 2.0}],
                [{"nodeid": 0, "depth": 0, "leaf": 0.5}],
                [{"nodeid": 0, "depth": 0, "leaf": -0.5}]
            ]
        }"#;
        let err = ModelManifest::from_json(bad).unwrap_err();
        assert!(err.to_string().contains("sums to"), "got: {err}");
    }

    #[test]
    fn test_load_real_model_v3_manifest() {
        let model_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/model_v3.json");
        let json = std::fs::read_to_string(&model_path)
            .expect("model_v3.json not found — run train_model_v2.py first");
        let manifest = ModelManifest::from_json(&json).expect("failed to parse model_v3.json");

        assert_eq!(manifest.labels, LABELS);
        assert!(manifest.tau.is_finite() && manifest.tau > 0.0);
        assert_eq!(
            manifest.version.as_deref(),
            Some("0.2.0"),
            "model_v3.json must carry its release version 0.2.0"
        );

        // Per-label tree counts pinned by LABEL_PARAMS in train_model_v2.py.
        let expected: &[usize] = &[200, 200, 200, 80, 80, 160, 160];
        let counts: Vec<usize> = manifest
            .model
            .trees_per_label
            .iter()
            .map(Vec::len)
            .collect();
        assert_eq!(counts, expected);
    }

    #[test]
    fn test_combine_rotation_v5_vote_gate() {
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 0.0; // horizontal fold -> cardinal
        features[N_LINES_INDEX] = 16.0;

        // Strong line-vote for upright overrides a wrong high-confidence head.
        features[VOTE_FRAC_INDEX] = 0.9;
        let preds = [1.0, 1.0, 0.0, 0.0, 0.0, 0.99, 1.0];
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 0.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.7777777777777778).abs() < 1e-9, "conf={}", d.confidence);
        assert!(d.needs_review, "margin 0.4 conf 0.778 < tau 0.9 must review");
        assert!((d.flip_vote_count - 16.0).abs() < 1e-9);
        assert!((d.flip_vote_agreement - 0.9).abs() < 1e-9);

        // Strong vote for flipped -> 180.
        features[VOTE_FRAC_INDEX] = 0.1;
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 180.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.flip_vote_agreement - 0.9).abs() < 1e-9);

        // Tie band defers to the head and reports head confidence.
        features[VOTE_FRAC_INDEX] = 0.5;
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 180.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.99).abs() < 1e-12, "conf={}", d.confidence);
        assert!(!d.needs_review, "head conf 0.99 >= tau 0.9");
        assert!((d.flip_vote_agreement - 0.5).abs() < 1e-9);

        // Weak non-tie vote -> low confidence, needs_review fires.
        features[VOTE_FRAC_INDEX] = 0.6;
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 0.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.1111111111111111).abs() < 1e-9, "conf={}", d.confidence);
        assert!(d.needs_review);
    }

    #[test]
    fn test_combine_rotation_glyph_gate() {
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 0.0; // horizontal fold -> cardinal
        features[N_LINES_INDEX] = 16.0;
        features[VOTE_FRAC_INDEX] = 0.5; // line votes would defer to head
        let preds = [1.0, 1.0, 0.0, 0.0, 0.0, 0.99, 1.0];

        // Baked calibration consts: P(flipped) = sigmoid(b0 + b1*L + b2*N),
        // confidence = trust * max(P, 1-P). With L=3.7, N=20 the evidence is
        // clearly flipped, so the gate rounds to 1 and echoes the glyph
        // evidence; its calibrated confidence (not the head's 0.99) decides
        // needs_review. Recompute the expected confidence from the constants
        // so the test stays correct for any baked values.
        let ev = GlyphEvidence {
            sum_logodds: 3.7,
            count: 20.0,
        };
        let d = combine_rotation_calibrated(&features, &preds, 0.9, None, Some(ev));
        let z = GLYPH_CAL_B0 + GLYPH_CAL_B1 * ev.sum_logodds + GLYPH_CAL_B2 * ev.count;
        let p_flip = sigmoid(z);
        let expected_conf = preds[6] * p_flip.max(1.0 - p_flip);
        assert!((d.rotation_deg - 180.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.glyph_count - 20.0).abs() < 1e-9);
        assert!((d.glyph_logodds - 3.7).abs() < 1e-9);
        assert!((d.confidence - expected_conf).abs() < 1e-9, "conf={} expected={}", d.confidence, expected_conf);
        assert!(d.needs_review, "glyph conf {:.3} < tau 0.9", d.confidence);

        // Few glyphs -> falls back to the line-vote gate (head defer here).
        let ev = GlyphEvidence {
            sum_logodds: 3.7,
            count: 2.0,
        };
        let d = combine_rotation_calibrated(&features, &preds, 0.9, None, Some(ev));
        assert!((d.rotation_deg - 180.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.glyph_count - 2.0).abs() < 1e-9);
        assert!((d.confidence - 0.99).abs() < 1e-12, "conf={}", d.confidence);
        assert!(!d.needs_review, "head conf 0.99 >= tau 0.9");

        // Glyph/GBDT disagreement routes to review even when the glyph gate's
        // own confidence clears tau: the two independent flip signals must
        // agree, or the decision is flagged for a human. Here glyph says
        // flipped (L=6, N=40 -> P(flip) ~ 0.94) but the GBDT head says
        // upright (0.1); the answer stays glyph's (rotation 180) and only
        // needs_review is forced.
        let ev = GlyphEvidence {
            sum_logodds: 6.0,
            count: 40.0,
        };
        let preds_disagree = [1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 1.0];
        let d = combine_rotation_calibrated(&features, &preds_disagree, 0.9, None, Some(ev));
        let z = GLYPH_CAL_B0 + GLYPH_CAL_B1 * 6.0 + GLYPH_CAL_B2 * 40.0;
        let p_flip = sigmoid(z);
        let expected_conf = 1.0 * p_flip.max(1.0 - p_flip);
        assert!(expected_conf >= 0.9, "test needs glyph conf >= tau: {expected_conf}");
        assert!((d.confidence - expected_conf).abs() < 1e-9, "conf={}", d.confidence);
        assert!((d.rotation_deg - 180.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!(d.needs_review, "glyph/GBDT disagreement must force review despite conf {expected_conf:.3} >= tau");
    }

    #[test]
    fn test_combine_rotation_matches_python_formula() {
        // axis = 17 deg, flip=0 -> rotation = (180-17) mod 180 = 163.
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 17.0_f64.to_radians();

        let preds = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8];
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 163.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.8 * 0.9).abs() < 1e-12, "conf={}", d.confidence);
        assert!(d.needs_review, "conf 0.72 < tau 0.9 must review");

        // flip=1 -> rotation = 163 + 180 = 343.
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.rotation_deg - 343.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.9).abs() < 1e-12, "conf={}", d.confidence);
        assert!(!d.needs_review, "conf 0.9 >= tau 0.5 and is_document");

        // Not a document -> review regardless of confidence.
        let preds = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let d = combine_rotation(&features, &preds, 0.0);
        assert!(d.needs_review, "not is_document must force review");

        // axis exactly 0, flip=0 -> rotation = 0 (upright).
        features[RADON_AXIS_INDEX] = 0.0;
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0];
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.rotation_deg - 0.0).abs() < 1e-9, "deg={}", d.rotation_deg);
    }

    #[test]
    fn test_axis_agreement_counts() {
        // radon 30deg; boundary 35deg (agrees within 10deg); docstrum link
        // direction 120deg -> text axis 30deg (agrees), concentrated.
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 30.0_f64.to_radians();
        features[N_LINES_INDEX] = 12.0;
        features[VOTE_FRAC_INDEX] = 0.8;
        features[BOUNDARY_VISIBLE_INDEX] = 1.0;
        features[BOUNDARY_LONG_EDGE_INDEX] = 35.0_f64.to_radians();
        features[DOCSTRUM_CONC_INDEX] = 0.7;
        features[DOCSTRUM_MEAN_INDEX] = 120.0_f64.to_radians();
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0];
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.axis_vote_count - 3.0).abs() < 1e-9, "votes={}", d.axis_vote_count);
        assert!((d.axis_agreement - 1.0).abs() < 1e-9, "agree={}", d.axis_agreement);

        // boundary disagrees (80deg vs 30) and docstrum is diffuse -> only
        // radon + boundary count; agreement = 1/2.
        features[BOUNDARY_LONG_EDGE_INDEX] = 80.0_f64.to_radians();
        features[DOCSTRUM_CONC_INDEX] = 0.2;
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.axis_vote_count - 2.0).abs() < 1e-9, "votes={}", d.axis_vote_count);
        assert!((d.axis_agreement - 0.5).abs() < 1e-9, "agree={}", d.axis_agreement);
    }

    #[test]
    fn test_calibrate_confidence() {
        // identity curve: bin i = i/20, so calibrate(raw) ~ raw.
        let identity: Vec<f64> = (0..21).map(|i| i as f64 / 20.0).collect();
        assert!((calibrate_confidence(0.37, &identity) - 0.37).abs() < 1e-9);
        // clamps out-of-range raw values
        assert!((calibrate_confidence(-1.0, &identity) - 0.0).abs() < 1e-9);
        assert!((calibrate_confidence(2.0, &identity) - 1.0).abs() < 1e-9);
        // malformed curve is identity
        assert!((calibrate_confidence(0.6, &[0.0, 1.0]) - 0.6).abs() < 1e-9);
        // constant curve maps everything to its constant
        let zeros = vec![0.0; 21];
        assert!((calibrate_confidence(0.9, &zeros) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_combine_rotation_calibration_applied() {
        // axis 17 deg, non-cardinal -> flip from head p=0.9, trust=0.8 ->
        // raw confidence 0.8*0.9 = 0.72. A curve that maps 0.72 -> 0.95 lifts
        // the decision above tau 0.9 and flips needs_review off.
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 17.0_f64.to_radians();
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8];
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.confidence - 0.72).abs() < 1e-12);
        assert!(d.needs_review, "raw 0.72 < tau 0.9");

        let mut curve = vec![0.0; 21];
        curve[14] = 0.95; // bin 14 = 0.70
        curve[15] = 0.95; // bin 15 = 0.75 -> interpolates to 0.95 at 0.72
        let d = combine_rotation_calibrated(&features, &preds, 0.9, Some(&curve), None);
        assert!((d.confidence - 0.95).abs() < 1e-12, "conf={}", d.confidence);
        assert!(!d.needs_review, "calibrated 0.95 >= tau 0.9");
    }
}
