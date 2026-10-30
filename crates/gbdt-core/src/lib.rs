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
pub const GLYPH_CAL_B0: f64 = 0.3380543378;
pub const GLYPH_CAL_B1: f64 = 0.1624453914;
pub const GLYPH_CAL_B2: f64 = -0.0038413291;
/// C2: below this |sum_logodds| the glyph gate is a coin flip against the
/// GBDT flip head; at or above it the glyph gate is right 39/39 on 600 clean
/// cardinal pages, so the head must not veto it (review-rate fix).
pub const GLYPH_STRONG_LOGODDS: f64 = 30.0;

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
    /// Quarter-turn decision (0/90/180/270), the nearest multiple of 90.
    pub quarter: f64,
    /// Signed deskew residual `quarter - rotation_deg` in (-45, 45]. Not
    /// snapped: a real 0.5-1.0 deg tilt survives the fold-snap.
    pub skew_deg: f64,
    /// The axis (fine angle) is trustworthy even if the flip bit is not.
    pub axis_ok: bool,
    /// The 0-vs-180 bit is trustworthy.
    pub flip_ok: bool,
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

    // v0.4.0 behaviour: the deterministic radon axis is the answer; the
    // boundary/docstrum estimates are REPORTED as agreement votes only. A
    // v0.5.0 "consensus override" let boundary+docstrum flip the radon axis
    // on 90/270 rotated born-digital invoices (sideways pages read as upright
    // at high confidence, not flagged) — feedback.txt section 2's dangerous
    // failure mode. Do not override the radon axis here.
    let axis_rad = features[RADON_AXIS_INDEX];

    let axis_deg = axis_rad.to_degrees().rem_euclid(180.0);
    // C1: one fold convention. The mod-180 axis is blind to the 0/180 fold;
    // signed_tilt_deg returns the small signed residual inside the band so the
    // evidence frame and the wrap formula below agree (no skipped deskew, no
    // two snap constants).
    let tilt = features_core::signed_tilt_deg(axis_deg);
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
    // S4: S3 deskews the mask at the refined fine axis for ANY tilt, so the
    // deterministic glyph/line-vote gates (previously cardinal-only) apply
    // everywhere; the GBDT flip head is the tie-band fallback and the
    // disagreement cross-check below, never the primary off-cardinal signal.
    let (glyph_logodds, glyph_count) = match glyph {
        Some(ev) => (ev.sum_logodds, ev.count),
        None => (0.0, 0.0),
    };
    let signals = [
        glyph_logodds,
        glyph_count,
        features.get(VOTE_FRAC_INDEX).copied().unwrap_or(0.5),
        features.get(FLIP_SCORE_INDEX).copied().unwrap_or(0.0),
        flip_p,
        trust_p,
    ];
    let p_flip = flip_combiner_score(&signals);
    let flip = if p_flip >= 0.5 { 1.0 } else { 0.0 };
    let flip_conf = p_flip.max(1.0 - p_flip);

    let rotation_deg = if tilt.abs() < features_core::FOLD_BAND_DEG {
        (tilt + 180.0 * flip).rem_euclid(360.0)
    } else {
        (180.0 - axis_deg).rem_euclid(180.0) + 180.0 * flip
    };
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
    let needs_review = confidence < tau || preds[0] < 0.5;

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

    // S5: split the decision for callers that want to run their own deskew.
    // `quarter` is the nearest multiple of 90; `skew_deg` is the signed deskew
    // residual in (-45, 45] the caller applies after `quarter`
    // (rotate(quarter) then rotate(skew_deg) == rotate(-rotation_deg) for the
    // small-residual case the field is meant for). q keeps the unwrapped 360
    // so rotation_deg near 0/360 yields skew +t instead of +t-360.
    let q = (rotation_deg / 90.0).round() * 90.0;
    let quarter = q % 360.0;
    let skew_deg = q - rotation_deg;
    let flip_ok = flip_conf >= tau;
    let axis_ok = trust_p >= tau;

    RotationDecision {
        rotation_deg,
        confidence,
        needs_review,
        quarter,
        skew_deg,
        axis_ok,
        flip_ok,
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
        let (
            labels,
            tau,
            raw_trees,
            trees_per_label,
            version,
            confidence_calibration,
            label_thresholds,
        ) = match input {
            ModelInput::Manifest {
                labels,
                tau,
                trees,
                trees_per_label,
                version,
                confidence_calibration,
                label_thresholds,
            } => (
                labels,
                tau,
                trees,
                trees_per_label,
                version,
                confidence_calibration,
                label_thresholds,
            ),
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

/// Flip-decision combiner: a tiny depth-3 GBDT over six evidence signals
/// [glyph_logodds, glyph_count, vote_frac, flip_score, flip_p, trust_p].
/// Baked by scripts/train_flip_combiner.py; scored through the SAME tree
/// infra as the seven heads.
const FLIP_COMBINER_TREES_JSON: &str = r#"[[{"nodeid":0,"depth":0,"split":"f0","split_condition":-8.90514755,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f4","split_condition":0.584627688,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f0","split_condition":-19.2400494,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.197292194},{"nodeid":8,"leaf":-0.0808387771}]},{"nodeid":4,"leaf":0.0309608616}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.456875145,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.0600081496},{"nodeid":10,"leaf":0.0713992789}]},{"nodeid":6,"leaf":0.199634999}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":4.38602924,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.180332318},{"nodeid":4,"depth":2,"split":"f3","split_condition":-0.0356671065,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0613611005},{"nodeid":8,"leaf":-0.0636336729}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f3","split_condition":-0.0979847088,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.0993416682},{"nodeid":10,"leaf":-0.000827320328}]},{"nodeid":6,"leaf":0.182066664}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":4.38602924,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-19.2400494,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.166275367},{"nodeid":4,"depth":2,"split":"f3","split_condition":-0.0114345113,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0525749438},{"nodeid":8,"leaf":-0.0557260886}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f3","split_condition":-0.0979847088,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.107657373},{"nodeid":10,"leaf":-0.012116109}]},{"nodeid":6,"leaf":0.16772306}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":4.00239038,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.156579435},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.456875145,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0657969341},{"nodeid":8,"leaf":0.0415823124}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.564239442,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.0215444546},{"nodeid":10,"leaf":0.103845537}]},{"nodeid":6,"leaf":0.157667801}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":6.1051836,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-19.2400494,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.51655829,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.148094341},{"nodeid":8,"leaf":-0.0449898653}]},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.456875145,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.0581281967},{"nodeid":10,"leaf":0.0394101143}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f0","split_condition":11.9898977,"yes":11,"no":12,"missing":12,"children":[{"nodeid":11,"leaf":0.0604590587},{"nodeid":12,"leaf":-0.0120443404}]},{"nodeid":6,"leaf":0.148533925}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-19.2400494,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f5","split_condition":0.417509884,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0607948117},{"nodeid":4,"leaf":-0.141688958}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f3","split_condition":-0.0855188146,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0544660687},{"nodeid":8,"leaf":-0.028215399}]},{"nodeid":6,"leaf":0.141568452}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-19.2400494,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f0","split_condition":-71.5501099,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.135972932},{"nodeid":6,"leaf":-0.0565928109}]},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.607703447,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0269466434},{"nodeid":8,"leaf":0.0528910719}]}]},{"nodeid":2,"leaf":0.136535347}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-19.2400494,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.131279156},{"nodeid":4,"leaf":-0.0418413728}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.584627688,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0236927196},{"nodeid":8,"leaf":0.0530888513}]},{"nodeid":6,"leaf":0.131721869}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-5.32821846,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.127289414},{"nodeid":4,"depth":2,"split":"f5","split_condition":0.00896167755,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0356334336},{"nodeid":8,"leaf":-0.0582152605}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.607703447,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.0224695709},{"nodeid":10,"leaf":0.0652041957}]},{"nodeid":6,"leaf":0.127668306}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":7.72058821,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.123848692},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.564239442,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0344909914},{"nodeid":8,"leaf":0.0373206995}]}]},{"nodeid":2,"depth":1,"split":"f5","split_condition":0.0197029486,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f5","split_condition":0.010200982,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.0605078749},{"nodeid":10,"leaf":-0.0322236232}]},{"nodeid":6,"depth":2,"split":"f0","split_condition":23.1759396,"yes":11,"no":12,"missing":12,"children":[{"nodeid":11,"leaf":0.0496326312},{"nodeid":12,"leaf":0.123916999}]}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":5.48221302,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-36.4744148,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.120297968},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.584627688,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0299369991},{"nodeid":8,"leaf":0.0269047618}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f0","split_condition":11.9898977,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.0568900369},{"nodeid":10,"leaf":0}]},{"nodeid":6,"leaf":0.121227145}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":7.72058821,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.118346885},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.619316876,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0221171025},{"nodeid":8,"leaf":0.037987899}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f1","split_condition":15,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.071094729},{"nodeid":10,"leaf":-0.0102360165}]},{"nodeid":6,"leaf":0.118312828}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-19.2400494,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.527601182,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.116416208},{"nodeid":6,"leaf":-0.00795156788}]},{"nodeid":4,"depth":2,"split":"f3","split_condition":-0.131602511,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0388515778},{"nodeid":8,"leaf":-0.0221925955}]}]},{"nodeid":2,"leaf":0.116308741}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":6.1051836,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f1","split_condition":23,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.422033846,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0343061574},{"nodeid":8,"leaf":0.0193572994}]},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.584627688,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.113303654},{"nodeid":10,"leaf":0}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f0","split_condition":11.9898977,"yes":11,"no":12,"missing":12,"children":[{"nodeid":11,"leaf":0.0492558032},{"nodeid":12,"leaf":0}]},{"nodeid":6,"leaf":0.114328779}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":4.00239038,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-36.4744148,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.11212676},{"nodeid":4,"depth":2,"split":"f3","split_condition":0.199312717,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.00771979336},{"nodeid":8,"leaf":-0.0457898043}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.638769746,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0},{"nodeid":10,"leaf":0.0893749297}]},{"nodeid":6,"leaf":0.112573408}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":6.1051836,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.110871159},{"nodeid":4,"depth":2,"split":"f3","split_condition":0.199312717,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.00721645821},{"nodeid":8,"leaf":-0.043981757}]}]},{"nodeid":2,"depth":1,"split":"f1","split_condition":27,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.379469603,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.0685009062},{"nodeid":10,"leaf":-0.00442424556}]},{"nodeid":6,"leaf":0.110147335}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.109447502},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.427539796,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.0212466531},{"nodeid":6,"leaf":0.0143079339}]}]},{"nodeid":2,"leaf":0.10957282}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-71.5501099,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"leaf":-0.108207099},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.584627688,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.0128747299},{"nodeid":6,"leaf":0.0292735547}]},{"nodeid":4,"leaf":0.108279362}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":7.72058821,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.107031383},{"nodeid":4,"depth":2,"split":"f3","split_condition":-0.0152990269,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0148422485},{"nodeid":8,"leaf":-0.0242578667}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f0","split_condition":11.9898977,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":0.0644391254},{"nodeid":10,"leaf":0}]},{"nodeid":6,"leaf":0.107075669}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-71.5501099,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"leaf":-0.105933309},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.67268306,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.00813590921},{"nodeid":6,"leaf":0.0380573608}]},{"nodeid":4,"leaf":0.10605891}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-71.5501099,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"leaf":-0.105030753},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f3","split_condition":0.0364903659,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":0.0137239024},{"nodeid":6,"leaf":-0.0108610233}]},{"nodeid":4,"leaf":0.105037294}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-19.2400494,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f5","split_condition":0.417509884,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0248548519},{"nodeid":4,"leaf":-0.104673542}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":23.1759396,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f3","split_condition":-0.151388168,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":0.0302620828},{"nodeid":8,"leaf":-0.00446186122}]},{"nodeid":6,"leaf":0.103830457}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.103290275},{"nodeid":4,"depth":2,"split":"f3","split_condition":0.149758458,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":0.00173504336},{"nodeid":6,"leaf":-0.0254962724}]}]},{"nodeid":2,"leaf":0.10330414}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-36.4744148,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f3","split_condition":-0.485659659,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0303812418},{"nodeid":4,"leaf":-0.102861717}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.194705397,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0839990601},{"nodeid":8,"leaf":0.00226764474}]},{"nodeid":6,"leaf":0.102510653}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.10172645},{"nodeid":4,"depth":2,"split":"f3","split_condition":0.81596452,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.00206215377},{"nodeid":6,"leaf":-0.063583307}]}]},{"nodeid":2,"leaf":0.10170754}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-36.4744148,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f4","split_condition":0.383175761,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.101303175},{"nodeid":4,"leaf":-0.0289351791}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.67268306,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.00548793888},{"nodeid":8,"leaf":0.0390075408}]},{"nodeid":6,"leaf":0.100971818}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-71.5501099,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"leaf":-0.100318208},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"depth":2,"split":"f4","split_condition":0.281404346,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.0373756401},{"nodeid":6,"leaf":0.013701777}]},{"nodeid":4,"leaf":0.100269221}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0996136218},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.203589216,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":-0.0767658129},{"nodeid":6,"leaf":0.00138486258}]}]},{"nodeid":2,"leaf":0.0995596051}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":23.1759396,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0989317596},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.199402973,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.0816703588},{"nodeid":8,"leaf":-0.00457374146}]}]},{"nodeid":2,"depth":1,"split":"f2","split_condition":0.5625,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":0.0995316207},{"nodeid":6,"leaf":0.0271511115}]}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":23.1759396,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-71.5501099,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.0982538909},{"nodeid":4,"depth":2,"split":"f3","split_condition":-0.0909647793,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":0.0180069394},{"nodeid":6,"leaf":-0.0127272801}]}]},{"nodeid":2,"leaf":0.0974784046}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":44.5854492,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f0","split_condition":-36.4744148,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":-0.096241653},{"nodeid":4,"depth":2,"split":"f4","split_condition":0.323153973,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"leaf":0.0162070151},{"nodeid":6,"leaf":-0.00682658423}]}]},{"nodeid":2,"leaf":0.0974769592}]}],[{"nodeid":0,"depth":0,"split":"f0","split_condition":-5.32821846,"yes":1,"no":2,"missing":2,"children":[{"nodeid":1,"depth":1,"split":"f5","split_condition":0.00896167755,"yes":3,"no":4,"missing":4,"children":[{"nodeid":3,"leaf":0.0183692686},{"nodeid":4,"depth":2,"split":"f2","split_condition":1,"yes":7,"no":8,"missing":8,"children":[{"nodeid":7,"leaf":-0.095454447},{"nodeid":8,"leaf":0}]}]},{"nodeid":2,"depth":1,"split":"f0","split_condition":44.5854492,"yes":5,"no":6,"missing":6,"children":[{"nodeid":5,"depth":2,"split":"f4","split_condition":0.223899409,"yes":9,"no":10,"missing":10,"children":[{"nodeid":9,"leaf":-0.0731490478},{"nodeid":10,"leaf":0.00905007683}]},{"nodeid":6,"leaf":0.0967261493}]}]}]]"#;

fn flip_combiner_trees() -> &'static Vec<Tree> {
    static TREES: std::sync::OnceLock<Vec<Tree>> = std::sync::OnceLock::new();
    TREES.get_or_init(|| {
        let raw: Vec<Vec<RawNode>> = serde_json::from_str(FLIP_COMBINER_TREES_JSON).unwrap();
        raw.iter().map(|t| Tree::from_raw(t)).collect()
    })
}

fn flip_combiner_score(signals: &[f64]) -> f64 {
    let sum: f64 = flip_combiner_trees()
        .iter()
        .map(|t| t.evaluate(signals))
        .sum();
    sigmoid(sum)
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
    #[test]
    #[test]
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
        assert!(
            (d.axis_vote_count - 3.0).abs() < 1e-9,
            "votes={}",
            d.axis_vote_count
        );
        assert!(
            (d.axis_agreement - 1.0).abs() < 1e-9,
            "agree={}",
            d.axis_agreement
        );

        // boundary disagrees (80deg vs 30) and docstrum is diffuse -> only
        // radon + boundary count; agreement = 1/2.
        features[BOUNDARY_LONG_EDGE_INDEX] = 80.0_f64.to_radians();
        features[DOCSTRUM_CONC_INDEX] = 0.2;
        let d = combine_rotation(&features, &preds, 0.5);
        assert!(
            (d.axis_vote_count - 2.0).abs() < 1e-9,
            "votes={}",
            d.axis_vote_count
        );
        assert!(
            (d.axis_agreement - 0.5).abs() < 1e-9,
            "agree={}",
            d.axis_agreement
        );
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
}
