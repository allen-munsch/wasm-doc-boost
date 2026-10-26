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
pub const LABELS_V2: [&str; 7] = [
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
pub fn combine_rotation(features: &[f64], preds: &[f64], tau: f64) -> (f64, f64, bool) {
    let axis_rad = features[RADON_AXIS_INDEX];
    let mut axis_deg = axis_rad.to_degrees().rem_euclid(180.0);
    if axis_deg < 1.0 || axis_deg > 179.0 {
        axis_deg = 0.0;
    }
    let flip_p = preds[5];
    let trust_p = preds[6];
    let flip = if flip_p >= 0.5 { 1.0 } else { 0.0 };
    let rotation_deg = (180.0 - axis_deg).rem_euclid(180.0) + 180.0 * flip;
    let confidence = trust_p * flip_p.max(1.0 - flip_p);
    let needs_review = confidence < tau || preds[0] < 0.5;
    (rotation_deg, confidence, needs_review)
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
/// `{"labels": [...], "tau": 0.9, "trees": [[...], ...], "trees_per_label": [...]}`.
/// The legacy flat tree array is also accepted for backward compatibility.
#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub labels: Vec<String>,
    pub tau: f64,
    pub version: Option<u32>,
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
        version: Option<u32>,
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
    /// or a legacy flat tree array.
    ///
    /// The legacy flat array carries no label names or tau; those callers must
    /// keep using `Model::from_xgboost_json(json, num_labels)` instead.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let input: ModelInput = serde_json::from_str(json)?;
        let (labels, tau, raw_trees, trees_per_label, version) = match input {
            ModelInput::Manifest {
                labels,
                tau,
                trees,
                trees_per_label,
                version,
            } => (labels, tau, trees, trees_per_label, version),
            ModelInput::Flat(trees) => (Vec::new(), default_tau(), trees, None, None),
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
            model,
        })
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

            if split.children.len() >= 1 {
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

        assert_eq!(manifest.labels, LABELS_V2);
        assert!(manifest.tau.is_finite() && manifest.tau > 0.0);
        assert_eq!(
            manifest.version,
            Some(3),
            "model_v3.json must carry version 3"
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
    fn test_combine_rotation_matches_python_formula() {
        // axis = 17 deg, flip=0 -> rotation = (180-17) mod 180 = 163.
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 17.0_f64.to_radians();

        let preds = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8];
        let (deg, conf, review) = combine_rotation(&features, &preds, 0.9);
        assert!((deg - 163.0).abs() < 1e-9, "deg={deg}");
        assert!((conf - 0.8 * 0.9).abs() < 1e-12, "conf={conf}");
        assert!(review, "conf 0.72 < tau 0.9 must review");

        // flip=1 -> rotation = 163 + 180 = 343.
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let (deg, conf, review) = combine_rotation(&features, &preds, 0.5);
        assert!((deg - 343.0).abs() < 1e-9, "deg={deg}");
        assert!((conf - 0.9).abs() < 1e-12, "conf={conf}");
        assert!(!review, "conf 0.9 >= tau 0.5 and is_document");

        // Not a document -> review regardless of confidence.
        let preds = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let (_, _, review) = combine_rotation(&features, &preds, 0.0);
        assert!(review, "not is_document must force review");

        // axis exactly 0, flip=0 -> rotation = 0 (upright).
        features[RADON_AXIS_INDEX] = 0.0;
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0];
        let (deg, _, _) = combine_rotation(&features, &preds, 0.5);
        assert!((deg - 0.0).abs() < 1e-9, "deg={deg}");
    }
}
