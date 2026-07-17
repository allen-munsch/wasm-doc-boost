use serde::Deserialize;

/// A single node in a GBDT tree (XGBoost JSON dump format).
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum RawNode {
    Split(Box<RawSplitNode>),
    Leaf(RawLeafNode),
}

#[derive(Debug, Deserialize, Clone)]
struct RawSplitNode {
    split: String,           // e.g. "f0"
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
#[derive(Clone)]
struct Tree {
    /// For each node: if Some(feature_index, threshold), it's a split node
    /// with left = 2*idx+1, right = 2*idx+2. If None, it's a leaf with the given value.
    nodes: Vec<Node>,
}

#[derive(Clone)]
enum Node {
    Split { feature: usize, threshold: f64 },
    Leaf { value: f64 },
}

/// A trained model: `trees_per_label[i]` is the list of trees for label `i`.
pub struct Model {
    trees_per_label: Vec<Vec<Tree>>,
    num_labels: usize,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Model {
    /// Parse from XGBoost JSON dump format: `[{...per-label trees...}]`.
    /// Each label's trees are dumped consecutively. `num_labels` tells how many
    /// labels there are; trees are split evenly.
    pub fn from_xgboost_json(json: &str, num_labels: usize) -> Result<Self, serde_json::Error> {
        let raw_trees: Vec<Vec<RawNode>> = serde_json::from_str(json)?;
        let total_trees = raw_trees.len();
        assert!(
            total_trees % num_labels == 0,
            "Total trees ({total_trees}) not divisible by num_labels ({num_labels})"
        );
        let trees_per_label_count = total_trees / num_labels;

        let all_trees: Vec<Tree> = raw_trees
            .into_iter()
            .map(|nodes| Tree::from_raw(&nodes))
            .collect();

        let mut trees_per_label = Vec::with_capacity(num_labels);
        for label_idx in 0..num_labels {
            let start = label_idx * trees_per_label_count;
            let end = start + trees_per_label_count;
            trees_per_label.push(all_trees[start..end].to_vec());
        }

        Ok(Self {
            trees_per_label,
            num_labels,
        })
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
                Node::Split {
                    feature,
                    threshold,
                } => {
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
        // Tree 0 (label 0): split on f0 < 0.5 -> leaf -1.0 else leaf 1.0
        // Tree 1 (label 0): split on f1 < 10.0 -> leaf 0.5 else leaf -0.5
        // Tree 2 (label 1): always leaf 2.0
        // Tree 3 (label 1): split on f0 < 0.5 -> leaf 0.0 else leaf -1.0
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

        // Label 0: Tree0 + Tree1
        // f0=0.3 (< 0.5) -> Tree0 leaf -1.0
        // f1=5.0 (< 10.0) -> Tree1 leaf 0.5
        // Logit0 = -0.5, sigmoid(-0.5) ≈ 0.3775
        //
        // Label 1: Tree2 + Tree3
        // Tree2 always leaf 2.0
        // f0=0.3 (< 0.5) -> Tree3 leaf 0.0
        // Logit1 = 2.0, sigmoid(2.0) ≈ 0.8808

        let feats = [0.3, 5.0];
        let preds = model.predict(&feats);
        assert_eq!(preds.len(), 2);
        let eps = 1e-4;
        assert!((preds[0] - 0.3775).abs() < eps, "got {}", preds[0]);
        assert!((preds[1] - 0.8808).abs() < eps, "got {}", preds[1]);

        // Test opposite branch
        let feats2 = [0.7, 15.0];
        let preds2 = model.predict(&feats2);
        // Label 0: 1.0 + (-0.5) = 0.5, sigmoid(0.5) ≈ 0.6225
        // Label 1: 2.0 + (-1.0) = 1.0, sigmoid(1.0) ≈ 0.7311
        assert!((preds2[0] - 0.6225).abs() < eps, "got {}", preds2[0]);
        assert!((preds2[1] - 0.7311).abs() < eps, "got {}", preds2[1]);
    }

    #[test]
    fn test_sigmoid_zero_logit() {
        // Empty model — no trees means logit 0, sigmoid(0) = 0.5
        let _json = r#"[]"#;
        // 0 labels, but divisible by anything...
        // Let's just test sigmoid directly
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_single_label_deeper_tree() {
        // One label, one tree with depth 2
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

        // f0=0.3, f1=5.0: left->left leaf=1.0, sigmoid(1.0) ≈ 0.7311
        let p = model.predict(&[0.3, 5.0]);
        assert!((p[0] - 0.7311).abs() < 1e-4, "got {}", p[0]);

        // f0=0.3, f1=15.0: left->right leaf=-1.0, sigmoid(-1.0) ≈ 0.2689
        let p = model.predict(&[0.3, 15.0]);
        assert!((p[0] - 0.2689).abs() < 1e-4, "got {}", p[0]);

        // f0=0.7: right leaf=2.0, sigmoid(2.0) ≈ 0.8808
        let p = model.predict(&[0.7, 0.0]);
        assert!((p[0] - 0.8808).abs() < 1e-4, "got {}", p[0]);
    }

    #[test]
    fn test_load_real_model_json() {
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/model.json");
        let json = std::fs::read_to_string(&model_path)
            .expect("model.json not found — run train_model.py first");
        let model = Model::from_xgboost_json(&json, 5).expect("failed to parse model.json");

        assert_eq!(model.trees_per_label.len(), 5);
        for label_trees in &model.trees_per_label {
            assert_eq!(label_trees.len(), 200);
        }

        // Real features from features.npz row 2 (FATURA2 invoice), 81 features
        let features: [f64; 81] = [
            246.123812046576603, 34.923879349001929, -4.443664893809219, 19.739888039217384,
            247.122260544271910, 31.903393179906757, -4.764195873488212, 23.470811483256185,
            247.532749132346197, 31.367676930807438, -4.955053603047298, 25.250841738048003,
            246.870520142500425, 255.000000000000000, 32.407771450260064, 1.454332126645650,
            18.695114823028216, 0.006275965441643473, 0.003434552489341624, 3931.454784852435296,
            40.646012087349590, 133.521769859907039, 76.837490849194154, 0.071774551037398,
            0.100312531505192, 0.143714084081057, 0.145427966528884, 0.103185805020667,
            0.096783949994959, 0.155408811372114, 0.155660852908559, 0.099505998588567,
            0.425415734719450, 0.023607702371042, 0.907200329875224, 0.006723422833442480,
            0.013490504256713, 0.003841955904824274, 0.008977297533289675, 0.013648766644837,
            0.024170486793277, 0.013381357782144, 0.024547042130539, 0.831303208906352,
            0.059915957214582, 9.268125625294253, 0.065779744731642, 0.552380058960189,
            0.823738922588006, 1.543996712304540, 450.652177042278367, 0.184738407022927,
            0.015611303781912, 0.026538050982966, 0.046101782876172, 0.060511491070856,
            58.427173010246584, 67.897516987679396, 0.005117209552312913, 0.046923483506546,
            0.094351612300198, 0.028879022928423, 0.064522629689254, 0.099366427708991,
            0.114550564071695, 0.046213505423751, 0.099715069222561, 0.089351782708269,
            0.316125902440300, 0.117206813966898, 2.035787200752338, 6436.676039550857240,
            298.000000000000000, 1.166666666666667, 1.035714285714286, 0.084070796460177,
            2.964483545128707, 2.770999604749259, 31.425569211252650, 26.000000000000000,
            0.671554506328608,
        ];
        let preds = model.predict(&features);
        assert_eq!(preds.len(), 5);
        assert!(preds[0] > 0.5, "is_document: expected >0.5, got {:.6}", preds[0]);
        assert!(preds[1] > 0.5, "is_digital: expected >0.5, got {:.6}", preds[1]);
        assert!(preds[2] < 0.5, "is_paper: expected <0.5, got {:.6}", preds[2]);
        assert!(preds[3] < 0.5, "is_crumpled: expected <0.5, got {:.6}", preds[3]);
        assert!(preds[4] < 0.5, "is_shadow: expected <0.5, got {:.6}", preds[4]);
    }
}
