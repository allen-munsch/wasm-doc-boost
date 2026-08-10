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
        let trees_per_label_count = total_trees / num_labels;
        let remainder = total_trees % num_labels;

        let all_trees: Vec<Tree> = raw_trees
            .into_iter()
            .map(|nodes| Tree::from_raw(&nodes))
            .collect();

        let mut trees_per_label = Vec::with_capacity(num_labels);
        let mut offset = 0;
        for label_idx in 0..num_labels {
            let extra = if label_idx < remainder { 1 } else { 0 };
            let count = trees_per_label_count + extra;
            trees_per_label.push(all_trees[offset..offset + count].to_vec());
            offset += count;
        }

        Ok(Self { trees_per_label })
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
        let model = Model::from_xgboost_json(&json, 9).expect("failed to parse model.json");

        assert_eq!(model.trees_per_label.len(), 9);
        // 1160 trees / 9 = 128 r 8: first 8 labels get 129, last gets 128
        assert_eq!(model.trees_per_label[0].len(), 129);
        assert_eq!(model.trees_per_label[8].len(), 128);

        // FATURA2 invoice (digital document, rotation_0) from features_v3.npz row 8
        let features: [f64; 103] = [
            2.461238120465766e2, 3.492387934900193e1, -4.443664893809219e0, 1.973988803921738e1,
            2.471222605442719e2, 3.190339317990676e1, -4.764195873488212e0, 2.347081148325618e1,
            2.475327491323462e2, 3.136767693080744e1, -4.955053603047298e0, 2.525084173804800e1,
            2.468705201425004e2, 2.550000000000000e2, 3.240777145026006e1, 1.454332126645650e0,
            1.869511482302822e1, 6.275965441643473e-3, 3.434552489341624e-3, 3.931454784852435e3,
            4.064601208734959e1, 1.335217698599070e2, 7.683749084919415e1, 7.177455103739823e-2,
            1.003125315051921e-1, 1.437140840810566e-1, 1.454279665288840e-1, 1.031858050206674e-1,
            9.678394999495916e-2, 1.554088113721141e-1, 1.556608529085593e-1, 9.950599858856740e-2,
            4.254157347194501e-1, 2.360770237104151e-2, 4.966209847836272e-1, 9.709197459375979e-2,
            3.245125814963981e-1, -5.000000000002246e-1, 9.999659519258044e-1, 9.072003298752237e-1,
            6.723422833442480e-3, 1.349050425671251e-2, 3.841955904824274e-3, 8.977297533289675e-3,
            1.364876664483737e-2, 2.417048679327658e-2, 1.338135778214364e-2, 2.454704213053919e-2,
            8.313032089063523e-1, 5.991595721458197e-2, 9.268125625294253e0, 6.577974473164151e-2,
            5.523800589601892e-1, 8.237389225880064e-1, 1.543996712304540e0, 4.506521770422784e2,
            1.847384070229266e-1, 1.561130378191151e-2, 2.653805098296607e-2, 4.610178287617173e-2,
            6.051149107085564e-2, 5.842717301024658e1, 6.789751698767940e1, 5.117209552312913e-3,
            4.692348350654557e-2, 9.435161230019849e-2, 2.887902292842280e-2, 6.452262968925411e-2,
            9.936642770899121e-2, 1.145505640716952e-1, 4.621350542375085e-2, 9.971506922256088e-2,
            8.935178270826852e-2, 3.161259024403004e-1, 1.172068139668980e-1, 2.035787200752338e0,
            6.436676039550857e3, 2.980000000000000e2, 1.166666666666667e0, 1.035714285714286e0,
            8.407079646017700e-2, 2.699115044247787e-1, 2.964483545128707e0, 2.770999604749259e0,
            3.265613783201723e0, 2.645170991941413e0, 3.142556921125265e1, 2.600000000000000e1,
            6.715545063286079e-1, 1.991299470105093e2, 2.534688717347571e1, 7.856189426640489e0,
            1.408923034444260e0, 2.441767554879263e0, 4.883177822707074e-2, 1.149353380936564e2,
            2.936605981794538e0, 2.732351940771008e0, 3.246944644140906e0, 2.638496526708007e0,
            1.840801092244015e0, 1.885444059560367e0, -1.523034770747767e0,
        ];
        let preds = model.predict(&features);
        assert_eq!(preds.len(), 9);
        // FATURA2 is a document at rotation_0
        assert!(preds[0] > 0.5, "is_document: expected >0.5, got {:.6}", preds[0]);

        // COCO image (not a document, rotation_0) from features_v3.npz row 0
        let coco_features: [f64; 103] = [
            7.056054077148535e1, 6.216058803953266e1, 9.739075448121478e-1, 2.339576351975996e-1,
            6.665669555664023e1, 6.019008064316477e1, 1.103049146130306e0, 6.070152091328977e-1,
            6.314204101562432e1, 5.506596698820423e1, 1.348378811888754e0, 1.526615131684659e0,
            6.742327465820082e1, 4.700000000000000e1, 5.975246723755046e1, 7.373735654174316e0,
            3.002629790642568e1, 3.200460660115559e-1, 4.986932353180536e-2, 3.319842725897739e3,
            1.219657705743403e2, 1.846904936335013e2, 3.407132264559156e2, 1.117797851562500e-1,
            1.465877590531086e-1, 1.240594887138646e-1, 1.100673208078497e-1, 1.368196418357020e-1,
            1.329035948431381e-1, 1.211334536014432e-1, 1.211114533374401e-1, 1.073172878074537e-1,
            2.419131182827015e0, 3.126831054687500e-2, 3.407982894805474e-2, 7.686246535777325e-2,
            5.467981398576892e-1, 1.465862308901814e-1, 9.999972001779965e-1, 5.177080145746936e-1,
            6.636453323467752e-2, 9.757060056727093e-2, 5.090023430755950e-2, 8.734122579849550e-2,
            1.430385990874337e-1, 8.132938710075226e-2, 4.427179676902208e-2, 9.927241336786287e-2,
            7.553335799728697e-2, 2.543778517696387e-1, 2.465161546610169e1, 2.615109090004271e-1,
            1.759759046604669e-2, 3.847427254712225e-1, 1.914808321473062e0, 6.618853590223723e2,
            0.000000000000000e0, 1.427392884099375e-1, 4.799919220217253e-2, 5.763122558593750e-1,
            3.493505647125090e3, 4.392773243655279e1, 4.110035190986130e1, 1.289581109262095e1,
            5.092230325598577e-2, 7.948099120372677e-2, 3.340017808871297e-2, 1.234813367721312e-1,
            2.394651804604688e-1, 1.180499501184600e-1, 2.832208645491476e-2, 8.193076918314983e-2,
            7.039388883156893e-2, 1.745533156308767e-1, 1.058025249684684e-1, 9.452368069711427e-1,
            1.569022269275137e0, 3.850000000000000e2, 9.166666666666667e-1, 5.000000000000000e-1,
            1.742424242424243e-1, 4.848484848484849e-1, 1.829098428453267e1, 2.585253452846870e1,
            2.081886650348333e1, 3.456568561485985e1, 7.262114333294655e1, 4.300000000000000e1,
            1.720579836283417e0, 5.854012815259910e2, 7.658264492029280e2, 7.644046273607767e-1,
            1.563535743823487e0, 1.853014553098322e0, 6.814453125000000e-1, 3.259234940168774e1,
            1.842072265302755e1, 2.606166192473689e1, 2.094316263365222e1, 3.483556188494462e1,
            2.117459160328423e0, 2.169197622119238e0, -1.567831763461863e0,
        ];
        let coco_preds = model.predict(&coco_features);
        assert_eq!(coco_preds.len(), 9);
        assert!(coco_preds[0] < 0.5, "COCO is_document: expected <0.5, got {:.6}", coco_preds[0]);
    }
}
