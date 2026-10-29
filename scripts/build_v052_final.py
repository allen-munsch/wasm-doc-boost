#!/usr/bin/env python3
"""Build the v0.5.2 release model: retrain the three doc-TYPE heads
(is_document/is_digital/is_paper) with the DocLayNet born-digital corpus
(upright, sample-weighted 16x to match v9's 16-angle augmentation), keep the
four remaining heads (is_crumpled/is_shadow/flip/trust) on v9 train only.

Produces data/model_v052_final.json + data/report_v052_final.json in the exact
manifest/report format of train_model.py, so benchmark_v052.py and goal_gate.py
run unchanged. Also writes data/doclaynet_report.json (held-out DocLayNet test
page accuracies) for the README.
"""
import argparse, json, os, sys
import numpy as np
import xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as t

DOC_TYPE_HEADS = ['is_document', 'is_digital', 'is_paper']
KEEP_V9_HEADS = ['is_crumpled', 'is_shadow', 'flip', 'trust']
DOCLAYNET_WEIGHT = 16.0  # upright sample == one full 16-angle base image


def train_one(X, y, name, sample_weight=None, device='cuda'):
    pos = y.sum(); neg = len(y) - pos
    scale_pos_weight = min(neg / max(pos, 1), 20.0) if pos > 0 else 1.0
    p = t._params_for(name)
    m = xgb.XGBClassifier(
        device=device, tree_method='hist', objective='binary:logistic',
        scale_pos_weight=scale_pos_weight, max_depth=p['max_depth'],
        learning_rate=p['learning_rate'], n_estimators=p['n_estimators'],
        subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
        min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'],
        reg_lambda=p['reg_lambda'], eval_metric='auc', random_state=42, n_jobs=-1,
    )
    m.fit(t._head_X(X, name), y, sample_weight=sample_weight)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='data/features_v9_fs.npz')
    ap.add_argument('--doclaynet', default='data/doclaynet_feats.npz')
    ap.add_argument('--output', default='data/model_v052_final.json')
    ap.add_argument('--report', default='data/report_v052_final.json')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    X, y, filenames, rot = t.load_data(args.features)
    train_idx, cal_idx, test_idx = t.split_groups(X, y, filenames, rot)

    d = np.load(args.doclaynet, allow_pickle=True)
    Xd = d['features'].astype(np.float64); yd = d['labels'].astype(np.int8)
    print(f'DocLayNet: {len(Xd)} samples')

    models = []
    for i, name in enumerate(t.LABEL_NAMES):
        if name in DOC_TYPE_HEADS:
            Xtr = np.concatenate([X[train_idx], Xd])
            ytr = np.concatenate([y[train_idx, i], yd[:, i]])
            sw = np.concatenate([np.ones(len(train_idx)), np.full(len(Xd), DOCLAYNET_WEIGHT)])
            m = train_one(Xtr, ytr, name, sample_weight=sw, device=args.device)
        else:
            m = train_one(X[train_idx], y[train_idx, i], name, device=args.device)
        models.append(m)
        print(f'  trained {name}: {m.n_estimators} trees')

    P_cal = t.predict(models, X[cal_idx])
    P_test = t.predict(models, X[test_idx])
    P_train = t.predict(models, X[train_idx])

    trait_thresholds = t.select_trait_thresholds(y[cal_idx], P_cal)
    report = {
        'train': t.trait_metrics(y[train_idx], P_train, 'train', trait_thresholds),
        'cal': t.trait_metrics(y[cal_idx], P_cal, 'cal', trait_thresholds),
        'test': t.trait_metrics(y[test_idx], P_test, 'test', trait_thresholds),
        'rotation_train': t.rotation_metrics(X[train_idx], y[train_idx], rot[train_idx], P_train, 'train', filenames[train_idx]),
        'rotation_cal': t.rotation_metrics(X[cal_idx], y[cal_idx], rot[cal_idx], P_cal, 'cal', filenames[cal_idx]),
        'rotation_test': t.rotation_metrics(X[test_idx], y[test_idx], rot[test_idx], P_test, 'test', filenames[test_idx]),
    }
    rows, chosen, calibration = t.calibrate_confidence(
        X[cal_idx], y[cal_idx], rot[cal_idx], P_cal, filenames[cal_idx])
    report['tau_sweep'] = rows
    report['chosen_tau'] = chosen
    report['confidence_calibration'] = calibration
    pinned_tau = float(chosen['tau']) if chosen else 0.90
    report['pinned_tau'] = pinned_tau
    report['label_thresholds'] = trait_thresholds

    t.export_merged_model(models, args.output, pinned_tau, calibration, trait_thresholds)
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report written: {args.report}')

    # held-out DocLayNet test page eval (150 pages, upright) for the README
    tf = np.load('data/doclaynet_test_feats.npz', allow_pickle=True)
    Xt = tf['features']
    Pt = t.predict(models, Xt)
    dl = {}
    for i, name in enumerate(t.LABEL_NAMES[:5]):
        gt = 1.0 if name in ('is_document', 'is_digital') else 0.0
        pred = (Pt[i] >= trait_thresholds.get(name, 0.5)).astype(float)
        dl[f'{name}_acc'] = float((pred == gt).mean())
    json.dump(dl, open('data/doclaynet_report.json', 'w'), indent=2)
    print('DocLayNet held-out:', json.dumps(dl))


if __name__ == '__main__':
    main()
