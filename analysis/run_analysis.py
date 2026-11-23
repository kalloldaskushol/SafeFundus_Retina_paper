"""Reproduce SafeFundus results and perform reviewer-requested post hoc analyses.

No model training, threshold optimization on unknown labels, or manuscript edits.
See README.txt for interpretation, bootstrap scope, and reproduction commands.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def roc_curve(y, score, drop_intermediate=False):
    """Empirical ROC with tied scores grouped; equivalent thresholds to sklearn."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    return np.r_[0., fp/(len(y)-y.sum())], np.r_[0., tp/y.sum()], np.r_[np.inf, ss[ends]]


def roc_auc_score(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.sum((tpr[1:]+tpr[:-1]) * np.diff(fpr) / 2))


def average_precision_score(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(yy)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * tp / (ends+1)))


def f1_score(truth, pred, average='macro', zero_division=0):
    truth, pred = np.asarray(truth, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp, fn = [(a.sum(axis=0)).astype(float) for a in [truth&pred, ~truth&pred, truth&~pred]]
    if average == 'micro':
        return float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
    den = 2*tp+fp+fn
    return float(np.divide(2*tp, den, out=np.full_like(den, float(zero_division)), where=den!=0).mean())

SEEDS = [42, 123, 2026]
DISEASES = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'MYA', 'ARMD', 'BRVO', 'ODP', 'ODE', 'LS', 'RS', 'CSR', 'CRS']
CLASSES = ['NORMAL'] + DISEASES
METHODS = ['Standard JointEnergy', 'Normal-Aware JointEnergy', 'MC Dropout', 'SafeFundus fusion']
METRICS = ['familiar_referral', 'normal_referral', 'known_referral', 'unknown_referral', 'r2n', 'explicit_normal', 'forced_known', 'auroc', 'aupr', 'fpr95']
PARTS = {'classifier': '06C_SwinTiny', 'energy': '07B_NormalAware', 'mc': '08_MC_', 'fusion': '09_SafeFundus', 'external': '10B_RFMiD2'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def extract_evidence(archive, dest):
    """Only export small CSV/JSON evidence; do not extract model weights."""
    manifest = []
    with zipfile.ZipFile(archive) as outer:
        for role, marker in PARTS.items():
            candidates = [x for x in outer.namelist() if marker in x and x.endswith('.zip')]
            assert len(candidates) == 1, (role, candidates)
            member = candidates[0]
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                for info in inner.infolist():
                    if not info.filename.endswith(('.csv', '.json')):
                        continue
                    path = (dest / role / info.filename).resolve()
                    assert path.is_relative_to(dest.resolve())
                    raw = inner.read(info)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    manifest.append({'role': role, 'outer_member': member, 'inner_member': info.filename,
                                     'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return manifest


def load(root, role, path):
    return pd.read_csv(root / role / path)


def aligned(frame, ids=None):
    frame = frame.copy()
    frame['ID'] = frame['ID'].astype(str)
    assert not frame['ID'].duplicated().any(), 'Duplicate image ID'
    frame = frame.set_index('ID', drop=False)
    if ids is not None:
        assert set(frame.index) == set(ids), 'Image identity mismatch'
        return frame.loc[ids]
    return frame.sort_index(key=lambda index: index.astype(int))


def unknown_decisions(frame, thresholds):
    pred = frame[['prob_' + c for c in DISEASES]].to_numpy(float) >= np.array([thresholds[c] for c in DISEASES])
    no_disease = ~pred.any(axis=1)
    explicit = frame['prob_NORMAL'].to_numpy(float) >= thresholds['NORMAL']
    return no_disease, explicit


def detector_metrics(familiar, unknown, threshold, groups, no_disease, explicit):
    fr, ur = familiar > threshold, unknown > threshold
    normal, known = groups == 'Normal', groups == 'Known_Only'
    y = np.r_[np.zeros(len(familiar)), np.ones(len(unknown))]
    score = np.r_[familiar, unknown]
    fpr, tpr, _ = roc_curve(y, score)
    return dict(zip(METRICS, [fr.mean(), fr[normal].mean(), fr[known].mean(), ur.mean(),
                ((~ur) & no_disease).mean(), ((~ur) & no_disease & explicit).mean(),
                ((~ur) & (~no_disease)).mean(), roc_auc_score(y, score), average_precision_score(y, score),
                fpr[tpr >= .95].min()]))


def summarize(frame, grouping, metrics):
    rows = []
    for key, group in frame.groupby(grouping, sort=False, dropna=False):
        if len(grouping) == 1:
            key = (key,) if not isinstance(key, tuple) else key
        for metric in metrics:
            values = group[metric].to_numpy(float)
            finite = np.isfinite(values)
            rows.append(dict(zip(grouping, key)) | {'metric': metric, 'mean': np.nanmean(values) if finite.any() else np.nan,
                         'seed_sd': np.nanstd(values, ddof=1) if finite.sum() > 1 else (0. if finite.any() else np.nan)})
    return pd.DataFrame(rows)


def sample_indices(groups, n_unknown, reps, rng):
    """Shared case resamples for all fixed models; familiar case mix held fixed."""
    chunks = []
    for label in ['Normal', 'Known_Only']:
        indices = np.flatnonzero(groups == label)
        assert len(indices)
        chunks.append(indices[rng.integers(0, len(indices), (reps, len(indices)))])
    return np.concatenate(chunks, axis=1), rng.integers(0, n_unknown, (reps, n_unknown))


def weights(indices, n):
    out = np.zeros((len(indices), n), dtype=np.float64)
    np.add.at(out, (np.arange(len(indices))[:, None], indices), 1)
    return out


def boot_fixed(sf, su, threshold, groups, no_disease, explicit, fi, ui):
    fr, ur = sf > threshold, su > threshold
    res = {}
    boot_fr, boot_ur = fr[fi], ur[ui]
    res['familiar_referral'] = boot_fr.mean(axis=1)
    sampled_groups = groups[fi]
    for label, metric in [('Normal', 'normal_referral'), ('Known_Only', 'known_referral')]:
        mask = sampled_groups == label
        res[metric] = (boot_fr & mask).sum(axis=1) / mask.sum(axis=1)
    res['unknown_referral'] = boot_ur.mean(axis=1)
    res['r2n'] = ((~boot_ur) & no_disease[ui]).mean(axis=1)
    res['explicit_normal'] = ((~boot_ur) & no_disease[ui] & explicit[ui]).mean(axis=1)
    res['forced_known'] = ((~boot_ur) & (~no_disease[ui])).mean(axis=1)
    pair = (su[:, None] > sf[None, :]).astype(float) + .5 * (su[:, None] == sf[None, :])
    auc, aupr, fpr95 = [], [], []
    combined = np.r_[sf, su]
    labels = np.r_[np.zeros(len(sf)), np.ones(len(su))]
    order = np.argsort(-combined, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(combined[order])), len(combined)-1]
    for start in range(0, len(fi), 500):
        wf, wu = weights(fi[start:start+500], len(sf)), weights(ui[start:start+500], len(su))
        auc.extend(((wu @ pair) * wf).sum(axis=1) / (len(sf) * len(su)))
        combined_w = np.c_[wf, wu][:, order]
        total = np.cumsum(combined_w, axis=1)[:, ends]
        positive = np.cumsum(combined_w * labels[order][None, :], axis=1)[:, ends]
        precision = np.divide(positive, total, out=np.zeros_like(total), where=total!=0)
        aupr.extend((np.diff(np.c_[np.zeros(len(positive)), positive], axis=1) / len(su) * precision).sum(axis=1))
        fpr95.extend(np.where(positive >= .95*len(su), (total-positive)/len(sf), np.inf).min(axis=1))
    res['auroc'] = np.asarray(auc)
    res['aupr'] = np.asarray(aupr)
    res['fpr95'] = np.asarray(fpr95)
    return res


def matched_threshold(sf, q):
    k = math.floor(q * len(sf) + 1e-12)
    return float(np.partition(sf, len(sf) - k - 1)[len(sf) - k - 1])


def boot_matched(sf, su, q, no_disease, fi, ui):
    n, k = len(sf), math.floor(q * len(sf) + 1e-12)
    sampled_f = sf[fi]
    threshold = np.partition(sampled_f, n-k-1, axis=1)[:, n-k-1]
    uref = su[ui] > threshold[:, None]
    return {'familiar_referral': (sampled_f > threshold[:, None]).mean(axis=1),
            'unknown_referral': uref.mean(axis=1), 'r2n': ((~uref) & no_disease[ui]).mean(axis=1)}


def confidence_rows(label, boot_by_seed):
    rows = []
    common = set.intersection(*(set(x) for x in boot_by_seed))
    for metric in sorted(common):
        for level, data in [(str(seed), b[metric]) for seed, b in zip(SEEDS, boot_by_seed)] + [('mean_of_three_seeds', np.mean([b[metric] for b in boot_by_seed], axis=0))]:
            lower, upper = np.quantile(data, [.025, .975])
            rows.append(label | {'seed_or_estimand': level, 'metric': metric, 'ci95_lower': lower, 'ci95_upper': upper})
    return rows


def main(archive: Path, output: Path, reps=10000):
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / 'evidence'
    tables = output / 'tables'
    figures = output / 'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    manifest = extract_evidence(archive, evidence)
    pd.DataFrame(manifest).to_csv(output / 'input_manifest.csv', index=False)
    print('Evidence extracted; reconstructing original results.', flush=True)
    checks, internal_rows, external_rows, counts, class_rows = [], [], [], [], []
    bundles = {}
    for seed in SEEDS:
        base, final, energy = {}, {}, {}
        for split, suffix in [('validation', 'validation'), ('familiar', 'familiar_test'), ('unknown', 'unknown_test')]:
            base[split] = aligned(load(evidence, 'classifier', f'seed_{seed}/predictions/{suffix}_logits_predictions_15class.csv'))
            final[split] = aligned(load(evidence, 'fusion', f'per_seed/seed_{seed}/{split}_final_scores.csv'), base[split].index)
            energy[split] = aligned(load(evidence, 'energy', f'per_seed/seed_{seed}/{suffix if split != "validation" else "validation"}_normalaware_energy_scores.csv'), base[split].index)
        assert [len(base[x]) for x in ['validation', 'familiar', 'unknown']] == [560, 571, 32]
        thresholds = load(evidence, 'classifier', f'seed_{seed}/tables/validation_thresholds_15class.csv').set_index('Class')['Threshold'].to_dict()
        no_disease, explicit = unknown_decisions(base['unknown'], thresholds)
        groups = base['familiar']['Group'].to_numpy(str)
        assert set(groups) == {'Normal', 'Known_Only'}
        e_val = energy['validation']
        evidence15_val = e_val['Normal_Evidence'].to_numpy() + e_val['Disease_Evidence'].to_numpy()
        raw_na = e_val['NormalAware_Unknownness'].to_numpy()
        model_scores = {}
        for split in base:
            ev = energy[split]
            standard = (evidence15_val.mean() - ev['Normal_Evidence'].to_numpy() - ev['Disease_Evidence'].to_numpy()) / evidence15_val.std(ddof=0)
            j = final[split]['JointEnergy_Normalized'].to_numpy()
            u = final[split]['MC_Dropout_Normalized'].to_numpy()
            fusion = .5*j + .5*u
            np.testing.assert_allclose(fusion, final[split]['Fusion_R'], atol=1e-12)
            np.testing.assert_allclose((ev['NormalAware_Unknownness'].to_numpy()-raw_na.mean()) / raw_na.std(ddof=0), j, atol=1e-12)
            reconstructed15 = np.logaddexp(0., base[split][['logit_'+c for c in CLASSES]].to_numpy(float)).sum(axis=1)
            np.testing.assert_allclose(reconstructed15, ev['Normal_Evidence'] + ev['Disease_Evidence'], atol=2e-12)
            model_scores[split] = dict(zip(METHODS, [standard, j, u, fusion]))
        for method in METHODS:
            sv, sf, su = [model_scores[s][method] for s in ['validation', 'familiar', 'unknown']]
            threshold = float(np.quantile(sv, .9, method='linear'))
            result = detector_metrics(sf, su, threshold, groups, no_disease, explicit)
            internal_rows.append({'seed': seed, 'method': method, 'threshold': threshold, 'validation_referral': (sv > threshold).mean()} | result)
            counts.append({'dataset': 'RFMiD', 'seed': seed, 'method': method, 'unknown_n': len(su),
                           'familiar_n': len(sf), 'familiar_referred_n': int((sf > threshold).sum()),
                           'unknown_referred_n': int((su > threshold).sum()),
                           'r2n_n': int(((su <= threshold) & no_disease).sum()),
                           'explicit_normal_n': int(((su <= threshold) & no_disease & explicit).sum()),
                           'forced_known_n': int(((su <= threshold) & ~no_disease).sum())})
        baseline = {'seed': seed, 'method': 'Classifier only', 'threshold': np.nan, 'validation_referral': 0.,
                    'familiar_referral': 0., 'normal_referral': 0., 'known_referral': 0., 'unknown_referral': 0.,
                    'r2n': no_disease.mean(), 'explicit_normal': (no_disease & explicit).mean(),
                    'forced_known': (~no_disease).mean(), 'auroc': np.nan, 'aupr': np.nan, 'fpr95': np.nan}
        internal_rows.append(baseline)
        counts.append({'dataset': 'RFMiD', 'seed': seed, 'method': 'Classifier only', 'unknown_n': 32, 'familiar_n': 571,
                       'familiar_referred_n': 0, 'unknown_referred_n': 0, 'r2n_n': int(no_disease.sum()),
                       'explicit_normal_n': int((no_disease & explicit).sum()), 'forced_known_n': int((~no_disease).sum())})
        # External results use only original frozen parameters, never recalibrated thresholds.
        ext = {s: aligned(load(evidence, 'external', f'per_seed/seed_{seed}/rfmid2_{s}_predictions.csv')) for s in ['familiar', 'unknown', 'mixed']}
        assert [len(ext[s]) for s in ['familiar', 'unknown', 'mixed']] == [98, 50, 19]
        params = json.loads((evidence/'fusion'/f'per_seed/seed_{seed}/fusion_validation_parameters.json').read_text())
        ef, eu = ext['familiar'], ext['unknown']
        eth = params['fusion_primary_threshold']
        end = eu['Num_Known_Diseases_Predicted'].to_numpy() == 0
        eexplicit = eu['prob_cal_NORMAL'].to_numpy() >= params['calibrated_thresholds']['NORMAL']
        exgroups = ef['External_Group'].to_numpy(str)
        eresult = detector_metrics(ef['Fusion_R'].to_numpy(), eu['Fusion_R'].to_numpy(), eth, exgroups, end, eexplicit)
        eresult['mixed_referral'] = (ext['mixed']['Fusion_R'].to_numpy() > eth).mean()
        external_rows.append({'seed': seed, 'method': 'Frozen SafeFundus'} | eresult)
        for field, actual in [('Referral', eu['Fusion_R'].to_numpy() > eth), ('Rare_to_Normal_Error', (eu['Fusion_R'].to_numpy() <= eth) & end),
                              ('Explicit_Normal_False_Reassurance', (eu['Fusion_R'].to_numpy() <= eth) & end & eexplicit),
                              ('Forced_Into_Known', (eu['Fusion_R'].to_numpy() <= eth) & (~end))]:
            np.testing.assert_array_equal(eu[field].to_numpy(int), actual.astype(int))
        counts.append({'dataset': 'RFMiD 2.0', 'seed': seed, 'method': 'Frozen SafeFundus', 'unknown_n': 50, 'familiar_n': 98,
                       'familiar_referred_n': int(ef['Referral'].sum()), 'unknown_referred_n': int(eu['Referral'].sum()),
                       'r2n_n': int(eu['Rare_to_Normal_Error'].sum()), 'explicit_normal_n': int(eu['Explicit_Normal_False_Reassurance'].sum()),
                       'forced_known_n': int(eu['Forced_Into_Known'].sum())})
        truth = ef[['true_'+c for c in DISEASES]].to_numpy(int)
        probs = ef[['prob_cal_'+c for c in DISEASES]].to_numpy(float)
        predicted = probs >= np.array([params['calibrated_thresholds'][c] for c in DISEASES])
        per_class = load(evidence, 'external', f'per_seed/seed_{seed}/external_per_class_disease_metrics.csv').set_index('Disease').loc[DISEASES]
        f1s = []
        for i, c in enumerate(DISEASES):
            tp = int((truth[:, i] & predicted[:, i]).sum())
            fp = int(((1-truth[:, i]) & predicted[:, i]).sum())
            fn = int((truth[:, i] & ~predicted[:, i]).sum())
            assert [tp, fp, fn] == [int(per_class.loc[c, k]) for k in ['TP', 'FP', 'FN']]
            f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        positives = truth.sum(axis=0) > 0
        cresult = {'seed': seed, 'familiar_n': len(ef), 'classes_all': 14, 'classes_with_positives': int(positives.sum()),
                   'macro_f1_from_confusion': np.mean(f1s), 'macro_f1_from_probabilities': f1_score(truth, predicted, average='macro', zero_division=0),
                   'macro_f1_positive_classes': np.mean(np.array(f1s)[positives]), 'micro_f1': f1_score(truth, predicted, average='micro', zero_division=0),
                   'map_positive_classes': np.mean([average_precision_score(truth[:, i], probs[:, i]) for i in np.flatnonzero(positives)]),
                   'macro_auroc_positive_classes': np.mean([roc_auc_score(truth[:, i], probs[:, i]) for i in np.flatnonzero(positives)])}
        class_rows.append(cresult)
        bundles[seed] = {'scores': model_scores, 'groups': groups, 'nd': no_disease, 'explicit': explicit,
                         'ids_f': base['familiar']['ID'].tolist(), 'ids_u': base['unknown']['ID'].tolist(),
                         'external': ext, 'ext_groups': exgroups, 'ext_nd': end, 'ext_explicit': eexplicit,
                         'ext_threshold': eth, 'ext_truth': truth, 'ext_pred': predicted}
    internal, external = pd.DataFrame(internal_rows), pd.DataFrame(external_rows)
    # Verify full-precision archived result records independently of rounded manuscript entries.
    expected = load(evidence, 'fusion', 'tables/ablation_results_by_seed.csv')
    mapping = {'familiar_referral':'familiar_test_false_referral_rate', 'normal_referral':'normal_test_false_referral_rate',
               'known_referral':'known_only_test_false_referral_rate', 'unknown_referral':'unknown_referral_sensitivity',
               'r2n':'post_referral_rare_to_normal_rate', 'explicit_normal':'post_referral_explicit_normal_false_reassurance',
               'forced_known':'post_referral_forced_known_rate', 'auroc':'unknown_auroc', 'aupr':'unknown_aupr', 'fpr95':'fpr_at_95_tpr'}
    name_map = {'Classifier only':'Classifier_Only', 'Normal-Aware JointEnergy':'JointEnergy', 'MC Dropout':'MC_Dropout', 'SafeFundus fusion':'Fusion_50_50'}
    def compare(dataset, seed, method, metric, actual, expected, tol=2e-6):
        if not np.isfinite(expected):
            assert not np.isfinite(actual)
            return
        error = abs(float(actual)-float(expected))
        checks.append({'dataset': dataset, 'seed': seed, 'method': method, 'metric': metric, 'actual': actual,
                       'archived': expected, 'absolute_error': error, 'tolerance': tol, 'pass': error <= tol})
    for _, row in internal.iterrows():
        source = load(evidence, 'energy', 'tables/jointenergy_method_results_by_seed.csv') if row['method']=='Standard JointEnergy' else expected
        label = 'Standard15_Sum' if row['method']=='Standard JointEnergy' else name_map[row['method']]
        saved = source[(source.seed==row.seed)&(source.method==label)].iloc[0]
        for metric, field in mapping.items():
            compare('RFMiD', row.seed, row['method'], metric, row[metric], saved[field])
    exp_ext = load(evidence, 'external', 'tables/rfmid2_external_results_by_seed.csv').set_index('seed')
    emap = {'familiar_referral':'familiar_false_referral_rate','normal_referral':'normal_false_referral_rate','known_referral':'known_only_false_referral_rate',
            'unknown_referral':'unknown_referral_sensitivity','r2n':'rare_to_normal_error_rate','explicit_normal':'explicit_normal_false_reassurance_rate',
            'forced_known':'unknown_forced_into_known_rate','auroc':'unknown_auroc','aupr':'unknown_aupr','fpr95':'fpr_at_95_tpr','mixed_referral':'mixed_referral_rate'}
    for _, row in external.iterrows():
        for metric, field in emap.items():
            compare('RFMiD 2.0', row.seed, 'Frozen SafeFundus', metric, row[metric], exp_ext.loc[row.seed, field])
    cmapping = {'macro_f1_from_confusion':'disease_macro_f1_all14','macro_f1_from_probabilities':'disease_macro_f1_all14',
                'macro_f1_positive_classes':'disease_macro_f1_evaluable','micro_f1':'disease_micro_f1','map_positive_classes':'disease_mAP','macro_auroc_positive_classes':'disease_macro_auroc'}
    for row in class_rows:
        for metric, field in cmapping.items():
            compare('RFMiD 2.0 classification', row['seed'], 'Frozen SafeFundus', metric, row[metric], exp_ext.loc[row['seed'], field], tol=2e-6)
    checks = pd.DataFrame(checks)
    checks.to_csv(tables/'verification_audit.csv', index=False)
    assert checks['pass'].all(), checks.loc[~checks['pass']].to_dict('records')
    for seed in SEEDS[1:]:
        for key in ['ids_f','ids_u']:
            assert bundles[seed][key] == bundles[SEEDS[0]][key]
        for split in ['familiar','unknown','mixed']:
            assert bundles[seed]['external'][split]['ID'].tolist() == bundles[SEEDS[0]]['external'][split]['ID'].tolist()
    print(f'All {len(checks)} numerical audit checks passed.', flush=True)
    internal.to_csv(tables/'table_II_reproduced_by_seed.csv', index=False)
    external.to_csv(tables/'table_III_reproduced_by_seed.csv', index=False)
    summarize(internal, ['method'], METRICS).to_csv(tables/'table_II_reproduced_mean_sd.csv', index=False)
    summarize(external, ['method'], METRICS+['mixed_referral']).to_csv(tables/'table_III_reproduced_mean_sd.csv', index=False)
    pd.DataFrame(counts).to_csv(tables/'per_seed_case_counts.csv', index=False)
    pd.DataFrame(class_rows).to_csv(tables/'external_classification_recheck.csv', index=False)
    rng = np.random.default_rng(20261121)
    ref = bundles[42]
    fi, ui = sample_indices(ref['groups'], 32, reps, rng)
    efi, eui = sample_indices(ref['ext_groups'], 50, reps, rng)
    mixed_i = rng.integers(0, 19, (reps, 19))
    np.savez_compressed(output/'bootstrap_indices.npz', internal_familiar=fi, internal_unknown=ui,
                        external_familiar=efi, external_unknown=eui, external_mixed=mixed_i)
    boot, ci_rows = {}, []
    for method in ['Classifier only']+METHODS:
        boot[method] = []
        for seed in SEEDS:
            d = bundles[seed]
            if method == 'Classifier only':
                values = {'r2n': d['nd'][ui].mean(axis=1), 'unknown_referral': np.zeros(reps), 'familiar_referral': np.zeros(reps)}
            else:
                sv, sf, su = [d['scores'][s][method] for s in ['validation','familiar','unknown']]
                values = boot_fixed(sf, su, np.quantile(sv,.9), d['groups'], d['nd'], d['explicit'], fi, ui)
            boot[method].append(values)
        ci_rows.extend(confidence_rows({'dataset':'RFMiD','method':method}, boot[method]))
    eboots = []
    for seed in SEEDS:
        d = bundles[seed]
        ef, eu = d['external']['familiar'], d['external']['unknown']
        values = boot_fixed(ef['Fusion_R'].to_numpy(), eu['Fusion_R'].to_numpy(), d['ext_threshold'], d['ext_groups'], d['ext_nd'], d['ext_explicit'], efi, eui)
        values['mixed_referral'] = d['external']['mixed']['Referral'].to_numpy()[mixed_i].mean(axis=1)
        wf = weights(efi, len(ef))
        truth, pred = d['ext_truth'].astype(bool), d['ext_pred']
        tp, fp, fn = wf@(truth&pred), wf@(~truth&pred), wf@(truth&~pred)
        den = 2*tp+fp+fn
        values['macro_f1_all14'] = np.divide(2*tp, den, out=np.zeros_like(den), where=den!=0).mean(axis=1)
        eboots.append(values)
    ci_rows.extend(confidence_rows({'dataset':'RFMiD 2.0','method':'Frozen SafeFundus'}, eboots))
    pd.DataFrame(ci_rows).to_csv(tables/'paired_bootstrap_confidence_intervals.csv', index=False)
    diffs = []
    for method in ['Classifier only']+METHODS[:-1]:
        for metric in ['r2n','unknown_referral','auroc']:
            if metric not in boot[method][0]:
                continue
            values = np.mean([boot['SafeFundus fusion'][i][metric]-boot[method][i][metric] for i in range(3)], axis=0)
            p = internal[internal.method=='SafeFundus fusion'][metric].mean()-internal[internal.method==method][metric].mean()
            lo, hi = np.quantile(values,[.025,.975])
            diffs.append({'comparison':'fusion minus '+method,'metric':metric,'difference':p,'ci95_lower':lo,'ci95_upper':hi})
    pd.DataFrame(diffs).to_csv(tables/'paired_method_differences_original_operating_points.csv', index=False)
    print('Paired bootstrap completed; evaluating matched burden and fusion weights.', flush=True)
    matched, matched_ci, weight_rows, weight_ci, curve_rows, overlap = [], [], [], [], [], []
    matched_boots = {}
    for q in [.05,.10,.15]:
        for method in METHODS:
            bvalues=[]
            for seed in SEEDS:
                d=bundles[seed]; sf=d['scores']['familiar'][method]; su=d['scores']['unknown'][method]
                threshold=matched_threshold(sf,q)
                result=detector_metrics(sf,su,threshold,d['groups'],d['nd'],d['explicit'])
                matched.append({'target_test_familiar_burden':q,'seed':seed,'method':method,'retrospective_test_threshold':threshold,
                                'familiar_referred_n':int((sf>threshold).sum()),'unknown_referred_n':int((su>threshold).sum()),'r2n_n':int(((su<=threshold)&d['nd']).sum())}|result)
                bvalues.append(boot_matched(sf,su,q,d['nd'],fi,ui))
            matched_ci.extend(confidence_rows({'target_test_familiar_burden':q,'method':method},bvalues))
            matched_boots[(q,method)] = bvalues
    grid=np.linspace(0,1,1001)
    for seed in SEEDS:
        d=bundles[seed]
        for method in METHODS:
            sf=d['scores']['familiar'][method]; su=d['scores']['unknown'][method]
            y=np.r_[np.zeros(len(sf)),np.ones(len(su))]
            fpr,tpr,_=roc_curve(y,np.r_[sf,su],drop_intermediate=False)
            for q in grid:
                curve_rows.append({'seed':seed,'method':method,'test_familiar_burden':q,'unknown_referral':tpr[fpr<=q+1e-12].max()})
        j=d['scores']['unknown']['Normal-Aware JointEnergy']; u=d['scores']['unknown']['MC Dropout']
        jr=j>np.quantile(d['scores']['validation']['Normal-Aware JointEnergy'],.9)
        ur=u>np.quantile(d['scores']['validation']['MC Dropout'],.9)
        rr=d['scores']['unknown']['SafeFundus fusion']>np.quantile(d['scores']['validation']['SafeFundus fusion'],.9)
        overlap.append({'seed':seed,'unknown_n':32,'energy_referrals':int(jr.sum()),'mc_referrals':int(ur.sum()),'intersection':int((jr&ur).sum()),
                        'energy_only':int((jr&~ur).sum()),'mc_only':int((ur&~jr).sum()),'union':int((jr|ur).sum()),'fusion_referrals':int(rr.sum()),
                        'fusion_outside_union':int((rr&~(jr|ur)).sum()),'union_not_fusion':int(((jr|ur)&~rr).sum())})
    for alpha in [0.,.25,.5,.75,1.]:
        bvalues=[]
        for seed in SEEDS:
            d=bundles[seed]; score={s:alpha*d['scores'][s]['Normal-Aware JointEnergy']+(1-alpha)*d['scores'][s]['MC Dropout'] for s in ['validation','familiar','unknown']}
            threshold=float(np.quantile(score['validation'],.9))
            result=detector_metrics(score['familiar'],score['unknown'],threshold,d['groups'],d['nd'],d['explicit'])
            weight_rows.append({'alpha_energy':alpha,'seed':seed,'threshold_validation_only':threshold,'validation_referral':(score['validation']>threshold).mean()}|result)
            bvalues.append(boot_fixed(score['familiar'],score['unknown'],threshold,d['groups'],d['nd'],d['explicit'],fi,ui))
        weight_ci.extend(confidence_rows({'alpha_energy':alpha},bvalues))
    matched=pd.DataFrame(matched); weighted=pd.DataFrame(weight_rows)
    matched_summary=summarize(matched,['target_test_familiar_burden','method'],METRICS)
    matched_summary=matched_summary.merge(pd.DataFrame(matched_ci).query('seed_or_estimand=="mean_of_three_seeds"').drop(columns='seed_or_estimand'),on=['target_test_familiar_burden','method','metric'],how='left')
    weight_summary=summarize(weighted,['alpha_energy'],METRICS)
    weight_summary=weight_summary.merge(pd.DataFrame(weight_ci).query('seed_or_estimand=="mean_of_three_seeds"').drop(columns='seed_or_estimand'),on=['alpha_energy','metric'],how='left')
    matched.to_csv(tables/'matched_burden_by_seed.csv',index=False)
    matched_summary.to_csv(tables/'matched_burden_mean_sd_ci.csv',index=False)
    weighted.to_csv(tables/'fusion_weights_by_seed.csv',index=False)
    weight_summary.to_csv(tables/'fusion_weights_mean_sd_ci.csv',index=False)
    pd.DataFrame(matched_ci).to_csv(tables/'matched_burden_all_confidence_intervals.csv',index=False)
    pd.DataFrame(weight_ci).to_csv(tables/'fusion_weights_all_confidence_intervals.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(tables/'referral_curves_by_seed.csv',index=False)
    pd.DataFrame(overlap).to_csv(tables/'referral_overlap.csv',index=False)
    mdiffs=[]
    for q in [.05,.10,.15]:
        for method in METHODS[:-1]:
            for metric in ['unknown_referral','r2n']:
                values=np.mean([matched_boots[(q,'SafeFundus fusion')][i][metric]-matched_boots[(q,method)][i][metric] for i in range(3)],axis=0)
                a=matched[(matched.target_test_familiar_burden==q)&(matched.method=='SafeFundus fusion')][metric].mean()
                b=matched[(matched.target_test_familiar_burden==q)&(matched.method==method)][metric].mean()
                lo,hi=np.quantile(values,[.025,.975])
                mdiffs.append({'target_test_familiar_burden':q,'comparison':'fusion minus '+method,'metric':metric,'difference':a-b,'ci95_lower':lo,'ci95_upper':hi})
    pd.DataFrame(mdiffs).to_csv(tables/'paired_method_differences_matched_burden.csv',index=False)
    # Endpoints and original equal weighting must reproduce the original methods.
    for alpha,method in [(0.,'MC Dropout'),(.5,'SafeFundus fusion'),(1.,'Normal-Aware JointEnergy')]:
        for seed in SEEDS:
            a=weighted[(weighted.alpha_energy==alpha)&(weighted.seed==seed)].iloc[0]
            b=internal[(internal.method==method)&(internal.seed==seed)].iloc[0]
            np.testing.assert_allclose(a[METRICS].astype(float),b[METRICS].astype(float),atol=1e-12)
    assert (matched['familiar_referral']<=matched['target_test_familiar_burden']+1e-12).all()
    assert np.allclose(external['unknown_referral']+external['forced_known']+external['r2n'],1.)
    assert (external['explicit_normal']<=external['r2n']).all()
    config={'archive_name':archive.name,'archive_sha256':sha256(archive),'bootstrap_replicates':reps,'bootstrap_rng_seed':20261121,
            'model_seeds':SEEDS,'bootstrap_estimand':'Mean metric of the three fixed models; paired image sampling shared across seeds and methods.',
            'bootstrap_familiar_sampling':'Stratified by Normal/Known_Only with observed subgroup sizes fixed.',
            'bootstrap_unknown_sampling':'32 internal or 50 external unique image IDs sampled with replacement; models are not independent cohorts.',
            'bootstrap_interval':'Percentile 2.5/97.5%; conditional on models and original validation calibration; does not capture retraining or calibration uncertainty.',
            'matched_threshold':'Retrospective familiar-test order statistic with strict score > threshold; at most floor(q*n) familiar referrals. Unknown labels never select threshold.',
            'matched_bootstrap':'Recompute retrospective threshold in each familiar-test bootstrap resample and evaluate paired unknown resample.',
            'fusion_analysis':'Post hoc sensitivity only; original equal weight and original frozen external evaluation retained.',
            'validation_threshold':'np.quantile(validation_score,0.90,method=linear), strict > comparison.',
            'numerical_audit_checks':len(checks),'all_audit_checks_passed':bool(checks['pass'].all()),'python':platform.python_version(),
            'numpy':np.__version__,'pandas':pd.__version__}
    save_json(output/'analysis_config.json',config)
    make_figure(pd.DataFrame(curve_rows),weighted,weight_summary,figures)
    write_compact_tables(matched_summary,weight_summary,tables)
    print('Analysis complete:',output,flush=True)
    return config


def make_figure(curves,weighted,summary,figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':9,'axes.titlesize':10})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.15))
    colors=['#777777','#0072B2','#D55E00','#009E73']
    short=['Standard JointEnergy','Normal-Aware JointEnergy','MC Dropout','Fixed 0.5/0.5 fusion']
    for method,color,label,style in zip(METHODS,colors,short,[':','--','-.','-']):
        data=curves[curves.method==method].groupby('test_familiar_burden').unknown_referral.mean()
        mask=data.index<=.25
        axes[0].plot(data.index[mask]*100,data.values[mask]*100,label=label,color=color,ls=style,lw=1.6,drawstyle='steps-post')
    for q in [5,10,15]: axes[0].axvline(q,color='#BBBBBB',lw=.6,zorder=0)
    axes[0].set(xlim=(0,25),ylim=(0,100),xlabel='Familiar test referral burden (%)',ylabel='Unknown referral sensitivity (%)',title='(a) Matched burden (retrospective)')
    axes[0].legend(loc='lower right',fontsize=7,frameon=False)
    for metric,color,label,marker in [('unknown_referral','#0072B2','Unknown referral','o'),('r2n','#D55E00','Rare-to-Normal Error','s'),('familiar_referral','#777777','Familiar referral','^')]:
        sub=summary[summary.metric==metric].sort_values('alpha_energy')
        axes[1].plot(sub.alpha_energy,sub['mean']*100,color=color,marker=marker,label=label,lw=1.5,ms=4)
    axes[1].axvline(.5,color='#555555',ls=':',lw=1)
    axes[1].text(.52,95,'Original weight',fontsize=7,color='#555555')
    axes[1].set(xlim=(-.03,1.03),ylim=(0,100),xticks=[0,.25,.5,.75,1],xlabel=r'Energy weight $\alpha$ (MC: $1-\alpha$)',ylabel='Rate (%)',title='(b) Validation-calibrated weight sweep')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0, .88),fontsize=7,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#DDDDDD',lw=.5)
    fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.88,wspace=.34)
    fig.savefig(figures/'reviewer_analyses_two_panel.pdf')
    fig.savefig(figures/'reviewer_analyses_two_panel.png',dpi=300)
    plt.close(fig)


def write_compact_tables(matched,weighted,tables):
    def display(row,percent=True):
        m,s=float(row['mean']),float(row['seed_sd'])
        scale=100 if percent else 1
        return f'{m*scale:.2f} +/- {s*scale:.2f}' if percent else f'{m:.4f} +/- {s:.4f}'
    rows=[]
    for (q,method),group in matched.groupby(['target_test_familiar_burden','method'],sort=False):
        d=group.set_index('metric')
        rows.append({'Nominal test burden (%)':q*100,'Method':method,'Achieved familiar referral (%)':display(d.loc['familiar_referral']),
                     'Unknown referral (%)':display(d.loc['unknown_referral']),'R2N (%)':display(d.loc['r2n'])})
    pd.DataFrame(rows).to_csv(tables/'matched_burden_readable.csv',index=False)
    rows=[]
    for alpha,group in weighted.groupby('alpha_energy',sort=False):
        d=group.set_index('metric')
        rows.append({'Energy weight':alpha,'MC weight':1-alpha,'Familiar referral (%)':display(d.loc['familiar_referral']),
                     'Unknown referral (%)':display(d.loc['unknown_referral']),'R2N (%)':display(d.loc['r2n']),'AUROC':display(d.loc['auroc'],False)})
    pd.DataFrame(rows).to_csv(tables/'fusion_weights_readable.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    main(args.archive,args.output,args.bootstrap)
