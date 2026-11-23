"""Verify scientific result conventions without any manuscript dependency."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_analysis as analysis

ROOT = Path(__file__).resolve().parent


def verify():
    truth = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(analysis.roc_auc_score(truth, score), 0.75)
    assert np.isclose(analysis.average_precision_score(truth, score), 5 / 6)
    assert np.isclose(analysis.roc_auc_score(truth, np.ones(4)), 0.5)
    assert np.isclose(analysis.average_precision_score(truth, np.ones(4)), 0.5)
    familiar = np.array([0., 0., 1., 1., 2., 2., 3., 3.])
    for budget in [0.05, 0.1, 0.15, 0.5]:
        threshold = analysis.matched_threshold(familiar, budget)
        assert np.mean(familiar > threshold) <= budget
    groups = np.array(["Normal", "Normal", "Known_Only", "Known_Only"])
    fi, ui = analysis.sample_indices(groups, 3, 80, np.random.default_rng(17))
    assert np.all((groups[fi] == "Normal").sum(axis=1) == 2)
    assert np.all((groups[fi] == "Known_Only").sum(axis=1) == 2)
    sf = np.array([0.2, 0.4, 0.6, 0.8])
    su = np.array([0.3, 0.7, 0.9])
    boot = analysis.boot_fixed(sf, su, 0.5, groups, np.array([True, True, False]),
                               np.array([False, True, False]), fi, ui)
    for index in range(80):
        yy = np.r_[np.zeros(4), np.ones(3)]
        ss = np.r_[sf[fi[index]], su[ui[index]]]
        assert np.isclose(boot["auroc"][index], analysis.roc_auc_score(yy, ss))
        assert np.isclose(boot["aupr"][index], analysis.average_precision_score(yy, ss))
        fpr, tpr, _ = analysis.roc_curve(yy, ss)
        assert np.isclose(boot["fpr95"][index], fpr[tpr >= 0.95].min())
    np.testing.assert_allclose(np.mean([boot["r2n"]] * 3, axis=0), boot["r2n"])
    audit = pd.read_csv(ROOT / "tables/verification_audit.csv")
    assert len(audit) == 192 and audit["pass"].all()
    weights = pd.read_csv(ROOT / "tables/fusion_weights_by_seed.csv")
    baseline = pd.read_csv(ROOT / "tables/table_II_reproduced_by_seed.csv")
    for alpha, method in [(0., "MC Dropout"), (0.5, "SafeFundus fusion"), (1., "Normal-Aware JointEnergy")]:
        np.testing.assert_allclose(weights[weights.alpha_energy == alpha][analysis.METRICS].to_numpy(),
                                   baseline[baseline.method == method][analysis.METRICS].to_numpy(), atol=1e-12)
    counts = pd.read_csv(ROOT / "tables/per_seed_case_counts.csv")
    assert ((counts.unknown_referred_n + counts.r2n_n + counts.forced_known_n) == counts.unknown_n).all()
    assert (counts.explicit_normal_n <= counts.r2n_n).all()
    matched = pd.read_csv(ROOT / "tables/matched_burden_by_seed.csv")
    assert (matched.familiar_referral <= matched.target_test_familiar_burden + 1e-12).all()
    assert matched.groupby(["target_test_familiar_burden", "seed"]).familiar_referred_n.nunique().eq(1).all()
    notebook = json.loads((ROOT / "SafeFundus_Reviewer_Analyses_Kaggle.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"analysis_notebook_cell_{number}", "exec")
    print("PASS: metric examples, paired bootstrap, seed dependence, 192 saved numeric checks, "
          "fusion endpoints, outcome partitions, matched burdens, and analysis notebook syntax.")


if __name__ == "__main__":
    verify()
