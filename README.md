# SafeFundus research code and saved outputs

This repository contains the research notebooks, trained checkpoints, prediction
outputs, evaluation tables, and scientific plots for SafeFundus.

It does **not** contain the manuscript, LaTeX source, paper drafts, personal
writing, reviewer letters, reference papers, fundus images, or source datasets.
Prediction CSVs contain model outputs and evaluation labels/IDs; they are saved
evaluation artifacts, not copies of the original image datasets.

## Contents

- `notebooks/`: the 12 original Kaggle notebooks, with their existing outputs.
- `outputs/`: unpacked saved run outputs, grouped by experiment.
- `analysis/`: post hoc comparison/bootstrap code and its saved numerical outputs.
- `tools/`: small utilities for reproducing the analysis from the saved outputs.
- `provenance/`: file hashes and the original-to-repository artifact mapping.

Training/inference notebooks are preserved as supplied. Their technical Markdown
cells explain the experiments; no manuscript sections have been added. The
ResNet50 notebook is present, but the supplied archive has no separate saved
ResNet50 output bundle. No missing results are invented.

## Download

Model checkpoints use Git LFS. Install Git LFS before cloning, or run `git lfs pull`
after cloning to fetch the actual checkpoint files rather than pointer files.
The checkpoints require about 1.26 GB of LFS storage.

```sh
git lfs install
git clone https://github.com/kalloldaskushol/SafeFundus_Retina_paper.git
cd SafeFundus_Retina_paper
git lfs pull
```

Only load the PyTorch checkpoints if you trust their source. Some checkpoint
formats can use Python pickle internally.

## Training and inference order

Run the numbered notebooks in `notebooks/` using the input/output attachment
instructions inside each notebook. The original environment was Kaggle; image
datasets must be obtained separately and attached to your own environment.

The progression is backbone screening (01-05), classifier/loss experiments
(06/06B), the explicit-NORMAL classifier (06C), Normal-Aware JointEnergy (07B),
MC Dropout (08), fusion/calibration (09), and frozen external validation (10B).
Do not use held-out unknown test labels to choose model or fusion parameters.

The root `requirements.txt` lists training dependencies without claiming to
reproduce an exact Kaggle environment. The notebooks record available execution
details and remain the source for experiment-specific settings.

## Reproduce the post hoc analyses without retraining

Python 3.10+ is recommended. Only the small CSV/JSON saved outputs are needed;
neither the checkpoints nor the original image datasets are required.

```sh
python -m pip install -r analysis/requirements.txt
python tools/build_analysis_archive.py --destination .local/analysis_inputs.zip
python analysis/run_analysis.py --archive .local/analysis_inputs.zip --output .local/reproduced_analysis --bootstrap 10000
python analysis/verify_results.py
```

For a full 10,000-resample check of the self-contained notebook against every
saved analysis CSV, also run `python analysis/smoke_notebook.py`. Its temporary
archives, reproduced files and validation report stay under ignored `.local/`.

The temporary input archive is reconstructed from saved outputs already in this
repository. It is not the original 1.17 GB outer archive, so its archive hash will
differ even though the scientific input CSV/JSON files are unchanged.

`analysis/SafeFundus_Reviewer_Analyses_Kaggle.ipynb` is the self-contained notebook
version of the same analysis. Its original input instructions refer to the
historical outer archive, which is intentionally not duplicated here.

Saved statistical analyses use paired image-level bootstrap sampling. Repeated
seeds evaluate the same images and do not create independent extra cases.
Matched test-burden comparisons are retrospective. Fusion-weight sensitivity
uses familiar validation data for each threshold and retains the original
equal-weight method rather than selecting a replacement using test performance.

## Publication scope

Document-generation and manuscript-editing scripts are intentionally excluded
because they embed or depend on paper text. Plotting helpers are included with
their destination redirected to `analysis/figures/`; their numerical plotting
logic is unchanged. Existing checkpoints and run artifacts are not retrained or
recomputed by the upload process. The provenance files describe exact copies
and any path-only portability adjustments.
