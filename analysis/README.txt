SafeFundus saved-output analyses

run_analysis.py and SafeFundus_Reviewer_Analyses_Kaggle.ipynb contain the same
analysis logic. The archive-based entry point is preserved from the local run.
No model training or original image datasets are needed for these analyses.

From the repository root:
  python -m pip install -r analysis/requirements.txt
  python tools/build_analysis_archive.py --destination .local/analysis_inputs.zip
  python analysis/run_analysis.py --archive .local/analysis_inputs.zip --output .local/reproduced_analysis --bootstrap 10000
  python analysis/verify_results.py

Full notebook reproduction (all 20 saved CSVs, 10,000 resamples):
  python analysis/smoke_notebook.py

The saved tables in analysis/tables are the original 10,000-resample outputs.
The archive rebuilt by the helper contains only previously saved CSV/JSON
artifacts, not original input images or model weights. Its archive hash differs
from the original full archive, but its scientific inputs are byte-identical.

Three seeds evaluate the same images. They are not independent extra images.
The paired bootstrap describes image-sampling uncertainty conditional on the
fixed models and original validation calibration, not retraining uncertainty.
Matched test-burden thresholds are retrospective descriptions. The five-weight
sensitivity sweep recalibrates using familiar validation scores only and does
not replace the original equal-weight method based on unknown test results.

The compact/stacked figure helper scripts save scientific plots under
analysis/figures. Their numerical plotting logic is unchanged from the local
scripts; only their output directory has been made repository-local.
