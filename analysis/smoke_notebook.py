"""Reproduce every saved analysis CSV using the self-contained notebook code."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    spec = importlib.util.spec_from_file_location("build_analysis_archive", REPO / "tools/build_analysis_archive.py")
    archive_helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive_helper)
    archive = REPO / ".local/notebook_validation_inputs.zip"
    archive_helper.build(archive, REPO / "outputs")
    notebook = json.loads((HERE / "SafeFundus_Reviewer_Analyses_Kaggle.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    namespace = {"__name__": "safefundus_notebook_validation"}
    exec(compile(code, "SafeFundus_Reviewer_Analyses_Kaggle.ipynb:analysis", "exec"), namespace)
    output = REPO / ".local/notebook_reproduction"
    namespace["main"](archive, output, reps=10000)
    checks = []
    for original in sorted((HERE / "tables").glob("*.csv")):
        reproduced = output / "tables" / original.name
        match = reproduced.is_file() and hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(reproduced.read_bytes()).digest()
        checks.append({"table": original.name, "byte_identical": match})
        assert match, original.name
    result = {"bootstrap_replicates": 10000, "all_table_comparisons_passed": True, "table_checks": checks}
    (REPO / ".local/notebook_reproduction_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: self-contained notebook reproduced all {len(checks)} saved CSV tables byte-for-byte.")


if __name__ == "__main__":
    main()
