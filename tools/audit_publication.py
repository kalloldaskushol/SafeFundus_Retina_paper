"""Check the explicitly staged scientific publication set before pushing."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SECRET_PATTERNS = {
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "credential_url": r"https?://[^\s/:]+:[^\s/@]+@",
    "openai_key": r"\bsk-[A-Za-z0-9_-]{20,}",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    allowed_roots = {"notebooks", "outputs", "analysis", "tools", "provenance"}
    allowed_top = {"README.md", ".gitattributes", ".gitignore", "requirements.txt"}
    forbidden_suffixes = {".tex", ".bib", ".cls", ".bst", ".doc", ".docx", ".zip", ".jpg", ".jpeg"}
    notebook_count = 0
    checkpoints = 0
    text_files = 0
    for name in names:
        relative = Path(name)
        assert relative.parts[0] in allowed_roots or name in allowed_top, name
        assert relative.suffix.lower() not in forbidden_suffixes, name
        assert not any(part.lower() in {"dataset", "datasets", "data", "__pycache__", ".local", ".git"}
                       for part in relative.parts), name
        path = ROOT / name
        if path.suffix.lower() == ".pdf":
            assert re.fullmatch(r"analysis/figures/reviewer_analyses_(?:two_panel|compact_row|single_column)\.pdf", name), name
        if path.suffix.lower() == ".png":
            assert "figures" in relative.parts and relative.parts[0] in {"outputs", "analysis"}, name
        if path.suffix.lower() in {".pth", ".npz"}:
            pointer = subprocess.run(["git", "show", ":" + name], cwd=ROOT, capture_output=True, check=True).stdout
            assert pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n"), name
            assert ("oid sha256:" + sha(path)).encode() in pointer, name
            if path.suffix.lower() == ".pth":
                checkpoints += 1
        else:
            assert path.stat().st_size < 100 * 1024 * 1024, name
        if path.suffix.lower() in {".py", ".ipynb", ".json", ".csv", ".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            for label, pattern in SECRET_PATTERNS.items():
                assert not re.search(pattern, text), (name, label)
            text_files += 1
        if path.suffix.lower() == ".ipynb":
            notebook_count += 1
            notebook = json.loads(text)
            assert notebook.get("nbformat") == 4
            assert all(cell.get("cell_type") in {"code", "markdown", "raw"} for cell in notebook["cells"])
        if path.suffix.lower() == ".py":
            compile(path.read_text(encoding="utf-8"), name, "exec")
    with (ROOT / "provenance/artifact_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 323
    for row in records:
        assert row["repository_path"] in names
        assert sha(ROOT / row["repository_path"]) == row["sha256"], row["repository_path"]
    assert notebook_count == 13 and checkpoints == 13
    summary = {"staged_files": len(names), "notebooks": notebook_count, "checkpoints": checkpoints,
               "scientific_artifacts_with_matching_hashes": len(records), "text_files_secret_scanned": text_files,
               "manuscript_and_dataset_paths_excluded": True, "model_files_use_verified_lfs_pointers": True,
               "python_syntax_checks_passed": True}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    audit()
