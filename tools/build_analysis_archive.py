"""Rebuild the small CSV/JSON archive expected by the preserved analysis code."""
from pathlib import Path
import argparse
import io
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = [
    "06C_SwinTiny_ASL_NormalAware_MultiSeed_Classifier",
    "07B_NormalAware_JointEnergy_OpenSet_Scoring",
    "08_MC_Dropout_Uncertainty",
    "09_SafeFundus_Fusion_Calibration_Ablation",
    "10B_RFMiD2_External_Validation_Fixed",
]


def build(destination, outputs):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        for name in BUNDLES:
            folder = outputs / name
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json"})
            if not files:
                raise ValueError("No saved numerical outputs found in " + str(folder))
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as inner:
                for path in files:
                    inner.write(path, path.relative_to(folder).as_posix())
                    total += 1
            outer.writestr("Models After Run/" + name + ".zip", payload.getvalue())
    print(f"Saved {total} CSV/JSON artifacts in {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    build(args.destination, args.outputs.resolve())
