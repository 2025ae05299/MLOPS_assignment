"""Write the narrative notebook that calls the same library code as the scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    cells = [
        new_markdown_cell(
            """# PulseGuard EDA and modeling notes

Educational notebook for the UCI Heart Disease collection. The classifier is not a medical device and must not be used to diagnose or treat anyone.

This notebook calls the same acquisition, plotting, and training code as `scripts/`, so a figure produced here is the figure produced by CI."""
        ),
        new_code_cell(
            """from pathlib import Path
import json
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.data.acquire import acquire_dataset
from pulseguard.data.preprocess import load_clean_frame

frame = acquire_dataset(force_download=False)
print(frame.shape)
print(frame.groupby("source_site")["heart_disease"].agg(["count", "mean"]))
frame.head(3)"""
        ),
        new_markdown_cell(
            """## Why this table is not the usual 303-row extract

Cleveland is only one of four processed UCI sites. Combining them makes missingness and site shift visible. Switzerland records unmeasured cholesterol as 0. Those zeros are converted to missing before any chart or model sees them."""
        ),
        new_code_cell(
            """import matplotlib.pyplot as plt
from IPython.display import Image, display

# The EDA script is the source of the committed figures.
import runpy
runpy.run_path(str(ROOT / "scripts" / "run_eda.py"), run_name="__main__")

summary = json.loads((ROOT / "reports" / "eda_summary.json").read_text())
print(json.dumps(summary, indent=2))
for name in ["class_balance.png", "histograms.png", "correlation_heatmap.png", "missingness_by_site.png"]:
    display(Image(filename=str(ROOT / "reports" / "figures" / name)))"""
        ),
        new_markdown_cell(
            """## Modeling choice

Three candidates share one pipeline: logistic regression, random forest, and histogram gradient boosting. Selection is not raw accuracy. The training script maximizes `0.70 * ROC-AUC + 0.30 * recall` on out-of-fold train scores, then tunes an F1.5 threshold and touches the holdout split once.

Re-running the next cell retrains the serving bundle and refreshes MLflow. Skip it if you only want to read the committed summary."""
        ),
        new_code_cell(
            """summary_path = ROOT / "reports" / "experiment_summary.json"
if summary_path.exists():
    print(summary_path.read_text())
else:
    print("No experiment summary yet. Run scripts/train.py.")

fig_dir = ROOT / "reports" / "figures"
for name in ["model_comparison.png", "roc_curve.png", "confusion_matrix.png", "feature_importance.png"]:
    path = fig_dir / name
    if path.exists():
        display(Image(filename=str(path)))"""
        ),
        new_markdown_cell(
            """## Inference contract

Serving accepts the 13 UCI fields plus optional `source_site`. Engineered columns are created inside the saved scikit-learn pipeline, so a notebook transform and the API transform cannot drift."""
        ),
        new_code_cell(
            """import json
from pulseguard.config import DISCLAIMER

payload = json.loads((ROOT / "examples" / "patient_lower.json").read_text())
print(payload)
print(DISCLAIMER)
print("Score this file with: python scripts/infer.py examples/patient_lower.json")"""
        ),
    ]
    notebook = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    destination = ROOT / "notebooks" / "01_eda_and_modeling.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
