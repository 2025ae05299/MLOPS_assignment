"""Create the EDA figures used in the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from pulseguard.data.preprocess import load_clean_frame  # noqa: E402
from pulseguard.viz.style import PALETTE, style_axes  # noqa: E402


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=140)
    plt.close(fig)


def main() -> None:
    frame = load_clean_frame()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    balance = frame.groupby(["source_site", "heart_disease"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    balance.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE["teal"], PALETTE["coral"]], legend=True)
    ax.set_title("Class balance by UCI collection site")
    ax.set_xlabel("Source site")
    ax.set_ylabel("Patients")
    ax.legend(["No disease marker (0)", "Disease marker (1)"], frameon=False)
    style_axes(ax)
    _save(fig, "class_balance.png")

    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for axis, column in zip(axes.ravel(), numeric):
        sns.histplot(frame[column].dropna(), bins=18, ax=axis, color=PALETTE["teal"], edgecolor="white")
        axis.set_title(column)
        style_axes(axis)
    axes.ravel()[-1].axis("off")
    _save(fig, "histograms.png")

    corr = frame[numeric + ["heart_disease"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", ax=ax, cbar=False)
    ax.set_title("Correlation of measured fields")
    _save(fig, "correlation_heatmap.png")

    missing_cols = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    missing = frame.groupby("source_site")[missing_cols].apply(lambda part: part.isna().mean())
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.heatmap(missing, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
    ax.set_title("Missing rate by site after sentinel cleanup")
    _save(fig, "missingness_by_site.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sns.boxplot(
        data=frame.dropna(subset=["chol"]),
        x="heart_disease",
        y="chol",
        hue="heart_disease",
        palette=[PALETTE["teal"], PALETTE["coral"]],
        legend=False,
        ax=ax,
    )
    ax.set_title("Cholesterol by educational target")
    ax.set_xlabel("heart_disease")
    style_axes(ax)
    _save(fig, "cholesterol_by_target.png")

    summary = {
        "rows": int(len(frame)),
        "positive_cases": int(frame["heart_disease"].sum()),
        "positive_rate": float(frame["heart_disease"].mean()),
        "sites": {site: int(count) for site, count in frame["source_site"].value_counts().items()},
        "missing_rate": {column: float(frame[column].isna().mean()) for column in missing_cols},
        "notes": [
            "The four processed UCI sites are modeled together instead of the usual 303-row Cleveland extract.",
            "Zero cholesterol values, concentrated in the Switzerland file, are treated as unmeasured.",
            "ca and thal are missing for a large share of non-Cleveland rows, so missingness is a feature.",
        ],
    }
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
