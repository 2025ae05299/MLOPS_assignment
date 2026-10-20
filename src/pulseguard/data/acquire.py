"""Download the four UCI processed heart-disease sites and write a clean CSV."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from pulseguard.config import (
    DATA_PROCESSED,
    DATA_RAW,
    RAW_COLUMNS,
    SEVERITY_COLUMN,
    SITE_FILES,
    TARGET_COLUMN,
    UCI_BASE,
)
from pulseguard.data.preprocess import apply_sentinel_rules


def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size < 50:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {dest}")
            return
        except Exception as exc:  # noqa: BLE001 - retry network and IO failures
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_site(path: Path, site: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
        on_bad_lines="error",
    )
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"{site} does not have {len(RAW_COLUMNS)} columns")
    frame["source_site"] = site
    return frame


def acquire_dataset(force_download: bool = False) -> pd.DataFrame:
    """Fetch raw UCI files if needed, clean sentinels, and persist the modeling table."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest_sites = []
    for site, filename in SITE_FILES.items():
        dest = DATA_RAW / filename
        if force_download or not dest.exists():
            _download(f"{UCI_BASE}/{filename}", dest)
        site_frame = _read_site(dest, site)
        frames.append(site_frame)
        manifest_sites.append(
            {
                "site": site,
                "file": filename,
                "rows": int(len(site_frame)),
                "sha256": _sha256(dest),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = apply_sentinel_rules(combined)
    combined = combined.dropna(subset=["age", SEVERITY_COLUMN]).copy()
    combined[SEVERITY_COLUMN] = combined[SEVERITY_COLUMN].astype(int)
    combined[TARGET_COLUMN] = (combined[SEVERITY_COLUMN] > 0).astype(int)

    cleaned_path = DATA_PROCESSED / "heart_disease_clean.csv"
    combined.to_csv(cleaned_path, index=False)

    quality = {
        "rows": int(len(combined)),
        "positive_cases": int(combined[TARGET_COLUMN].sum()),
        "positive_rate": float(combined[TARGET_COLUMN].mean()),
        "sites": manifest_sites,
        "missing_rate": {
            column: float(combined[column].isna().mean())
            for column in combined.columns
            if column not in {TARGET_COLUMN}
        },
        "sentinel_policy": [
            "Question marks in the UCI files are treated as missing.",
            "Cholesterol, resting blood pressure, and max heart rate at or below zero are treated as missing sentinels.",
            "Switzerland records cholesterol as 0 when it was not measured; those zeros are not real lipid values.",
            "Diagnosis severity num is retained, and heart_disease is 1 when num > 0.",
        ],
        "citation": "Janosi et al., Heart Disease, UCI ML Repository, https://doi.org/10.24432/C52P4X",
    }
    (DATA_PROCESSED / "manifest.json").write_text(json.dumps(quality, indent=2))
    return combined
