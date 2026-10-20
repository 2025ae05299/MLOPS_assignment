"""Download and clean the UCI Heart Disease collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.data.acquire import acquire_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the multi-site UCI heart-disease CSV.")
    parser.add_argument("--force", action="store_true", help="Re-download raw files even if present.")
    args = parser.parse_args()
    frame = acquire_dataset(force_download=args.force)
    print(f"Wrote {len(frame)} rows to data/processed/heart_disease_clean.csv")
    print(f"Positive rate: {frame['heart_disease'].mean():.3f}")


if __name__ == "__main__":
    main()
