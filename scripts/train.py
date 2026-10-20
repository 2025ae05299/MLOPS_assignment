"""Entry point for reproducible training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.models.train import main  # noqa: E402

if __name__ == "__main__":
    main()
