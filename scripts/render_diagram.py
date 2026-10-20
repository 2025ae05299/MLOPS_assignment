"""Draw the PulseGuard architecture diagram used by the report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.config import FIGURES_DIR  # noqa: E402
from pulseguard.viz.style import PALETTE  # noqa: E402


def render(path: Path | None = None) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destination = path or (FIGURES_DIR / "architecture.png")
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=color,
            edgecolor=PALETTE["ink"],
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="white", wrap=True)

    box(3, 40, 22, 14, "UCI four-site\nraw files", PALETTE["ink"])
    box(30, 40, 24, 14, "Clean + sentinel\nrules", PALETTE["teal"])
    box(59, 40, 24, 14, "Train / MLflow\nselect bundle", PALETTE["teal"])
    box(88, 40, 20, 14, "joblib\npipeline", PALETTE["coral"])
    box(8, 16, 28, 14, "GitHub Actions\nlint, test, train", PALETTE["ink"])
    box(42, 16, 28, 14, "Docker API\n/predict /metrics", PALETTE["coral"])
    box(76, 16, 30, 14, "Prometheus\n+ Grafana", PALETTE["teal"])
    ax.annotate("", xy=(30, 47), xytext=(25, 47), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    ax.annotate("", xy=(59, 47), xytext=(54, 47), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    ax.annotate("", xy=(88, 47), xytext=(83, 47), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    ax.annotate("", xy=(56, 30), xytext=(98, 40), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    ax.annotate("", xy=(76, 23), xytext=(70, 23), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    ax.set_title("PulseGuard educational architecture", color=PALETTE["ink"], loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    print(render())
