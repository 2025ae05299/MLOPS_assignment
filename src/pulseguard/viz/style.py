"""Plot styling so PulseGuard figures are consistent and not a default tutorial theme."""

PALETTE = {
    "teal": "#0F6E6B",
    "coral": "#D4583B",
    "ink": "#1C2B33",
    "sand": "#F7F1E8",
    "muted": "#8C8178",
    "gold": "#C48A2A",
}


def style_axes(ax) -> None:
    ax.set_facecolor(PALETTE["sand"])
    ax.figure.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.tick_params(colors=PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
