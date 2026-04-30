from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "writeup_source.md"
OUTPUT_PATH = PROJECT_ROOT / "writeup.pdf"


def main() -> None:
    text = SOURCE_PATH.read_text().strip().splitlines()

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.96
    for line in text:
        if not line.strip():
            y -= 0.022
            continue

        if line.startswith("# "):
            wrapped = textwrap.wrap(line[2:], width=40)
            for chunk in wrapped:
                ax.text(
                    0.08,
                    y,
                    chunk,
                    fontsize=16,
                    fontweight="bold",
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
                y -= 0.032
            y -= 0.01
            continue

        wrapped = textwrap.wrap(line, width=104)
        for chunk in wrapped:
            ax.text(
                0.08,
                y,
                chunk,
                fontsize=11,
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            y -= 0.022

    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
