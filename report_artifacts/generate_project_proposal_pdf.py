from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "project_proposal_source.md"
OUTPUT_PATH = PROJECT_ROOT / "project_proposal.pdf"

PAGE_SIZE = (8.5, 11)
LEFT = 0.08
RIGHT = 0.92
TOP = 0.95
BOTTOM = 0.07


def add_page(pdf: PdfPages, page_number: int):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        0.5,
        0.035,
        f"High-Cost Medicare Beneficiary Risk Proposal | Page {page_number}",
        fontsize=8,
        color="#666666",
        ha="center",
        va="bottom",
        family="DejaVu Sans",
    )
    return fig, ax


def flush_page(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def wrap_text(text: str, width: int, bullet: bool = False) -> list[str]:
    if bullet:
        content = text[2:].strip()
        wrapped = textwrap.wrap(content, width=width)
        if not wrapped:
            return ["-"]
        return [f"- {wrapped[0]}"] + [f"  {line}" for line in wrapped[1:]]
    return textwrap.wrap(text, width=width) or [""]


def draw_markdown_pdf() -> None:
    lines = SOURCE_PATH.read_text(encoding="utf-8").strip().splitlines()

    with PdfPages(OUTPUT_PATH) as pdf:
        page_number = 1
        fig, ax = add_page(pdf, page_number)
        y = TOP

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                y -= 0.018
                continue

            if line.startswith("# "):
                chunks = wrap_text(line[2:], width=44)
                font_size = 18
                font_weight = "bold"
                line_height = 0.038
                extra_after = 0.018
            elif line.startswith("## "):
                chunks = wrap_text(line[3:], width=58)
                font_size = 13
                font_weight = "bold"
                line_height = 0.029
                extra_after = 0.007
                y -= 0.007
            elif line.startswith("- "):
                chunks = wrap_text(line, width=92, bullet=True)
                font_size = 10
                font_weight = "normal"
                line_height = 0.021
                extra_after = 0.004
            else:
                chunks = wrap_text(line, width=96)
                font_size = 10.5
                font_weight = "normal"
                line_height = 0.022
                extra_after = 0.007

            needed = (len(chunks) * line_height) + extra_after
            if y - needed < BOTTOM:
                flush_page(pdf, fig)
                page_number += 1
                fig, ax = add_page(pdf, page_number)
                y = TOP

            for chunk in chunks:
                ax.text(
                    LEFT,
                    y,
                    chunk,
                    fontsize=font_size,
                    fontweight=font_weight,
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                    color="#111111",
                )
                y -= line_height

            y -= extra_after

        flush_page(pdf, fig)


def main() -> None:
    draw_markdown_pdf()
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
