from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data_exploration_qa_source.md"
OUTPUT_PATH = PROJECT_ROOT / "data_exploration_qa.pdf"

PAGE_SIZE = (8.5, 11)
LEFT = 0.09
RIGHT = 0.91
HEADER_Y = 0.94
TOP = 0.875
BOTTOM = 0.11
BODY_WIDTH = 92
TABLE_WIDTH = 96


def add_page(pdf: PdfPages, page_number: int):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        LEFT,
        HEADER_Y,
        "CMS High-Cost Claim Classifier",
        fontsize=8.5,
        color="#5A6472",
        ha="left",
        va="top",
        family="DejaVu Sans",
    )
    ax.text(
        RIGHT,
        0.055,
        f"Page {page_number}",
        fontsize=8,
        color="#5A6472",
        ha="right",
        va="bottom",
        family="DejaVu Sans",
    )
    ax.plot([LEFT, RIGHT], [HEADER_Y - 0.02, HEADER_Y - 0.02], color="#D8DEE8", linewidth=0.8)
    return fig, ax


def flush_page(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text


def wrap_line(text: str, width: int, hanging_indent: str = "") -> list[str]:
    text = clean_inline_markdown(text)
    wrapped = textwrap.wrap(text, width=width) or [""]
    if hanging_indent and len(wrapped) > 1:
        return [wrapped[0]] + [hanging_indent + line for line in wrapped[1:]]
    return wrapped


def parse_blocks(lines: list[str]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()

        if not stripped:
            blocks.append({"kind": "space"})
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            blocks.append({"kind": "code", "lines": code_lines})
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                candidate = lines[i].strip()
                if not re.match(r"^\|[\s:\-|]+\|$", candidate):
                    table_lines.append(candidate)
                i += 1
            blocks.append({"kind": "table", "lines": table_lines})
            continue

        if stripped.startswith("# "):
            blocks.append({"kind": "title", "text": stripped[2:]})
        elif stripped.startswith("## "):
            blocks.append({"kind": "heading", "text": stripped[3:]})
        elif stripped.startswith("- "):
            blocks.append({"kind": "bullet", "text": stripped[2:]})
        else:
            blocks.append({"kind": "paragraph", "text": stripped})
        i += 1

    return blocks


def block_height(block: dict[str, object]) -> float:
    kind = block["kind"]
    if kind == "space":
        return 0.014
    if kind == "title":
        lines = wrap_line(str(block["text"]), 42)
        return len(lines) * 0.04 + 0.025
    if kind == "heading":
        lines = wrap_line(str(block["text"]), 70)
        return len(lines) * 0.027 + 0.014
    if kind == "bullet":
        lines = wrap_line("- " + str(block["text"]), 96, "  ")
        return len(lines) * 0.019 + 0.005
    if kind == "code":
        return len(block["lines"]) * 0.017 + 0.020
    if kind == "table":
        total = 0.018
        for row in block["lines"]:
            total += max(1, len(clean_inline_markdown(row)) // TABLE_WIDTH + 1) * 0.015
        return total + 0.008
    lines = wrap_line(str(block["text"]), BODY_WIDTH)
    return len(lines) * 0.019 + 0.007


def draw_table(ax, y: float, table_lines: list[str]) -> float:
    for row_index, row in enumerate(table_lines):
        text = clean_inline_markdown(row)
        chunks = wrap_line(text, TABLE_WIDTH)
        for chunk in chunks:
            ax.text(
                LEFT + 0.005,
                y,
                chunk,
                fontsize=7.8,
                va="top",
                ha="left",
                family="DejaVu Sans Mono",
                color="#111111",
                bbox={
                    "facecolor": "#F3F6FA" if row_index == 0 else "#FFFFFF",
                    "edgecolor": "#E3E8F0",
                    "boxstyle": "square,pad=0.18",
                    "linewidth": 0.4,
                },
            )
            y -= 0.015
    return y - 0.010


def draw_pdf() -> None:
    blocks = parse_blocks(SOURCE_PATH.read_text(encoding="utf-8").splitlines())
    with PdfPages(OUTPUT_PATH) as pdf:
        page_number = 1
        fig, ax = add_page(pdf, page_number)
        y = TOP

        for block_index, block in enumerate(blocks):
            needed = block_height(block)
            if block["kind"] == "heading":
                next_content_height = 0.0
                lookahead_index = block_index + 1
                while lookahead_index < len(blocks) and blocks[lookahead_index]["kind"] == "space":
                    lookahead_index += 1
                if lookahead_index < len(blocks):
                    next_content_height = block_height(blocks[lookahead_index])
                needed += min(next_content_height, 0.16)
            if y - needed < BOTTOM:
                flush_page(pdf, fig)
                page_number += 1
                fig, ax = add_page(pdf, page_number)
                y = TOP

            kind = block["kind"]

            if kind == "space":
                y -= 0.014
            elif kind == "title":
                for chunk in wrap_line(str(block["text"]), 42):
                    ax.text(
                        LEFT,
                        y,
                        chunk,
                        fontsize=18,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        family="DejaVu Sans",
                        color="#172033",
                    )
                    y -= 0.04
                y -= 0.025
            elif kind == "heading":
                y -= 0.004
                for chunk in wrap_line(str(block["text"]), 70):
                    ax.text(
                        LEFT,
                        y,
                        chunk,
                        fontsize=12.5,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        family="DejaVu Sans",
                        color="#20324D",
                    )
                    y -= 0.029
                ax.plot([LEFT, RIGHT], [y + 0.004, y + 0.004], color="#E7EBF2", linewidth=0.6)
                y -= 0.012
            elif kind == "bullet":
                for chunk in wrap_line("- " + str(block["text"]), 96, "  "):
                    ax.text(LEFT + 0.015, y, chunk, fontsize=9.5, va="top", ha="left", family="DejaVu Sans")
                    y -= 0.020
                y -= 0.005
            elif kind == "code":
                for line in block["lines"]:
                    ax.text(
                        LEFT + 0.015,
                        y,
                        line,
                        fontsize=8.8,
                        va="top",
                        ha="left",
                        family="DejaVu Sans Mono",
                        bbox={"facecolor": "#F5F7FA", "edgecolor": "#E1E6EF", "boxstyle": "square,pad=0.15"},
                    )
                    y -= 0.017
                y -= 0.012
            elif kind == "table":
                y = draw_table(ax, y, block["lines"])
            else:
                for chunk in wrap_line(str(block["text"]), BODY_WIDTH):
                    ax.text(
                        LEFT,
                        y,
                        chunk,
                        fontsize=9.8,
                        va="top",
                        ha="left",
                        family="DejaVu Sans",
                        color="#151515",
                    )
                    y -= 0.0205
                y -= 0.008

        flush_page(pdf, fig)


def main() -> None:
    draw_pdf()
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
