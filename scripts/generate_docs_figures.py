#!/usr/bin/env python3
"""Generate static figures embedded in the selkit documentation.

Re-run after the tutorial pipeline changes to keep figures in sync:

    python3 scripts/generate_docs_figures.py

Produces PNG files under docs/_static/figures/. Commit the generated
files alongside any rst changes that reference them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "_static" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def figure_workflow() -> None:
    """Pipeline diagram: inputs → validate → fit → LRT + BEB → outputs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, fc):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, family="monospace")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"))

    # Inputs (left column)
    box(0.2, 3.2, 2.0, 0.9, "alignment.fa", "#dbeafe")
    box(0.2, 2.0, 2.0, 0.9, "tree.nwk", "#dbeafe")

    # Validate (middle-left)
    box(3.0, 2.6, 1.8, 0.9, "validate_inputs", "#fef3c7")

    # Engine fit (middle)
    box(5.4, 2.6, 2.2, 0.9, "run_site_models\n(M0, M1a, M2a,\nM7, M8, M8a)", "#fde68a")

    # LRT + BEB (middle-right)
    box(5.4, 0.6, 1.0, 0.9, "LRTs", "#fecaca")
    box(6.6, 0.6, 1.0, 0.9, "BEB", "#fecaca")

    # Outputs (right column)
    box(8.2, 3.2, 1.6, 0.9, "results.json", "#bbf7d0")
    box(8.2, 2.0, 1.6, 0.9, "*.tsv", "#bbf7d0")
    box(8.2, 0.8, 1.6, 0.9, "run.yaml", "#bbf7d0")

    # Arrows
    arrow(2.2, 3.65, 3.0, 3.2)
    arrow(2.2, 2.45, 3.0, 2.9)
    arrow(4.8, 3.05, 5.4, 3.05)
    arrow(6.5, 2.6, 5.9, 1.5)
    arrow(6.5, 2.6, 7.1, 1.5)
    arrow(7.6, 3.3, 8.2, 3.65)
    arrow(7.6, 3.0, 8.2, 2.45)
    arrow(6.0, 0.6, 8.2, 1.25)

    ax.text(5.0, 4.4, "selkit codeml site-models",
            ha="center", va="center", fontsize=13, weight="bold")

    _save(fig, "workflow.png")


def figure_beb_posterior(run_dir: Path) -> None:
    """Stem plot of P(omega > 1) per site under M2a, with a 0.95 threshold line."""
    beb_tsv = run_dir / "beb_M2a.tsv"
    if not beb_tsv.exists():
        print(f"skipping beb figure: {beb_tsv} not found", file=sys.stderr)
        return
    rows = beb_tsv.read_text().strip().splitlines()
    header = rows[0].split("\t")
    assert header == ["site", "p_positive", "mean_omega"], header
    sites = []
    p_pos = []
    for r in rows[1:]:
        cols = r.split("\t")
        sites.append(int(cols[0]))
        p_pos.append(float(cols[1]))
    sites = np.array(sites)
    p_pos = np.array(p_pos)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ["#dc2626" if p >= 0.95 else "#64748b" for p in p_pos]
    ax.bar(sites, p_pos, color=colors, width=0.8, linewidth=0)
    ax.axhline(0.95, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.7,
               label="P = 0.95 threshold")
    ax.set_xlabel("codon site (1-indexed)")
    ax.set_ylabel("P(ω > 1 | site)  under M2a")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, sites.max() + 1)
    n_sig = int((p_pos >= 0.95).sum())
    ax.set_title(
        f"NEB posterior of positive selection (HIV env V3, 13 taxa)  —  "
        f"{n_sig} sites with P ≥ 0.95",
        fontsize=11,
    )
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _save(fig, "beb_posterior_hiv13s.png")


def figure_lrt_decision() -> None:
    """Simple decision-tree diagram showing which LRT to care about."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#f1f5f9"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9.5)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"))
        if label:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label,
                    fontsize=9, color="#444")

    box(3.8, 5.5, 2.4, 0.8, "Run all 6 models", fc="#fde68a")

    box(0.5, 3.8, 2.6, 0.9, "M1a vs M2a\n(2 df, χ²)", fc="#dbeafe")
    box(3.7, 3.8, 2.6, 0.9, "M7 vs M8\n(2 df, χ²)", fc="#dbeafe")
    box(6.9, 3.8, 2.6, 0.9, "M8a vs M8\n(1 df, mixed χ²)", fc="#fef3c7")

    arrow(4.5, 5.5, 1.8, 4.7)
    arrow(5.0, 5.5, 5.0, 4.7)
    arrow(5.5, 5.5, 8.2, 4.7)

    box(0.5, 2.0, 2.6, 0.9, "p < 0.05\n⇒ evidence of ω > 1 class", fc="#bbf7d0")
    box(3.7, 2.0, 2.6, 0.9, "p < 0.05\n⇒ same conclusion, different null", fc="#bbf7d0")
    box(6.9, 2.0, 2.6, 0.9, "p < 0.05\n⇒ ω > 1 is truly > 1\n(not just = 1)", fc="#bbf7d0")

    arrow(1.8, 3.8, 1.8, 2.9)
    arrow(5.0, 3.8, 5.0, 2.9)
    arrow(8.2, 3.8, 8.2, 2.9)

    box(0.8, 0.2, 8.4, 1.2,
        "Sites flagged as positively selected are those with\nBEB posterior  P(ω > 1 | site)  ≥ 0.95  under M2a or M8",
        fc="#fecaca")
    arrow(1.8, 2.0, 3.5, 1.4)
    arrow(5.0, 2.0, 5.0, 1.4)
    arrow(8.2, 2.0, 6.5, 1.4)

    _save(fig, "lrt_decision.png")


def main() -> int:
    figure_workflow()
    figure_lrt_decision()

    run_dir = Path("/tmp/selkit-tutorial-out/hiv_13s")
    figure_beb_posterior(run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
