"""Dissertation figure: the same three-panel baseline comparison as
plot_baseline_comparison.py, but for ripple-carry adder circuits
(bit-width 2-10) instead of random Clifford+T circuits -- on the raw
(unpreprocessed) adder, matching that script's methodology of
characterising the baselines independently of the RL training pipeline's
own preprocessing.

Unlike the random-circuit sweep, adders are a deterministic construction:
there are only 9 distinct circuits here (one per bit-width), not repeated
samples -- each point on each panel is exactly one adder, not a mean over
several draws.

Usage:
    python scripts/plot_baseline_comparison_adders.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from circopt_adder.baselines import (
    baseline_basic_optimization,
    baseline_full_reduce,
    baseline_teleport_reduce,
)
from circopt_adder.generators import ripple_carry_adder
from circopt_adder.zx_utils import gate_count, t_count, two_qubit_gate_count

BIT_WIDTHS = list(range(2, 11))

METHODS = [
    ("basic_opt", baseline_basic_optimization),
    ("full_reduce", baseline_full_reduce),
    ("teleport_reduce", baseline_teleport_reduce),
]

COLORS = {
    "basic_opt": "#1f77b4", "full_reduce": "#7b2d90",
    "teleport_reduce": "#2ca02c",
}
MARKERS = {
    "basic_opt": "s", "full_reduce": "v", "teleport_reduce": "^",
}


def _plot_panel(ax, data, xlabel, ylabel, title):
    all_x = [x for name, _ in METHODS for x in data[name][0]]
    all_y = [y for name, _ in METHODS for y in data[name][1]]
    lo, hi = min(all_x), max(all_x)
    y_bottom = min(lo, min(all_y))
    y_top = max(hi, max(all_y))
    ax.fill_between([lo, hi], [lo, hi], [y_top, y_top], color="#f4c7c3", alpha=0.5, zorder=0)
    ax.fill_between([lo, hi], [y_bottom, y_bottom], [lo, hi], color="#c8e6c9", alpha=0.5, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=0.8, zorder=1)

    for name, _ in METHODS:
        xs, ys = data[name]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                marker=MARKERS[name], color=COLORS[name], label=name,
                linewidth=1.0, markersize=5, linestyle=":")

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_bottom, y_top)
    ax.margins(0)
    ax.grid(True, color="white", linewidth=0.8, zorder=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    gate_data = {name: ([], []) for name, _ in METHODS}
    twoq_data = {name: ([], []) for name, _ in METHODS}
    t_data = {name: ([], []) for name, _ in METHODS}

    for n_bits in BIT_WIDTHS:
        c = ripple_carry_adder(n_bits)
        init_gates = gate_count(c)
        init_2q = two_qubit_gate_count(c)
        init_t = t_count(c)
        for name, fn in METHODS:
            out = fn(c)
            gate_data[name][0].append(init_gates)
            gate_data[name][1].append(gate_count(out))
            twoq_data[name][0].append(init_2q)
            twoq_data[name][1].append(two_qubit_gate_count(out))
            t_data[name][0].append(init_t)
            t_data[name][1].append(t_count(out))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    _plot_panel(axes[0], gate_data, "Initial gates", "Gates after optimisation", "(a) Total gate count")
    _plot_panel(axes[1], twoq_data, "Initial 2-qubit gates", "2-qubit gates after optimisation", "(b) Two-qubit gate count")
    _plot_panel(axes[2], t_data, "Initial T-count", "T-count after optimisation", "(c) T-count")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("Deterministic baselines on ripple-carry adders (bit-width 2-10)", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = OUT_DIR / "baseline_comparison_adders.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
