"""Dissertation figure: compare the three deterministic baseline methods
discussed in this project (basic_opt, full_reduce, teleport_reduce)
across random 10-qubit Clifford+T circuits of increasing size, on the raw
(unpreprocessed) circuit -- matching scripts/sweep_baselines_100.py's
methodology, not the preprocessed circuits the RL agents actually train
on.

Three panels, extending the structure of Riu et al. (2025) Figure 2 (which
only covers (a)/(b)) with a third: (a) total gate count, (b) two-qubit
gate count, (c) T-count, each after optimisation against each circuit's
own initial count on that metric.

Usage:
    python scripts/plot_baseline_comparison.py
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
import pyzx as zx

from circopt_adder.baselines import (
    baseline_basic_optimization,
    baseline_full_reduce,
    baseline_teleport_reduce,
)
from circopt_adder.zx_utils import gate_count, t_count, two_qubit_gate_count

QUBITS = 10
DEPTHS = list(range(10, 321, 5))

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
    # Red region spans the full plotted height above the diagonal, not just up
    # to `hi` -- full_reduce's own output points run well above the
    # initial-gate-count range, and capping the shading at `hi` left those
    # points sitting on an unshaded white background instead of red.
    ax.fill_between([lo, hi], [lo, hi], [y_top, y_top], color="#f4c7c3", alpha=0.5, zorder=0)
    ax.fill_between([lo, hi], [y_bottom, y_bottom], [lo, hi], color="#c8e6c9", alpha=0.5, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=0.8, zorder=1)

    for name, _ in METHODS:
        xs, ys = data[name]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                marker=MARKERS[name], color=COLORS[name], label=name,
                linewidth=1.0, markersize=4, linestyle=":")

    # Fill the axes area right up to the data range, no default matplotlib
    # margin, so the shaded regions and data reach every border.
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

    for depth in DEPTHS:
        g = zx.generate.cliffordT(
            QUBITS, depth, p_t=0.25, p_s=0.25, p_hsh=0.25, p_cnot=0.25, seed=depth
        )
        c = zx.Circuit.from_graph(g).to_basic_gates()
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
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = OUT_DIR / "baseline_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
