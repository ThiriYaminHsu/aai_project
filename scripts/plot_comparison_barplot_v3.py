"""Chapter-4 comparison barplot, v2: mean % two-qubit reduction by method
and evaluation family from evaluation_results_final_v4.csv (the final
agents under the preprocess+agent pipeline, best-of-10; baselines on raw
circuits). Agents only (no baselines). Writes figures/comparison_barplot_v3.png into the LaTeX tree.

Usage:
    python scripts/plot_comparison_barplot_v2.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"

ARITH = ("add", "qcla", "mod")


def fam(s):
    if s.startswith("purerandom"):
        return "pure_random"
    if s.startswith("random"):
        return "structured_random"
    if s.startswith("adder"):
        return "adder_heldout"
    return "real_world"


def main():
    df = pd.read_csv(REPO_ROOT / "results" / "logs" / "evaluation_results_final_v4.csv")
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df.copy()
    df["family"] = df["circuit"].apply(fam)
    df = df[(df["family"] != "real_world") | df["circuit"].apply(lambda c: any(k in c.lower() for k in ARITH))]
    df["twoq_red"] = 100 * (df["initial_2q"] - df["final_2q"]) / df["initial_2q"]

    methods = ["agent_R_v2", "agent_P_v2", "agent_A_v2"]
    labels = ["Agent R", "Agent P", "Agent A"]
    colors = ["#55a868", "#4c72b0", "#8172b3"]
    fams = ["structured_random", "pure_random", "adder_heldout", "real_world"]
    fam_labels = ["structured-\nrandom", "pure-\nrandom", "adder\nheld-out", "real-world\narithmetic"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fams))
    w = 0.22
    for i, (m, lab, col) in enumerate(zip(methods, labels, colors)):
        sub = df[df["method"] == m]
        means = [sub[sub["family"] == f]["twoq_red"].mean() for f in fams]
        stds = [sub[sub["family"] == f]["twoq_red"].std() for f in fams]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=2, label=lab, color=col)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("% reduction in two-qubit gate count")
    ax.set_xticks(x)
    ax.set_xticklabels(fam_labels)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Two-qubit gate reduction by final agent and evaluation family (best-of-10)")
    fig.tight_layout()
    out = LATEX_FIGS / "comparison_barplot_v3.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
