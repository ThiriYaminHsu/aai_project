"""Riu-et-al-style pipeline figure for Chapter 3 (user request): a
horizontal flow with real rendered artefacts at each stage, adapted to
this dissertation's final search-distillation method:

  Original circuit -> graph-like ZX representation -> Environment
  (rewrite applied) <-> Agent (GATv2, action/observation) -> Circuit
  extraction -> STOP/basic cleanup -> Optimized circuit,

with the reward arrow of the original replaced by the winning-trajectory
/ behaviour-cloning arrow of the final method.

Writes Writing/latex/figures/pipeline_riu_style.png.

Usage:
    python3 scripts/plot_pipeline_riu_style.py
"""

import io
import sys
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyzx as zx
from PIL import Image
from qiskit import QuantumCircuit

from circopt_adder.zx_utils import circuit_to_graphlike

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"
OUT = LATEX_FIGS / "pipeline_riu_style.png"


def fig_to_img(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=False,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_circuit(circ):
    qc = QuantumCircuit.from_qasm_str(circ.to_basic_gates().to_qasm())
    return fig_to_img(qc.draw(output="mpl", fold=-1))


def draw_graph(g):
    fig = zx.draw_matplotlib(g, labels=False, figsize=(5, 2.4))
    return fig_to_img(fig)



def pad_to_aspect(img, ratio):
    """Pad img with white, centred, to width/height == ratio."""
    w, h = img.size
    if w / h < ratio:
        new_w, new_h = int(h * ratio), h
    else:
        new_w, new_h = w, int(w / ratio)
    out = Image.new("RGB", (new_w, new_h), "white")
    out.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return out

def agent_icon():
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    layers = [3, 4, 3]
    xs = [0.15, 0.5, 0.85]
    pos = []
    for x, nl in zip(xs, layers):
        ys = np.linspace(0.15, 0.85, nl)
        pos.append([(x, y) for y in ys])
    for l1, l2 in zip(pos[:-1], pos[1:]):
        for x1, y1 in l1:
            for x2, y2 in l2:
                ax.plot([x1, x2], [y1, y2], color="0.55", lw=0.8, zorder=1)
    for layer in pos:
        for x, y in layer:
            ax.scatter([x], [y], s=260, color="0.65", edgecolor="black", zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig_to_img(fig)


def main():
    # a small, legible example circuit (Riu et al. use a similar size);
    # resample until the pipeline visibly shrinks it, so the figure's
    # "optimized" panel is genuinely smaller than the original.
    # example pinned to seed 1 (16 gates, 5 CNOTs -> verified-equivalent
    # 10 gates, 3 CNOTs) so the published figure is reproducible
    random.seed(1)
    g0 = zx.generate.cliffordT(4, 16, p_t=0.2)
    c0 = zx.Circuit.from_graph(g0)
    g_like = circuit_to_graphlike(c0)
    g_env = g_like.copy()
    zx.simplify.lcomp_simp(g_env)
    c_ex = zx.extract_circuit(g_env.copy()).to_basic_gates()
    c_opt = zx.optimize.basic_optimization(c_ex.split_phase_gates())
    assert zx.compare_tensors(c0.to_tensor(), c_opt.to_tensor())
    print(f"example: original {len(c0.gates)} gates -> optimized {len(c_opt.gates)} gates (unitary equivalence verified)")

    img_orig = draw_circuit(c0)
    img_graph = draw_graph(g_like)
    img_env = draw_graph(g_env)
    img_extract = draw_circuit(c_ex)
    img_opt = draw_circuit(c_opt)

    img_agent = agent_icon()

    # ---- compose ----
    fig = plt.figure(figsize=(19, 6))

    def put(img, rect, title=None, boxed=False):
        ax = fig.add_axes(rect)
        ratio = (rect[2] * 19.0) / (rect[3] * 6.0)
        ax.imshow(pad_to_aspect(img, ratio), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(boxed)
        if title:
            ax.set_title(title, fontsize=13, style="italic", pad=6)
        return ax

    put(img_orig,    [0.005, 0.42, 0.16, 0.30], "Original circuit", boxed=True)
    put(img_graph,   [0.20,  0.42, 0.16, 0.30], "Graph representation", boxed=True)
    put(img_env,     [0.42,  0.55, 0.16,  0.30], "Environment", boxed=True)
    put(img_agent,   [0.44,  0.04, 0.12,  0.30], None, boxed=True)
    put(img_extract, [0.635, 0.42, 0.16, 0.30], "Circuit extraction", boxed=True)
    put(img_opt,     [0.835, 0.42, 0.16,  0.30], "Optimized circuit", boxed=True)

    fs = 13
    A = dict(arrowstyle="-|>", lw=2, color="black", mutation_scale=24)
    FA = plt.matplotlib.patches.FancyArrowPatch

    # original -> graph representation, with the sample's vertical
    # "Gate-based" annotation pointing at the transition
    fig.patches.append(FA((0.163, 0.57), (0.198, 0.57), transform=fig.transFigure, **A))
    fig.text(0.181, 0.775, "Gate-based", fontsize=fs, style="italic", ha="center")
    fig.patches.append(FA((0.181, 0.755), (0.181, 0.60), transform=fig.transFigure, **A))

    # graph output line forks: Action branch curves up into the
    # Environment; the lower branch sweeps down to the Agent
    fig.lines.append(plt.matplotlib.lines.Line2D([0.362, 0.395], [0.57, 0.57],
        transform=fig.transFigure, color="black", lw=2))
    fig.patches.append(FA((0.395, 0.57), (0.417, 0.645), transform=fig.transFigure,
        connectionstyle="arc3,rad=0.55", **A))
    fig.text(0.396, 0.745, "Action", fontsize=fs, style="italic", ha="center")
    fig.patches.append(FA((0.395, 0.57), (0.437, 0.22), transform=fig.transFigure,
        connectionstyle="arc3,rad=0.35", arrowstyle="-", lw=2, color="black"))

    # Observation: straight vertical drop from Environment to Agent
    fig.patches.append(FA((0.505, 0.545), (0.505, 0.35), transform=fig.transFigure, **A))
    fig.text(0.468, 0.50, "Observation", fontsize=12, style="italic", ha="center")
    fig.text(0.50, 0.015, "Agent (GATv2 policy)", fontsize=fs, style="italic", ha="center")

    # Environment -> Circuit extraction: dash-dot connector
    fig.patches.append(FA((0.585, 0.68), (0.632, 0.68), transform=fig.transFigure,
        linestyle=(0, (6, 2, 1, 2)), **A))

    # Reward: from the extraction down into the Agent
    fig.patches.append(FA((0.685, 0.38), (0.567, 0.12), transform=fig.transFigure,
        connectionstyle="arc3,rad=-0.2", **A))
    fig.text(0.685, 0.21, "Winning trajectories\n(behaviour cloning)", fontsize=fs - 1, style="italic", ha="center")

    # Basic-opt annotation, then the red Stop arrow into the output
    fig.text(0.813, 0.775, "Basic-opt", fontsize=fs, style="italic", ha="center")
    fig.patches.append(FA((0.813, 0.755), (0.813, 0.60), transform=fig.transFigure, **A))
    red = dict(arrowstyle="-|>", lw=2, color="red", mutation_scale=24)
    fig.patches.append(FA((0.795, 0.57), (0.832, 0.57), transform=fig.transFigure, **red))
    fig.text(0.813, 0.51, "Stop", fontsize=fs, style="italic", ha="center", color="red")

    fig.savefig(OUT, dpi=180, facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
