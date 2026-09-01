"""CNOT gate visualisation: standard quantum-circuit notation next to its
ZX-calculus diagram equivalent (a Z-spider connected to an X-spider by a plain
wire) -- the first concrete instance of the circuit-to-ZX-diagram translation
central to this dissertation (Chapter 2/3), not just a standalone gate symbol.

Usage:
    python scripts/plot_cnot_gate.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

Z_GREEN = "#5fae5f"
X_RED = "#d9534f"


def draw_circuit_panel(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    y_control, y_target = 1.8, 0.6
    ax.plot([0.3, 3.7], [y_control, y_control], color="black", lw=1.3, zorder=1)
    ax.plot([0.3, 3.7], [y_target, y_target], color="black", lw=1.3, zorder=1)
    ax.text(0.05, y_control, r"$q_0$", ha="right", va="center", fontsize=13)
    ax.text(0.05, y_target, r"$q_1$", ha="right", va="center", fontsize=13)

    x_gate = 2.0
    ax.plot([x_gate, x_gate], [y_target, y_control], color="black", lw=1.6, zorder=2)
    ax.add_patch(Circle((x_gate, y_control), 0.09, color="black", zorder=3))

    r = 0.24
    ax.add_patch(Circle((x_gate, y_target), r, facecolor="white",
                         edgecolor="black", lw=1.6, zorder=3))
    ax.plot([x_gate - r, x_gate + r], [y_target, y_target], color="black", lw=1.4, zorder=4)
    ax.plot([x_gate, x_gate], [y_target - r, y_target + r], color="black", lw=1.4, zorder=4)

    ax.text(x_gate, y_control + 0.32, "control", ha="center", fontsize=9, color="dimgray")
    ax.text(x_gate, y_target - 0.42, "target", ha="center", fontsize=9, color="dimgray")
    ax.set_title("Circuit notation", fontsize=12, pad=10)


def draw_zx_panel(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    y_z, y_x = 1.8, 0.6
    x_spider = 2.0

    ax.plot([0.3, x_spider], [y_z, y_z], color="black", lw=1.1, zorder=1)
    ax.plot([x_spider, 3.7], [y_z, y_z], color="black", lw=1.1, zorder=1)
    ax.plot([0.3, x_spider], [y_x, y_x], color="black", lw=1.1, zorder=1)
    ax.plot([x_spider, 3.7], [y_x, y_x], color="black", lw=1.1, zorder=1)
    ax.plot([x_spider, x_spider], [y_x, y_z], color="black", lw=1.3, zorder=1)

    ax.add_patch(Circle((x_spider, y_z), 0.16, facecolor=Z_GREEN,
                         edgecolor="black", lw=1.0, zorder=3))
    ax.add_patch(Circle((x_spider, y_x), 0.16, facecolor=X_RED,
                         edgecolor="black", lw=1.0, zorder=3))

    ax.text(x_spider, y_z + 0.34, "$Z$", ha="center", fontsize=10, color="dimgray")
    ax.text(x_spider, y_x - 0.4, "$X$", ha="center", fontsize=10, color="dimgray")
    ax.set_title("ZX-diagram", fontsize=12, pad=10)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    draw_circuit_panel(axes[0])
    draw_zx_panel(axes[1])
    fig.text(0.5, 0.5, "=", fontsize=22, ha="center", va="center")
    fig.tight_layout()
    fig.savefig("results/figures/cnot_gate.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    print("Saved results/figures/cnot_gate.png")


if __name__ == "__main__":
    main()
