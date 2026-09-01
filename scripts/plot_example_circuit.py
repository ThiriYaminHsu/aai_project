"""A small illustrative quantum circuit -- not tied to any experiment -- for the
Literature Review's general "what is a quantum circuit" exposition: a few
single-qubit gates and two CNOTs on three qubits, read left to right, standard
textbook circuit notation.

Usage:
    python scripts/plot_example_circuit.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

N_QUBITS = 3
GATES = [
    ("single", 0, "H"),
    ("cnot", 0, 1),
    ("single", 1, "T"),
    ("cnot", 1, 2),
    ("single", 0, "S"),
    ("single", 2, "H"),
]

WIRE_GAP = 1.0
COL_GAP = 1.0
MARGIN = 0.8


def qubit_y(q: int) -> float:
    return (N_QUBITS - 1 - q) * WIRE_GAP


def draw_single_qubit_gate(ax, x, y, label):
    size = 0.42
    box = FancyBboxPatch((x - size / 2, y - size / 2), size, size,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor="white", edgecolor="black", lw=1.4, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=13, zorder=4)


def draw_cnot(ax, x, y_control, y_target):
    ax.plot([x, x], [min(y_control, y_target), max(y_control, y_target)],
            color="black", lw=1.5, zorder=2)
    ax.add_patch(Circle((x, y_control), 0.08, color="black", zorder=3))
    r = 0.22
    ax.add_patch(Circle((x, y_target), r, facecolor="white", edgecolor="black",
                         lw=1.5, zorder=3))
    ax.plot([x - r, x + r], [y_target, y_target], color="black", lw=1.3, zorder=4)
    ax.plot([x, x], [y_target - r, y_target + r], color="black", lw=1.3, zorder=4)


def main() -> None:
    n_cols = len(GATES)
    width = MARGIN * 2 + COL_GAP * (n_cols + 1)
    height = WIRE_GAP * (N_QUBITS - 1) + 1.2

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.6, height - 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    x_start = MARGIN
    x_end = width - MARGIN
    for q in range(N_QUBITS):
        y = qubit_y(q)
        ax.plot([x_start, x_end], [y, y], color="black", lw=1.1, zorder=1)
        ax.text(x_start - 0.25, y, rf"$q_{q}$", ha="right", va="center", fontsize=13)

    for i, gate in enumerate(GATES):
        x = x_start + COL_GAP * (i + 1)
        if gate[0] == "single":
            _, q, label = gate
            draw_single_qubit_gate(ax, x, qubit_y(q), label)
        else:
            _, qc, qt = gate
            draw_cnot(ax, x, qubit_y(qc), qubit_y(qt))

    fig.tight_layout()
    fig.savefig("results/figures/example_circuit.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    print("Saved results/figures/example_circuit.png")


if __name__ == "__main__":
    main()
