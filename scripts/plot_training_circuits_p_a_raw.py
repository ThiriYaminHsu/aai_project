"""Regenerate the Agent P and Agent A example training-circuit figures
showing the RAW circuits, before _light_preprocess, matching the raw
Agent R figure (plot_training_circuit_agent_r_raw.py).

Agent P: one draw from its own recipe (2--15 qubits, 10--200 gates,
uniform gate types). Draws are resampled until one falls in a legible
size band for a full-width figure (<= 8 qubits, <= 90 gates); the figure
is an illustrative example, and the caption already notes each episode
resamples its size.

Agent A: the 2-bit Cuccaro adder, deterministic.

Writes training_circuits_agent_p_raw.png and
training_circuits_agent_a_2bit_raw.png into Writing/latex/figures.

Usage:
    python scripts/plot_training_circuits_p_a_raw.py
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import pyzx as zx
from qiskit import QuantumCircuit

from circopt_adder.generators import ripple_carry_adder

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"


def save(circ: "zx.Circuit", name: str, note: str):
    qasm = circ.to_basic_gates().to_qasm()
    qc = QuantumCircuit.from_qasm_str(qasm)
    fig = qc.draw(output="mpl", fold=-1)
    out = LATEX_FIGS / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}: {circ.qubits} qubits, {len(circ.gates)} gates ({note})")


def main():
    for _ in range(100):
        nq = random.randint(2, 15)
        ng = random.randint(10, 200)
        g = zx.generate.cliffordT(nq, ng)
        if nq <= 8 and ng <= 90:
            circ = zx.Circuit.from_graph(g) if not isinstance(g, zx.Circuit) else g
            save(circ, "training_circuits_agent_p_raw.png", "raw Agent P draw")
            break

    adder = ripple_carry_adder(2)
    save(adder, "training_circuits_agent_a_2bit_raw.png", "raw 2-bit adder")


if __name__ == "__main__":
    main()
