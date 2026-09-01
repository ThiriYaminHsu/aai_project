"""Two additional raw training-circuit examples each for Agent R and
Agent P, to visualise within-family variation (user request):

- Agent R: two more independent draws at the fixed 5 qubits / 70 gates /
  17-24-25-34 recipe -- same shape every episode, different gate
  placement.
- Agent P: one draw forced from the small end and one from the large end
  of its 2--15 qubit / 10--200 gate ranges, to make the size variation
  visible side by side. (The generator draws uniformly; the forced bands
  here are for illustration and the caption says the ranges.)

Writes training_circuits_agent_r_raw_{2,3}.png and
training_circuits_agent_p_raw_{small,large}.png.

Usage:
    python scripts/plot_training_circuits_variation.py
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

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"


def save(g, name: str, note: str):
    circ = zx.Circuit.from_graph(g) if not isinstance(g, zx.Circuit) else g
    qasm = circ.to_basic_gates().to_qasm()
    qc = QuantumCircuit.from_qasm_str(qasm)
    fig = qc.draw(output="mpl", fold=-1)
    out = LATEX_FIGS / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}: {circ.qubits} qubits, {len(circ.gates)} gates ({note})")


def main():
    for i in (2, 3):
        g = zx.generate.cliffordT(5, 70, p_t=0.17, p_s=0.24, p_hsh=0.25)
        save(g, f"training_circuits_agent_r_raw_{i}.png", f"Agent R draw {i}")

    nq, ng = random.randint(2, 4), random.randint(10, 30)
    save(zx.generate.cliffordT(nq, ng), "training_circuits_agent_p_raw_small.png", "Agent P small-end draw")

    nq, ng = random.randint(10, 14), random.randint(100, 150)
    save(zx.generate.cliffordT(nq, ng), "training_circuits_agent_p_raw_large.png", "Agent P large-end draw")


if __name__ == "__main__":
    main()
