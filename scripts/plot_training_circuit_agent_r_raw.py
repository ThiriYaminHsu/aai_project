"""Regenerate Figure 3.1 (Agent R example training circuit) showing the
RAW circuit exactly as zx.generate.cliffordT produces it (5 qubits, 70
gates, Riu et al.'s 17/24/25/34 gate-type recipe), BEFORE
_light_preprocess -- per the user's request to show the unprocessed
input. Drawn in the same qiskit-mpl style as the existing
training-circuit figures (via QASM round-trip).

Writes Writing/latex/figures/training_circuits_agent_r_raw.png.

Usage:
    python scripts/plot_training_circuit_agent_r_raw.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import pyzx as zx
from qiskit import QuantumCircuit

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"


def main():
    g = zx.generate.cliffordT(5, 70, p_t=0.17, p_s=0.24, p_hsh=0.25)
    circ = zx.Circuit.from_graph(g) if not isinstance(g, zx.Circuit) else g
    qasm = circ.to_basic_gates().to_qasm()
    qc = QuantumCircuit.from_qasm_str(qasm)
    fig = qc.draw(output="mpl", fold=-1)
    out = LATEX_FIGS / "training_circuits_agent_r_raw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}: {circ.qubits} qubits, {len(circ.gates)} gates (raw, unpreprocessed)")


if __name__ == "__main__":
    main()
