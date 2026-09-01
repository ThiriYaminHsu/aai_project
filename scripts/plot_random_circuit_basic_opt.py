"""Render the example circuit before/after *just* pyzx's basic_optimization
(no full_reduce) -- a circuit-level (not graph-based) peephole pass, so shown
as a gate-box circuit diagram (via Qiskit) rather than a ZX-diagram, since
that's the representation the algorithm actually operates on.

Usage:
    python scripts/plot_random_circuit_basic_opt.py
"""

import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyzx as zx
from PIL import Image, ImageDraw
from qiskit import qasm2

from circopt_adder.zx_utils import gate_count, two_qubit_gate_count, t_count

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures"

GATES = [
    ("single", 1, "T"), ("cnot", 0, 2), ("single", 3, "H"), ("cnot", 2, 1),
    ("single", 0, "S"), ("cnot", 1, 3), ("single", 0, "H"), ("single", 2, "T"),
]
N_QUBITS = 4

_RZ_TO_NAMED = [
    (r"0\.25\*pi", "t"), (r"1\.75\*pi", "tdg"), (r"-0\.25\*pi", "tdg"),
    (r"0\.5\*pi", "s"), (r"1\.5\*pi", "sdg"), (r"-0\.5\*pi", "sdg"),
    (r"1\.0\*pi", "z"), (r"-1\.0\*pi", "z"),
]


def _prettify_qasm_phases(qasm: str) -> str:
    for pattern, gate in _RZ_TO_NAMED:
        qasm = re.sub(rf"rz\({pattern}\) (q\[\d+\]);", rf"{gate} \1;", qasm)
    return qasm


def build_circuit() -> "zx.Circuit":
    c = zx.Circuit(N_QUBITS)
    for gate in GATES:
        if gate[0] == "single":
            _, q, label = gate
            c.add_gate(label, q)
        else:
            _, control, target = gate
            c.add_gate("CNOT", control, target)
    return c


def render_gate_box(circuit: "zx.Circuit") -> Image.Image:
    qasm = _prettify_qasm_phases(circuit.to_basic_gates().to_qasm())
    qc = qasm2.loads(qasm)
    fig = qc.draw("mpl", fold=-1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    circuit = build_circuit()
    c_bo = zx.basic_optimization(circuit.to_basic_gates())

    img_before = render_gate_box(circuit)
    img_after = render_gate_box(c_bo)

    target_h = 260
    def scale(img):
        w = int(img.width * target_h / img.height)
        return img.resize((w, target_h), Image.LANCZOS)
    img_before, img_after = scale(img_before), scale(img_after)

    label_h = 40
    gap = 30
    canvas_w = img_before.width + gap + img_after.width
    canvas_h = target_h + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img_before, (0, label_h))
    canvas.paste(img_after, (img_before.width + gap, label_h))
    draw.text((0, 4), f"original -- gates={gate_count(circuit)} 2q={two_qubit_gate_count(circuit)} t={t_count(circuit)}", fill="#33322f")
    draw.text((img_before.width + gap, 4), f"basic_optimization only -- gates={gate_count(c_bo)} 2q={two_qubit_gate_count(c_bo)} t={t_count(c_bo)}", fill="#33322f")

    out_path = OUT_DIR / "random_circuit_basic_opt.png"
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")

    print()
    print("original gates:  ", [str(g) for g in circuit.gates])
    print("basic_opt gates: ", [str(g) for g in c_bo.gates])


if __name__ == "__main__":
    main()
