"""Render the random circuit's ZX-diagram after each *pyzx built-in* deterministic
optimiser: full_reduce, basic_optimization+full_reduce, clifford_simp, and
teleport_reduce. All four are genuine pyzx 0.10.5 library functions (verified via
inspect.getfile against site-packages/pyzx/), not project-written code -- unlike
the cflow-zx approximation (causal_flow.py/baselines.baseline_cflow_opt), which
found no improving single move on this circuit and so is omitted here (its
"after" diagram would be identical to the original).

Usage:
    python scripts/plot_random_circuit_baselines.py
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyzx as zx
from PIL import Image, ImageDraw

from circopt_adder.zx_utils import circuit_to_graphlike, extract_and_cleanup, gate_count, two_qubit_gate_count, t_count

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures"

GATES = [
    ("single", 1, "T"), ("cnot", 0, 2), ("single", 3, "H"), ("cnot", 2, 1),
    ("single", 0, "S"), ("cnot", 1, 3), ("single", 0, "H"), ("single", 2, "T"),
]
N_QUBITS = 4


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


def render(g) -> Image.Image:
    fig = zx.draw(g, labels=False, figsize=(4.2, 3.0))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    circuit = build_circuit()
    g0 = circuit_to_graphlike(circuit)

    panels = []

    panels.append(("original", g0.copy(), circuit))

    g_fr = g0.copy()
    zx.simplify.full_reduce(g_fr)
    panels.append(("full_reduce", g_fr, extract_and_cleanup(g_fr.copy())))

    c_bo = zx.basic_optimization(circuit.to_basic_gates())
    g_bo = c_bo.to_graph()
    zx.simplify.full_reduce(g_bo)
    panels.append(("basic_opt+full_reduce", g_bo, extract_and_cleanup(g_bo.copy())))

    g_cs = g0.copy()
    zx.simplify.clifford_simp(g_cs)
    panels.append(("clifford_simp", g_cs, extract_and_cleanup(g_cs.copy())))

    g_tr = g0.copy()
    zx.simplify.teleport_reduce(g_tr)
    panels.append(("teleport_reduce", g_tr, extract_and_cleanup(g_tr.copy())))

    imgs = []
    labels = []
    for name, g, c_out in panels:
        imgs.append(render(g))
        if c_out is not None:
            labels.append(f"{name}\ngates={gate_count(c_out)}  2q={two_qubit_gate_count(c_out)}  t={t_count(c_out)}")
        else:
            labels.append(f"{name}\n(extraction failed)")

    target_h = 320
    def scale(img):
        w = int(img.width * target_h / img.height)
        return img.resize((w, target_h), Image.LANCZOS)
    imgs = [scale(im) for im in imgs]

    gap = 40
    label_h = 46
    canvas_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    canvas_h = target_h + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label in zip(imgs, labels):
        canvas.paste(img, (x, label_h))
        for i, line in enumerate(label.split("\n")):
            draw.text((x, 4 + i * 16), line, fill="#33322f")
        x += img.width + gap

    out_path = OUT_DIR / "random_circuit_baselines.png"
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
