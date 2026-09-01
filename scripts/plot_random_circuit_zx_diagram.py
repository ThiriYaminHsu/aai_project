"""Render the exact random circuit from plot_random_circuit_example.py as its
ZX-diagram (circuit.to_graph() -- the direct, gate-by-gate translation into ZX
generators, *before* simplification into graph-like form), side by side with
the gate-box circuit for comparison. Uses pyzx's own zx.draw() rather than a
hand-rolled renderer, so spider colours/shapes and Hadamard edges are authentic.

Usage:
    python scripts/plot_random_circuit_zx_diagram.py
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyzx as zx
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures"

# Must match plot_random_circuit_example.py's GATES exactly.
GATES = [
    ("single", 1, "T"),
    ("cnot", 0, 2),
    ("single", 3, "H"),
    ("cnot", 2, 1),
    ("single", 0, "S"),
    ("cnot", 1, 3),
    ("single", 0, "H"),
    ("single", 2, "T"),
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


def render_zx_diagram(g) -> Image.Image:
    fig = zx.draw(g, labels=False, figsize=(6, 3.2))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    circuit = build_circuit()
    g = circuit.to_graph()
    zx_img = render_zx_diagram(g)

    circuit_img = Image.open(OUT_DIR / "random_circuit_example.png").convert("RGB")

    # -- scale both panels to a shared height, then compose side by side --
    target_h = 380
    def scale(img):
        w = int(img.width * target_h / img.height)
        return img.resize((w, target_h), Image.LANCZOS)

    circuit_img = scale(circuit_img)
    zx_img = scale(zx_img)

    gap = 70
    label_h = 30
    canvas_w = circuit_img.width + gap + zx_img.width
    canvas_h = target_h + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(circuit_img, (0, label_h))
    canvas.paste(zx_img, (circuit_img.width + gap, label_h))

    arrow_y = label_h + target_h // 2
    arrow_x0 = circuit_img.width + 12
    arrow_x1 = circuit_img.width + gap - 14
    draw.line([(arrow_x0, arrow_y), (arrow_x1, arrow_y)], fill="black", width=2)
    draw.polygon([(arrow_x1, arrow_y - 6), (arrow_x1, arrow_y + 6), (arrow_x1 + 10, arrow_y)],
                 fill="black")

    draw.text((circuit_img.width // 2 - 30, 4), "circuit", fill="#52514e")
    draw.text((circuit_img.width + gap + zx_img.width // 2 - 45, 4), "ZX-diagram", fill="#52514e")

    out_path = OUT_DIR / "random_circuit_zx_diagram.png"
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height}) -- "
          f"{g.num_vertices()} spiders, {g.num_edges()} edges")


if __name__ == "__main__":
    main()
