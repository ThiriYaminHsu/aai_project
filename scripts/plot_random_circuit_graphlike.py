"""Render the Chapter 2 example circuit's ZX-diagram beside its graph-like
form (after pyzx's to_gh colour-change conversion), for the new literature
review figure explaining graph-like diagrams. Reuses the exact circuit of
plot_random_circuit_zx_diagram.py.

Usage: python3 scripts/plot_random_circuit_graphlike.py
"""
import io, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyzx as zx
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from plot_random_circuit_zx_diagram import build_circuit, render_zx_diagram

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"

def main():
    circuit = build_circuit()
    g = circuit.to_graph()
    gl = circuit.to_graph()
    zx.simplify.to_gh(gl)          # colour change: all spiders green, wires Hadamard
    zx.simplify.spider_simp(gl)   # fuse adjacent same-colour spiders
    zx.simplify.id_simp(gl)       # remove identity spiders
    img_a = render_zx_diagram(g)
    img_b = render_zx_diagram(gl)

    target_h = 380
    def scale(img):
        w = int(img.width * target_h / img.height)
        return img.resize((w, target_h), Image.LANCZOS)
    img_a, img_b = scale(img_a), scale(img_b)

    gap, label_h = 70, 30
    canvas = Image.new("RGB", (img_a.width + gap + img_b.width, target_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img_a, (0, label_h))
    canvas.paste(img_b, (img_a.width + gap, label_h))
    y = label_h + target_h // 2
    x0, x1 = img_a.width + 12, img_a.width + gap - 14
    draw.line([(x0, y), (x1, y)], fill="black", width=2)
    draw.polygon([(x1, y-6), (x1, y+6), (x1+10, y)], fill="black")
    draw.text((img_a.width//2 - 45, 4), "ZX-diagram", fill="#52514e")
    draw.text((img_a.width + gap + img_b.width//2 - 60, 4), "graph-like form", fill="#52514e")
    out = LATEX_FIGS / "random_circuit_graphlike.png"
    canvas.save(out)
    print(f"Saved {out} | before: {g.num_vertices()} spiders | after: {gl.num_vertices()} spiders")

if __name__ == "__main__":
    main()
