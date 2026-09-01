"""Apply the three trained agents (R, P, A -- the actual reported checkpoints,
results/checkpoints/agent_{R_random,P_pure_random,A_adder}.pt) to the same example
random circuit used throughout the dissertation's ZX-calculus exposition, under the
same best-of-10 evaluation protocol documented in the Methodology chapter
(run_agent_eval_best_of_n: 1 deterministic + 9 stochastic attempts, keep the best).

Usage:
    python scripts/plot_random_circuit_agents.py
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyzx as zx
import torch
from PIL import Image, ImageDraw

from circopt_adder.config import DEVICE, Config
from circopt_adder.evaluation import run_agent_eval_best_of_n
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import gate_count, two_qubit_gate_count, t_count, circuit_to_graphlike

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures"
CKPT_DIR = REPO_ROOT / "results" / "checkpoints"

GATES = [
    ("single", 1, "T"), ("cnot", 0, 2), ("single", 3, "H"), ("cnot", 2, 1),
    ("single", 0, "S"), ("cnot", 1, 3), ("single", 0, "H"), ("single", 2, "T"),
]
N_QUBITS = 4

AGENTS = [
    ("Agent R", "agent_R_random.pt"),
    ("Agent P", "agent_P_pure_random.pt"),
    ("Agent A", "agent_A_adder.pt"),
]


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


def load_policy(cfg: Config, name: str) -> ActorCriticGNN:
    policy = ActorCriticGNN(cfg).to(DEVICE)
    policy.load_state_dict(torch.load(CKPT_DIR / name, map_location=DEVICE))
    policy.eval()
    return policy


def render(g) -> Image.Image:
    fig = zx.draw(g, labels=False, figsize=(4.2, 3.0))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    cfg = Config()
    circuit = build_circuit()

    panels = [("original", circuit_to_graphlike(circuit), gate_count(circuit),
               two_qubit_gate_count(circuit), t_count(circuit))]

    for label, ckpt in AGENTS:
        policy = load_policy(cfg, ckpt)
        best_circuit, gates, t, twoq = run_agent_eval_best_of_n(policy, circuit, cfg, n_samples=10)
        panels.append((label, circuit_to_graphlike(best_circuit), gates, twoq, t))
        print(f"{label}: gates={gates} 2q={twoq} t={t}")

    imgs = [render(g) for _, g, *_ in panels]
    labels = [f"{name}\ngates={gates}  2q={twoq}  t={t}" for name, _, gates, twoq, t in panels]

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

    out_path = OUT_DIR / "random_circuit_agents.png"
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
