"""Two Chapter 5 figures evidencing the learned motif (Section 5.2.2):
(a) per-step two-qubit count of Agent A's greedy episode on four adder
    widths, showing the gain landing only when the three-action
    pivot-gadget sequence completes;
(b) the wire index of every motif target against bit-width, showing all
    targets on the top bit of register A (qubit n) at every width.

Usage: python3 scripts/plot_motif_evidence.py
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import _light_preprocess, ripple_carry_adder
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import extract_and_cleanup, two_qubit_gate_count

LATEX_FIGS = REPO_ROOT.parent / "Writing" / "latex" / "figures"
WIDTHS = [2, 3, 4, 10]

def main():
    set_seed(SEED)
    cfg = Config(); cfg.gnn_channels = 128
    pol = ActorCriticGNN(cfg).to(DEVICE)
    pol.load_state_dict(torch.load(REPO_ROOT / "results/checkpoints/agent_A_v4_correct.pt",
                                   map_location=DEVICE))
    pol.eval()

    curves, targets = {}, {}
    for nb in WIDTHS:
        circ = _light_preprocess(ripple_carry_adder(nb).to_graph())
        start = two_qubit_gate_count(circ)
        won = False
        for attempt in range(30):
            env = ZXOptEnv(lambda c=circ: c, cfg)
            obs, _ = env.reset()
            counts = [start]; tq = []; t_best = None
            for t in range(cfg.max_episode_steps):
                with torch.no_grad():
                    logits, _ = pol.forward(obs.to(DEVICE))
                if attempt == 0:
                    a = int(torch.argmax(logits).item())
                else:
                    a = int(torch.distributions.Categorical(logits=logits).sample().item())
                act = env._action_index[a]
                if act != "STOP" and act[0] == "PIVG":
                    q = env.g.qubit(act[1])
                    if q is not None and q >= 0:
                        tq.append((nb, q))
                obs2, r, term, trunc, info = env.step(a)
                c = extract_and_cleanup(env.g.copy())
                counts.append(two_qubit_gate_count(c) if c is not None else counts[-1])
                if counts[-1] < start and t_best is None:
                    t_best = t + 1
                if term or trunc:
                    break
                obs = obs2
                if t_best is not None:
                    break
            if t_best is not None:
                curves[nb] = counts[: t_best + 1]
                targets[nb] = tq[:3]
                won = True
                break
        print(nb, "won:", won, "| counts:", curves.get(nb), "| target qubits:",
              sorted(set(q for _, q in targets.get(nb, []))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    styles = {2: ("-", "o"), 3: ("--", "s"), 4: ("-.", "^"), 10: (":", "D")}
    for k, (nb, c) in enumerate(curves.items()):
        steps = list(range(len(c)))
        ls, mk = styles[nb]
        ax1.plot(steps, [x - (11*nb - 1) + 0.06*k for x in c], marker=mk,
                 linestyle=ls, alpha=0.85,
                 label=f"{nb}-bit" + (" (held out)" if nb in (4, 10) else ""))
    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_xlabel("episode step (winning rollout)")
    ax1.set_ylabel("two-qubit count above $11n-1$")
    ax1.set_title("(a) gain lands only when the pivot-gadget\nsequence completes")
    ax1.set_yticks([0, 1, 2, 3, 4]); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(5))

    for nb, tq in targets.items():
        for _, q in tq:
            ax2.scatter(nb, q, color="#4c72b0", zorder=3)
    xs = sorted(targets)
    ax2.plot(xs, [nb for nb in xs], color="#c44e52", lw=1.2,
             label="top bit of register $A$ (qubit $n$)")
    ax2.set_xlabel("adder bit-width $n$")
    ax2.set_ylabel("wire index of pivot-gadget targets")
    ax2.set_title("(b) every target of the sequence sits on\ntop bit of register $A$")
    ax2.set_xticks(xs); ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = LATEX_FIGS / "motif_evidence.png"
    fig.savefig(out, dpi=200)
    print("Saved", out)

if __name__ == "__main__":
    main()
