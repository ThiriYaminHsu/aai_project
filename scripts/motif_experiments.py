"""Motif-exploitation experiments (user go-ahead, 2026-09-01/02):

Experiment 1 -- exhaustive PIVG-triple search on the verified adders.
For widths 2-4: fully exhaustive over all sequences of 1..3 pivot-gadget
applications. For widths 5, 7, 10: exhaustive over LOCAL triples, where
actions after the first must touch a vertex within 2 hops of the first
action's vertices (the shape of the agent's discovered motif). Reports
the best two-qubit count reachable within the motif class per width.

Experiment 2 -- trace whether Agent A's real-world wins are PIVG-motif
wins: greedy + up to 10 sampled rollouts per arithmetic benchmark
circuit; for each winning episode, log the action-type sequence up to
its best point.

Usage:
    python3 scripts/motif_experiments.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.fetch_paper_circuits import load_all as load_paper
from circopt_adder.config import DEVICE, Config
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import _light_preprocess, ripple_carry_adder
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import (
    apply_pivot_gadget,
    circuit_to_graphlike,
    extract_and_cleanup,
    find_feasible_pivot_gadget,
    two_qubit_gate_count,
)

ARITH = ("add", "qcla", "mod")


def count(g):
    c = extract_and_cleanup(g)
    return None if c is None else two_qubit_gate_count(c)


def ball(g, verts, radius=2):
    seen = set(verts)
    frontier = set(verts)
    for _ in range(radius):
        nxt = set()
        for v in frontier:
            if v in g.graph:
                nxt |= set(g.neighbors(v))
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return seen


def exhaustive_triples(g0, start, local=None):
    """DFS over 1..3 PIVG applications; returns (best, n_nodes).

    pyzx's graph.copy() renumbers vertices, so action tuples enumerated
    on a parent graph do not address the same vertices on its copy. We
    therefore enumerate BY INDEX: feasible actions are re-enumerated on
    each fresh copy (deterministic order for identical structure), and
    the i-th action of the parent corresponds to the i-th of the copy.
    """
    best = [start]
    nodes = [0]

    def rec(g, depth, allowed):
        feas = list(find_feasible_pivot_gadget(g))
        for i, (v, w) in enumerate(feas):
            if allowed is not None and v not in allowed and w not in allowed:
                continue
            g2 = g.copy()
            feas2 = list(find_feasible_pivot_gadget(g2))
            if len(feas2) != len(feas):
                continue  # order correspondence broken; skip defensively
            v2, w2 = feas2[i]
            try:
                apply_pivot_gadget(g2, v2, w2)
            except Exception:
                continue
            nodes[0] += 1
            m = count(g2)
            if m is not None and m < best[0]:
                best[0] = m
            if depth < 3:
                nxt_allowed = allowed
                if local is not None and allowed is None:
                    nxt_allowed = ball(g2, [x for x in (v2, w2) if x in g2.graph], local)
                rec(g2, depth + 1, nxt_allowed)

    rec(g0, 1, None)
    return best[0], nodes[0]


def experiment1():
    print("=== Experiment 1: exhaustive PIVG triples on adders ===", flush=True)
    for nb, local in ((2, None), (3, None), (4, None), (5, 2), (7, 2), (10, 2)):
        circ = _light_preprocess(ripple_carry_adder(nb).to_graph())
        start = two_qubit_gate_count(circ)
        g = circuit_to_graphlike(circ)
        mode = "full" if local is None else f"local(r={local})"
        best, n = exhaustive_triples(g, start, local=local)
        print(f"[E1] {nb}-bit ({mode}): start={start} best-in-motif-class={best} "
              f"({n} sequences evaluated)", flush=True)


def experiment2():
    print("=== Experiment 2: trace Agent A wins on real-world circuits ===", flush=True)
    cfg = Config()
    cfg.gnn_channels = 128
    pol = ActorCriticGNN(cfg).to(DEVICE)
    pol.load_state_dict(torch.load(REPO_ROOT / "results/checkpoints/agent_A_v4_correct.pt",
                                   map_location=DEVICE))
    pol.eval()
    torch.manual_seed(0)

    def episode(circ, greedy):
        env = ZXOptEnv(lambda c=circ: c, cfg)
        obs, _ = env.reset()
        trace, best, t_best = [], env.best_metric, -1
        for t in range(cfg.max_episode_steps):
            with torch.no_grad():
                logits, _ = pol.forward(obs.to(DEVICE))
                a = int(torch.argmax(logits).item()) if greedy else \
                    torch.distributions.Categorical(logits=logits).sample().item()
            act = env._action_index[a]
            kind = act if act == "STOP" else act[0]
            obs2, r, term, trunc, info = env.step(a)
            trace.append(kind)
            if env.best_metric < best:
                best, t_best = env.best_metric, t
            if term or trunc:
                break
            obs = obs2
        return best, t_best, trace

    paper = load_paper()
    for name, c in sorted(paper.items()):
        if not any(k in name.lower() for k in ARITH):
            continue
        prep = _light_preprocess(c.to_basic_gates().to_graph())
        start = two_qubit_gate_count(prep)
        win = None
        b, tb, tr = episode(prep, True)
        if b < start:
            win = ("greedy", b, tb, tr)
        else:
            for k in range(10):
                b, tb, tr = episode(prep, False)
                if b < start:
                    win = (f"sample{k}", b, tb, tr)
                    break
        if win is None:
            print(f"[E2] {name}: no win (start {start})", flush=True)
        else:
            mode, b, tb, tr = win
            print(f"[E2] {name} ({mode}): start={start} best={b} "
                  f"actions-to-best={tr[:tb+1]}", flush=True)


if __name__ == "__main__":
    experiment1()
    experiment2()
