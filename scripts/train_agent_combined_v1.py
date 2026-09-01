"""Combined agent v1: one policy distilled from the pooled winning
trajectories of ALL THREE training families (user request, 2026-09-01,
overnight run). Exploratory -- NOT part of the dissertation's controlled
ablation; it answers "does mixing structured and random data preserve
Agent A's motif?".

Recipe = the final agents' own recipes, pooled:
  adders          -- Agent A v4 pipeline: blind chained search
                     (150/round, <=3 rounds) on widths {2,3,5,6,7,8,9},
                     then guided chained search (250/round, <=4 rounds).
  structured/pure -- R/P v3 pipeline: 150 fresh circuits x 15 blind
                     rollouts each, one winning trajectory per circuit.
  distills        -- intermediate policy from pooled blind pairs (guides
                     the adder search); FINAL policy distilled fresh from
                     guided adder pairs + R/P winners (150 epochs, 128ch).

Evaluation (best-of-10 = 1 greedy + 9 sampled, matching Chapter 4):
  adders 2-10 (incl. held-out 4, 10), 5 structured-random, 5 pure-random
  (untrained-network control alongside), 11 real-world arithmetic.

Saves results/checkpoints/agent_combined_v1.pt and
results/logs/agent_combined_eval.csv.

Usage:
    python3 scripts/train_agent_combined_v1.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch

import train_agent_a_v3 as v3
import scripts.expert_iteration_agents_rp as rp
from benchmarks.fetch_paper_circuits import load_all as load_paper
from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.generators import (
    _light_preprocess,
    make_pure_random_circuit_generator,
    make_random_circuit_generator,
    ripple_carry_adder,
)
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import two_qubit_gate_count

TRAIN_BITS = [2, 3, 5, 6, 7, 8, 9]
HELDOUT_BITS = [4, 10]
ARITH = ("add", "qcla", "mod")
EPOCHS = 150

rp.N_TRAIN, rp.N_ROLLOUTS = 150, 15


def best_of_10(policy, cfg, circ):
    start = two_qubit_gate_count(circ)
    best, _, _ = v3.episode(cfg, circ, policy=policy, greedy=True)
    for _ in range(9):
        m, _, _ = v3.episode(cfg, circ, policy=policy)
        best = min(best, m)
    return start, best


def main():
    set_seed(SEED)
    cfg = v3.make_cfg()
    adders = {b: _light_preprocess(ripple_carry_adder(b).to_graph()) for b in TRAIN_BITS}
    gens = {
        "structured_random": make_random_circuit_generator(
            cfg.n_qubits, cfg.n_gates_random, seed=SEED + 5),
        "pure_random": make_pure_random_circuit_generator(
            cfg.pure_random_min_qubits, cfg.pure_random_max_qubits,
            cfg.pure_random_min_gates, cfg.pure_random_max_gates, seed=SEED + 5),
    }

    print("=== combined Step 1a: blind chained search, adders ===", flush=True)
    v3.MAX_ROUNDS = 3
    adder_blind = []
    for b, c in adders.items():
        _, pairs, _ = v3.chained_collect(cfg, c, 150, policy=None, label=f"random {b}b")
        adder_blind.extend(pairs)
    print(f"adder blind pairs: {len(adder_blind)}", flush=True)

    print("=== combined Step 1b: blind search, random families ===", flush=True)
    family_pairs = {}
    for name, gen in gens.items():
        print(f"--- {name} ---", flush=True)
        family_pairs[name] = rp.collect_winners([gen() for _ in range(rp.N_TRAIN)], cfg)
        print(f"{name} pairs: {len(family_pairs[name])}", flush=True)

    print("=== combined Step 2: intermediate distill on pooled blind pairs ===", flush=True)
    pooled = adder_blind + family_pairs["structured_random"] + family_pairs["pure_random"]
    print(f"pooled blind dataset: {len(pooled)} pairs", flush=True)
    policy = v3.distill(list(pooled), cfg, epochs=EPOCHS)

    print("=== combined Step 3: guided chained search on adders + fresh distill ===", flush=True)
    v3.MAX_ROUNDS = 4
    adder_guided = []
    for b, c in adders.items():
        _, pairs, _ = v3.chained_collect(cfg, c, 250, policy=policy, label=f"guided {b}b")
        adder_guided.extend(pairs)
    final_data = adder_guided + family_pairs["structured_random"] + family_pairs["pure_random"]
    print(f"final dataset: {len(adder_guided)} guided adder + "
          f"{len(final_data) - len(adder_guided)} random-family pairs", flush=True)
    policy = v3.distill(list(final_data), cfg, epochs=EPOCHS)

    out = REPO_ROOT / "results" / "checkpoints" / "agent_combined_v1.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved {out}", flush=True)

    print("=== combined Step 4: evaluation (best-of-10) ===", flush=True)
    rows = []
    untrained = ActorCriticGNN(cfg).to(DEVICE)
    untrained.eval()

    for b in TRAIN_BITS + HELDOUT_BITS:
        circ = _light_preprocess(ripple_carry_adder(b).to_graph())
        start, best = best_of_10(policy, cfg, circ)
        tag = "heldout" if b in HELDOUT_BITS else "train"
        rows.append((f"adder{b}", tag, start, best, ""))
        print(f"[adder {b}-bit {tag}] start={start} combined={best}", flush=True)

    for name, gen in gens.items():
        for i in range(5):
            circ = gen()
            start, best = best_of_10(policy, cfg, circ)
            _, ubest = best_of_10(untrained, cfg, circ)
            rows.append((f"{name}_{i}", name, start, best, ubest))
            print(f"[{name} {i}] start={start} combined={best} untrained={ubest}", flush=True)

    for name, c in sorted(load_paper().items()):
        if not any(k in name.lower() for k in ARITH):
            continue
        circ = _light_preprocess(c.to_basic_gates().to_graph())
        start, best = best_of_10(policy, cfg, circ)
        rows.append((name, "real_world", start, best, ""))
        print(f"[real {name}] start={start} combined={best}", flush=True)

    csv_out = REPO_ROOT / "results" / "logs" / "agent_combined_eval.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "family", "start_2q", "combined_best10_2q", "untrained_best10_2q"])
        w.writerows(rows)
    print(f"Saved {csv_out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
