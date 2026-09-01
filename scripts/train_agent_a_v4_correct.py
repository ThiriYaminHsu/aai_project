"""Agent A v4: the winning v3.1 recipe, retrained on the CORRECTED
Cuccaro adder generator (carry wiring fixed and verified 2026-08-31;
see tests/test_adder_correctness.py and project log entries 62/64).

Pipeline (mirrors v3 + the v3.1 extension in one run):
  Phase 1: chained random search, 150 episodes/round, up to 3 rounds,
           widths {2,3,5,6,7,8,9}; 4 and 10 fully held out.
  Phase 2: 128-channel dedup distillation on the phase-1 pairs.
  Phase 3: guided chained search at the v3.1 budget (250/round, up to
           4 rounds); a FRESH policy is then distilled on the guided
           pairs alone, the configuration that won in v3.1.
  Phase 4: evaluation per width (greedy / best-of-20 / chained-greedy).

Saves results/checkpoints/agent_A_v4_correct.pt.

Usage:
    python3 scripts/train_agent_a_v4_correct.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch

import train_agent_a_v3 as v3
from circopt_adder.config import SEED, set_seed
from circopt_adder.generators import _light_preprocess, ripple_carry_adder

TRAIN_BITS = [2, 3, 5, 6, 7, 8, 9]
HELDOUT_BITS = [4, 10]


def main():
    set_seed(SEED)
    cfg = v3.make_cfg()
    circuits = {b: _light_preprocess(ripple_carry_adder(b).to_graph()) for b in TRAIN_BITS}

    print("=== v4 Phase 1: chained random search (corrected adders) ===", flush=True)
    v3.MAX_ROUNDS = 3
    dataset = []
    for b, c in circuits.items():
        _, pairs, _ = v3.chained_collect(cfg, c, 150, policy=None, label=f"random {b}b")
        dataset.extend(pairs)
    print(f"phase-1 dataset: {len(dataset)} pairs", flush=True)

    print("=== v4 Phase 2: distillation (128ch, dedup-by-construction) ===", flush=True)
    policy = v3.distill(list(dataset), cfg)

    print("=== v4 Phase 3: guided chained search (250/round, 4 rounds) + fresh distill ===", flush=True)
    v3.MAX_ROUNDS = 4
    guided_pairs = []
    for b, c in circuits.items():
        _, pairs, _ = v3.chained_collect(cfg, c, 250, policy=policy, label=f"guided {b}b")
        guided_pairs.extend(pairs)
    print(f"guided dataset: {len(guided_pairs)} pairs; distilling fresh policy", flush=True)
    policy = v3.distill(list(guided_pairs), cfg)

    out = REPO_ROOT / "results" / "checkpoints" / "agent_A_v4_correct.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved {out}", flush=True)

    print("=== v4 Phase 4a: evaluation, training widths ===", flush=True)
    v3.evaluate(policy, cfg, TRAIN_BITS, "train")
    print("=== v4 Phase 4b: evaluation, held-out widths (4, 10) ===", flush=True)
    v3.evaluate(policy, cfg, HELDOUT_BITS, "heldout")


if __name__ == "__main__":
    main()
