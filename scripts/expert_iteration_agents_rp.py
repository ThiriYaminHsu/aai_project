"""Expert-iteration for the Agent R / Agent P families (follow-up to
entry 51's Agent A experiment; user: "run it").

Question. Entry 51 showed search-distillation gives Agent A a transferable
search bias (held-out 42 vs basic_opt's 44, against a 0/40 random
control). Does the same recipe improve on the random-circuit families --
where the honest bar is NOT basic_opt but the UNTRAINED-network best-of-10
control (~10% mean 2Q reduction), which every RL-trained checkpoint so far
fails to beat?

Design (per family, structured_random = Agent R's recipe, pure_random =
Agent P's recipe; current six-action env; circuits arrive
light-preprocessed from the generators, matching evaluation):
  Phase 1: search  -- N_TRAIN fresh circuits x N_ROLLOUTS uniform-random
                      episodes each; keep winning trajectory prefixes
                      (cut at best point, STOP label appended).
  Phase 2: distill -- behaviour-clone a fresh policy per family.
  Phase 3: evaluate -- N_EVAL fresh circuits (never searched); best-of-10
                      distilled vs best-of-10 untrained (random-init,
                      fixed seed) vs a single basic_opt pass.
No guided-search round here (unlike the Agent A script) -- the target
measurement is distilled-vs-untrained as search priors, kept minimal for
time. Success criterion, per family: distilled best-of-10 mean 2Q
reduction > untrained best-of-10 on the same fresh circuits.

Usage:
    python scripts/expert_iteration_agents_rp.py
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from circopt_adder.baselines import baseline_basic_optimization
from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.generators import (
    make_pure_random_circuit_generator,
    make_random_circuit_generator,
)
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import two_qubit_gate_count
from scripts.expert_iteration_agent_a import rollout, distill

N_TRAIN = 150       # circuits searched per family
N_ROLLOUTS = 15     # random rollouts per training circuit
N_EVAL = 30         # fresh circuits per family for the final comparison
MAX_PAIRS = 1200    # cap on distillation dataset size per family
BC_EPOCHS = 60


def collect_winners(circuits, cfg):
    dataset = []
    n_circ_improved = 0
    for idx, circ in enumerate(circuits):
        start = two_qubit_gate_count(circ)
        best, best_traj = None, None
        for _ in range(N_ROLLOUTS):
            m, transitions, t_best, obs_after = rollout(circ, cfg, policy=None)
            if t_best < 0 or m >= start:
                continue
            if best is None or m < best:
                pairs = list(transitions[: t_best + 1])
                if obs_after is not None:
                    pairs.append((obs_after, 0))  # STOP
                best, best_traj = m, pairs
        if best_traj is not None:
            n_circ_improved += 1
            dataset.extend(best_traj)
        if (idx + 1) % 25 == 0:
            print(f"  [search] {idx + 1}/{len(circuits)} circuits, "
                  f"{n_circ_improved} improved, {len(dataset)} pairs so far", flush=True)
    print(f"  [search] done: {n_circ_improved}/{len(circuits)} circuits improved, "
          f"{len(dataset)} state-action pairs", flush=True)
    if len(dataset) > MAX_PAIRS:
        dataset = random.sample(dataset, MAX_PAIRS)
        print(f"  [search] capped dataset to {MAX_PAIRS} pairs", flush=True)
    return dataset


def best_of_10(policy, circuit, cfg):
    best = None
    for i in range(10):
        m, _, _, _ = rollout(circuit, cfg, policy=policy, greedy=(i == 0))
        best = m if best is None else min(best, m)
    return best


def evaluate_family(name, eval_circuits, distilled, untrained, cfg):
    tot_i = tot_bo = tot_d = tot_u = 0
    d_wins = u_wins = 0
    for circ in eval_circuits:
        i2q = two_qubit_gate_count(circ)
        try:
            bo = two_qubit_gate_count(baseline_basic_optimization(circ))
        except Exception:
            bo = i2q
        d = best_of_10(distilled, circ, cfg)
        u = best_of_10(untrained, circ, cfg)
        tot_i += i2q; tot_bo += bo; tot_d += d; tot_u += u
        d_wins += int(d < u); u_wins += int(u < d)
    def pct(x): return 100.0 * (tot_i - x) / tot_i
    print(f"  [eval {name}] mean 2Q reduction over {len(eval_circuits)} fresh circuits: "
          f"basic_opt={pct(tot_bo):.1f}%  distilled={pct(tot_d):.1f}%  untrained={pct(tot_u):.1f}%", flush=True)
    print(f"  [eval {name}] head-to-head circuits: distilled better={d_wins}, "
          f"untrained better={u_wins}, ties={len(eval_circuits) - d_wins - u_wins}", flush=True)


def main():
    set_seed(SEED)
    cfg = Config()

    untrained = ActorCriticGNN(cfg).to(DEVICE)
    untrained.eval()

    families = {
        "structured_random": make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random, seed=SEED + 3),
        "pure_random": make_pure_random_circuit_generator(
            cfg.pure_random_min_qubits, cfg.pure_random_max_qubits,
            cfg.pure_random_min_gates, cfg.pure_random_max_gates, seed=SEED + 3),
    }

    for name, gen in families.items():
        print(f"=== family: {name} ===", flush=True)
        train_circuits = [gen() for _ in range(N_TRAIN)]
        eval_circuits = [gen() for _ in range(N_EVAL)]   # drawn after; never searched

        print(f"=== {name}: Phase 1 search ===", flush=True)
        dataset = collect_winners(train_circuits, cfg)

        print(f"=== {name}: Phase 2 distillation ({len(dataset)} pairs) ===", flush=True)
        distilled = distill(dataset, cfg, epochs=BC_EPOCHS)

        print(f"=== {name}: Phase 3 evaluation ===", flush=True)
        evaluate_family(name, eval_circuits, distilled, untrained, cfg)

        out = REPO_ROOT / "results" / "checkpoints" / f"agent_expert_iteration_{name}.pt"
        torch.save(distilled.state_dict(), out)
        print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
