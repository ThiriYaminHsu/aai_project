"""Agent A v3: the winning recipe end to end (entries 51-61 synthesis).

Replication is explicitly not the goal here (user's call); this run
combines every component the project's controlled experiments validated:

  targets   -- chained search (iterated restarts, entry 59's dominant
               lever) collects deep winning trajectories, not single-round
               ones, on widths {2,3,5,6,7,8,9}; widths 4 and 10 stay
               fully held out.
  labels    -- deduplicated by construction: at most one winning
               trajectory per (width, round) (entry 61: label conflict was
               half the consolidation blocker).
  network   -- 128-channel GATv2 (entry 61: capacity was the other half).
  iteration -- one policy-guided chained search round after the first
               distillation, merged and re-distilled (entry 51: guided
               search strictly improved on blind search).

Evaluation: per width including held-outs, greedy single rollout,
best-of-20 sampled, and a chained-greedy pass, against the preprocessed
start (= basic_opt's level on these circuits).

Usage:
    python scripts/train_agent_a_v3.py
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F

from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import ripple_carry_adder, _light_preprocess
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import two_qubit_gate_count

TRAIN_BITS = [2, 3, 5, 6, 7, 8, 9]
HELDOUT_BITS = [4, 10]
N_PER_ROUND = 150
GUIDED_PER_ROUND = 100
MAX_ROUNDS = 3
CHANNELS = 128
BC_EPOCHS = 300
BC_LR = 1e-3


def make_cfg():
    cfg = Config()
    cfg.gnn_channels = CHANNELS
    return cfg


def episode(cfg, circuit, policy=None, greedy=False):
    """One episode; returns (best_metric, best_circuit, pairs) where pairs
    is the winning prefix (cut at the best point, STOP appended) or []."""
    env = ZXOptEnv(lambda c=circuit: c, cfg)
    obs, _ = env.reset()
    transitions = []
    prev_best = env.best_metric
    t_best, obs_after = -1, None
    for t in range(cfg.max_episode_steps):
        if policy is None:
            a = random.randrange(obs.n_actions)
        else:
            with torch.no_grad():
                logits, _ = policy.forward(obs.to(DEVICE))
                a = int(torch.argmax(logits).item()) if greedy else torch.distributions.Categorical(logits=logits).sample().item()
        transitions.append((obs, a))
        obs2, r, term, trunc, info = env.step(a)
        if env.best_metric < prev_best:
            prev_best, t_best, obs_after = env.best_metric, t, obs2
        if term or trunc:
            break
        obs = obs2
    pairs = []
    if t_best >= 0:
        pairs = list(transitions[: t_best + 1])
        if obs_after is not None:
            pairs.append((obs_after, 0))  # STOP at the best point
    return env.best_metric, env.best_circuit, pairs


def chained_collect(cfg, start_circuit, n_per_round, policy=None, label=""):
    """Chained search: per round keep the single best winning trajectory
    (dedup by construction); restart from its circuit while improving.
    Returns (final_best_metric, all_pairs, rounds_used)."""
    current, current_2q = start_circuit, two_qubit_gate_count(start_circuit)
    all_pairs, rounds_used = [], 0
    for r in range(MAX_ROUNDS):
        best_m, best_c, best_pairs = None, None, []
        for i in range(n_per_round):
            greedy = (policy is not None and i == 0)
            m, c, pairs = episode(cfg, current, policy=policy, greedy=greedy)
            if pairs and (best_m is None or m < best_m):
                best_m, best_c, best_pairs = m, c, pairs
        rounds_used = r + 1
        if best_m is not None and best_m < current_2q:
            all_pairs.extend(best_pairs)
            current, current_2q = best_c, best_m
        else:
            break
    print(f"  [{label}] final={current_2q} rounds={rounds_used} pairs+={len(all_pairs)}", flush=True)
    return current_2q, all_pairs, rounds_used


def distill(dataset, cfg, epochs=BC_EPOCHS):
    policy = ActorCriticGNN(cfg).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=BC_LR)
    for ep in range(epochs):
        random.shuffle(dataset)
        correct = 0
        for obs, a in dataset:
            logits, _ = policy.forward(obs.to(DEVICE))
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([a], device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int(int(torch.argmax(logits).item()) == a)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  [distill] epoch {ep}: acc={correct}/{len(dataset)}", flush=True)
    policy.eval()
    return policy


def evaluate(policy, cfg, bits_list, tag):
    for nb in bits_list:
        circ = _light_preprocess(ripple_carry_adder(nb).to_graph())
        start = two_qubit_gate_count(circ)
        g, _, _ = episode(cfg, circ, policy=policy, greedy=True)
        mins, hits = [], 0
        for _ in range(20):
            m, _, _ = episode(cfg, circ, policy=policy)
            mins.append(m)
            if m < start:
                hits += 1
        cg, _, _ = chained_collect(cfg, circ, 20, policy=policy, label=f"eval-chain {nb}b")
        print(f"[eval {tag}] {nb}-bit: start={start} greedy={g} min20={min(mins)} "
              f"hits={hits}/20 chained-greedy20={cg}", flush=True)


def main():
    set_seed(SEED)
    cfg = make_cfg()
    circuits = {b: _light_preprocess(ripple_carry_adder(b).to_graph()) for b in TRAIN_BITS}

    print("=== Phase 1: chained random search ===", flush=True)
    dataset = []
    for b, c in circuits.items():
        _, pairs, _ = chained_collect(cfg, c, N_PER_ROUND, policy=None, label=f"random {b}b")
        dataset.extend(pairs)
    print(f"phase-1 dataset: {len(dataset)} pairs", flush=True)

    print("=== Phase 2: distillation (128ch, dedup-by-construction) ===", flush=True)
    policy = distill(list(dataset), cfg)

    print("=== Phase 3: guided chained search + re-distill ===", flush=True)
    added = 0
    for b, c in circuits.items():
        _, pairs, _ = chained_collect(cfg, c, GUIDED_PER_ROUND, policy=policy, label=f"guided {b}b")
        dataset.extend(pairs); added += len(pairs)
    print(f"guided round added {added} pairs; re-distilling", flush=True)
    policy = distill(list(dataset), cfg)

    out = REPO_ROOT / "results" / "checkpoints" / "agent_A_v3_128dedup.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved {out}", flush=True)

    print("=== Phase 4a: evaluation, training widths ===", flush=True)
    evaluate(policy, cfg, TRAIN_BITS, "train")
    print("=== Phase 4b: evaluation, held-out widths (4, 10) ===", flush=True)
    evaluate(policy, cfg, HELDOUT_BITS, "heldout")


if __name__ == "__main__":
    main()
