"""Expert-iteration-lite for Agent A (project_log entries 46-50 context).

Rationale. Entry 50 showed the self-play ratchet finds ~1-gate improvements
over basic_opt on the 2-/3-bit adders but PPO never consolidates them (the
-0.10 reward-plateau signature), and the random-init control showed an
untrained policy's best-of-10 search matches a trained agent's -- search
discovers, RL fails to retain. This experiment separates the two halves
explicitly:

  Phase 1 (search):  pure uniform-random rollouts in the current
                     six-action ZXOptEnv on Agent A's three training
                     circuits ({2,3,5}-bit adders, light-preprocessed).
                     No network involved.
  Phase 2 (distill): behaviour-clone a fresh policy on the winning
                     trajectory prefixes (steps up to each episode's best
                     point, plus a STOP label at the best point).
  Phase 3 (iterate): policy-guided search with the distilled policy; if it
                     finds anything better, merge and distill again.
  Phase 4 (evaluate): greedy determinism per circuit, reproduction rate
                     over 40 stochastic samples, and generalisation to the
                     held-out 4-bit (interpolation) and 7-bit
                     (extrapolation) adders that search never touched.

Success criterion: the distilled policy reproduces the search-found best
greedily (or at high frequency), i.e. consolidation succeeds where PPO's
~1/40 reproduction failed. Failure is equally informative: it would locate
the problem in the network/representation rather than the RL signal.

Usage:
    python scripts/expert_iteration_agent_a.py
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

TRAIN_BITS = [2, 3, 5]
HELDOUT_BITS = [4, 7]
N_RANDOM_ROLLOUTS = 1000     # phase-1 random-search episodes per circuit
N_GUIDED_ROLLOUTS = 200      # phase-3 policy-guided episodes per circuit
MAX_TRAJ_PER_CIRCUIT = 20    # cap stored winners per circuit (post-dedupe)
BC_EPOCHS = 200
BC_LR = 1e-3


def make_circuit(n_bits):
    return _light_preprocess(ripple_carry_adder(n_bits).to_graph())


def rollout(env_circuit, cfg, policy=None, greedy=False):
    """One episode. Returns (episode_best_metric, transitions, t_best,
    obs_after_best) where transitions = [(obs, action_idx), ...]."""
    env = ZXOptEnv(lambda c=env_circuit: c, cfg)
    obs, _ = env.reset()
    transitions = []
    prev_best = env.best_metric
    t_best = -1                      # -1: never improved on the start
    obs_after_best = None
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
            prev_best = env.best_metric
            t_best = t
            obs_after_best = obs2    # may be None if the episode just ended
        if term or trunc:
            break
        obs = obs2
    return env.best_metric, transitions, t_best, obs_after_best


def search_phase(circuits, cfg, n_rollouts, policy=None, label="random"):
    """Best-of-n search per circuit; returns {bits: (best_metric, [winner
    trajectories])} where each winner is a list of (obs, action) pairs (the
    prefix to the best point, plus a STOP pair when available)."""
    results = {}
    for bits, circ in circuits.items():
        start = two_qubit_gate_count(circ)
        best = None
        winners = {}
        for i in range(n_rollouts):
            m, transitions, t_best, obs_after = rollout(circ, cfg, policy=policy)
            if t_best < 0:
                continue  # never improved on the starting circuit
            if best is None or m < best:
                best, winners = m, {}
            if m == best:
                prefix = transitions[: t_best + 1]
                key = tuple(a for _, a in prefix)
                if key not in winners and len(winners) < MAX_TRAJ_PER_CIRCUIT:
                    pairs = list(prefix)
                    if obs_after is not None:
                        pairs.append((obs_after, 0))  # STOP is always action index 0
                    winners[key] = pairs
        results[bits] = (best if best is not None else start, list(winners.values()))
        print(f"  [{label} search] {bits}-bit: start={start}, best found={results[bits][0]}, "
              f"winning trajectories kept={len(results[bits][1])}", flush=True)
    return results


def distill(dataset, cfg, epochs=BC_EPOCHS, lr=BC_LR):
    policy = ActorCriticGNN(cfg).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(dataset)
        total, correct = 0.0, 0
        for obs, a in dataset:
            logits, _ = policy.forward(obs.to(DEVICE))
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([a], device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            correct += int(int(torch.argmax(logits).item()) == a)
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"  [distill] epoch {ep}: mean loss={total/len(dataset):.4f}, "
                  f"argmax accuracy={correct}/{len(dataset)}", flush=True)
    policy.eval()
    return policy


def evaluate(policy, circuits, cfg, targets, n_samples=40):
    for bits, circ in circuits.items():
        start = two_qubit_gate_count(circ)
        g_best, _, _, _ = rollout(circ, cfg, policy=policy, greedy=True)
        mins, hits = [], 0
        for _ in range(n_samples):
            m, _, _, _ = rollout(circ, cfg, policy=policy)
            mins.append(m)
            if m < start:
                hits += 1
        tgt = targets.get(bits)
        print(f"  [eval] {bits}-bit: start={start}, search-best={tgt}, greedy={g_best}, "
              f"min over {n_samples} sampled={min(mins)}, samples beating start={hits}/{n_samples}", flush=True)


def main():
    set_seed(SEED)
    cfg = Config()
    train_circuits = {b: make_circuit(b) for b in TRAIN_BITS}
    heldout_circuits = {b: make_circuit(b) for b in HELDOUT_BITS}

    print("=== Phase 1: uniform-random search ===", flush=True)
    r1 = search_phase(train_circuits, cfg, N_RANDOM_ROLLOUTS, policy=None, label="random")

    dataset = [pair for bits in TRAIN_BITS for traj in r1[bits][1] for pair in traj]
    print(f"=== Phase 2: distillation ({len(dataset)} state-action pairs) ===", flush=True)
    policy = distill(dataset, cfg)

    print("=== Phase 3: policy-guided search ===", flush=True)
    r2 = search_phase(train_circuits, cfg, N_GUIDED_ROLLOUTS, policy=policy, label="guided")
    improved = any(r2[b][0] < r1[b][0] for b in TRAIN_BITS)
    if improved:
        print("  guided search improved on random search; distilling round 2", flush=True)
        merged = list(dataset)
        for b in TRAIN_BITS:
            if r2[b][0] < r1[b][0]:
                merged += [pair for traj in r2[b][1] for pair in traj]
        policy = distill(merged, cfg)
        for b in TRAIN_BITS:
            if r2[b][0] < r1[b][0]:
                r1[b] = r2[b]
    else:
        print("  guided search found nothing better; keeping round-1 policy", flush=True)

    targets = {b: r1[b][0] for b in TRAIN_BITS}
    print("=== Phase 4a: evaluation on training circuits ===", flush=True)
    evaluate(policy, train_circuits, cfg, targets)
    print("=== Phase 4b: evaluation on held-out adders (search never touched these) ===", flush=True)
    evaluate(policy, heldout_circuits, cfg, {})

    out = REPO_ROOT / "results" / "checkpoints" / "agent_A_expert_iteration.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved distilled policy to {out}", flush=True)


if __name__ == "__main__":
    main()
