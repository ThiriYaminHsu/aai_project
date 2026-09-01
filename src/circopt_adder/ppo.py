"""PPO training loop: rollout collection, GAE, clipped policy/value update.

Migrated from Section 7 of the dissertation notebook. Single-environment rollout
collection for clarity. Loss terms follow Eq. 10-13 of Riu et al.: clipped policy
loss, clipped value loss, entropy bonus.

Rollout *collection* (collect_rollout) remains single-environment and per-step --
running multiple ZXOptEnv copies in parallel would need real multiprocessing, since
pyzx's own graph operations have no batched/vectorised API of their own (see
docs/project_log.md for the scoping decision on why that wasn't attempted). The PPO
*update* (ppo_update), however, is batched: each episode's observation graph has a
different number of nodes, so evaluate_actions_batched combines a minibatch's graphs
into one torch_geometric.data.Batch and does a single GNN forward pass, using
torch_geometric.utils.softmax/scatter to keep each graph's action distribution
correctly segmented despite being processed together. evaluate_actions_loop (the
original one-graph-at-a-time version) is kept alongside it purely as a reference
implementation for tests/test_ppo_batching.py to check the batched version against.

Note: the original repo does not publish a training script -- only the environment,
the agent network class, and an evaluation script (agent_test.py) that loads a
pretrained checkpoint. This training loop is an independent implementation.
"""

import faulthandler
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import softmax as pyg_softmax

from .config import DEVICE, Config
from .env import ZXOptEnv
from .model import ActorCriticGNN


@dataclass
class Transition:
    data: Data
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


def collect_rollout(env: ZXOptEnv, policy: ActorCriticGNN, n_steps: int) -> List[Transition]:
    transitions = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            obs_dev = obs.to(DEVICE)
            action_idx, log_prob, value = policy.act(obs_dev)
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
        transitions.append(Transition(obs, action_idx, log_prob.item(), value.item(), reward, done))
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return transitions


def compute_gae(transitions: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.array([t.reward for t in transitions])
    values = np.array([t.value for t in transitions] + [0.0])
    dones = np.array([t.done for t in transitions])

    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(transitions))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns


def evaluate_actions_loop(policy: ActorCriticGNN, data_list: List[Data],
                           action_idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference (unbatched) implementation: one policy.evaluate_actions() call per
    graph, exactly as the original per-transition PPO loop did. Kept as a named
    function -- not just inlined in ppo_update -- so evaluate_actions_batched can be
    tested against it directly for numerical equivalence (tests/test_ppo_batching.py),
    rather than trusting the batched rewrite on faith. Not used in the hot path."""
    log_probs, entropies, values = [], [], []
    for data, action_idx in zip(data_list, action_idxs):
        data_dev = data.to(DEVICE)
        action_t = torch.tensor(action_idx, device=DEVICE)
        log_prob, entropy, value = policy.evaluate_actions(data_dev, action_t)
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def evaluate_actions_batched(policy: ActorCriticGNN, data_list: List[Data],
                              action_idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched equivalent of evaluate_actions_loop: one GNN forward pass over the
    whole minibatch (via torch_geometric.data.Batch) instead of one pass per graph.

    The tricky part is that batching concatenates every graph's action-node logits
    into one flat tensor, so a plain softmax/argmax would mix actions from different
    graphs together. torch_geometric.utils.softmax(src, index) and .utils.scatter
    do a proper *segmented* softmax/sum, grouped by which graph each entry belongs
    to (action_batch below) -- this is what keeps each graph's action distribution
    independent of every other graph's, despite the single batched forward pass.
    """
    batch = Batch.from_data_list(data_list).to(DEVICE)
    x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

    h = F.relu(policy.input_proj(x))
    for conv in policy.convs:
        h = F.relu(conv(h, edge_index, edge_attr=edge_attr))

    all_logits = policy.policy_head(h).squeeze(-1)
    action_mask = batch.action_node_mask
    action_logits = all_logits[action_mask]        # every graph's action logits, concatenated
    action_batch = batch.batch[action_mask]         # which graph (0..N-1) each entry above belongs to

    n_graphs = len(data_list)
    probs = pyg_softmax(action_logits, action_batch, num_nodes=n_graphs)
    log_probs_all = torch.log(probs.clamp_min(1e-12))
    entropy_terms = -probs * log_probs_all
    entropy_per_graph = pyg_scatter(entropy_terms, action_batch, dim=0, dim_size=n_graphs, reduce="sum")

    # Each transition's action_idx is a *local* index into its own graph's action
    # nodes; convert to a *global* index into the flat action_logits/log_probs_all
    # tensor via each graph's exclusive-prefix-sum offset (how many action nodes
    # every earlier graph in this batch contributed).
    counts = pyg_scatter(torch.ones_like(action_batch), action_batch, dim=0, dim_size=n_graphs, reduce="sum")
    offsets = torch.cumsum(counts, dim=0) - counts
    action_idxs_t = torch.tensor(action_idxs, device=DEVICE, dtype=offsets.dtype)
    global_idx = offsets + action_idxs_t
    log_prob_taken = log_probs_all[global_idx]

    pooled = global_mean_pool(policy.value_proj(h), batch.batch)
    values = policy.value_head(F.relu(pooled)).squeeze(-1)

    return log_prob_taken, entropy_per_graph, values


def ppo_update(policy: ActorCriticGNN, optimizer: torch.optim.Optimizer,
                transitions: List[Transition], advantages: np.ndarray, returns: np.ndarray,
                cfg: Config) -> dict:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idxs = np.arange(len(transitions))

    stats = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(cfg.n_epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), cfg.minibatch_size):
            batch_idx = idxs[start:start + cfg.minibatch_size]
            batch_transitions = [transitions[i] for i in batch_idx]
            data_list = [tr.data for tr in batch_transitions]
            action_idxs = [tr.action_idx for tr in batch_transitions]

            log_probs, entropies, values = evaluate_actions_batched(policy, data_list, action_idxs)

            old_log_probs = torch.tensor([tr.log_prob for tr in batch_transitions], device=DEVICE)
            advs = torch.tensor(advantages[batch_idx], device=DEVICE, dtype=torch.float32)
            rets = torch.tensor(returns[batch_idx], device=DEVICE, dtype=torch.float32)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advs
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, rets)
            entropy = entropies.mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


_LOG_COLUMNS = ["update", "timesteps", "mean_reward", "policy_loss", "value_loss", "entropy", "wall_time_s"]


def train(env: ZXOptEnv, policy: ActorCriticGNN, cfg: Config, run_name: str) -> pd.DataFrame:
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    n_updates = cfg.total_timesteps // cfg.n_steps
    log_rows = []

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_dir) / f"{run_name}_train_log.csv"
    checkpoint_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
    heartbeat_path = Path(cfg.log_dir) / f"{run_name}_heartbeat.txt"
    crash_path = Path(cfg.log_dir) / f"{run_name}_crash.txt"

    # Diagnostic instrumentation added after two silent, traceback-less process
    # deaths (pilot run + Agent A, twice) with no OOM evidence and no macOS crash
    # report found. A plain except-and-log can only catch genuine Python
    # exceptions; faulthandler additionally dumps a traceback on fatal signals
    # (SIGSEGV/SIGABRT/SIGBUS/SIGFPE, e.g. a C-level crash inside numpy/torch/
    # pyzx) before the process dies, and on SIGTERM (an external kill request,
    # as opposed to a crash). Neither can do anything about an uncatchable
    # SIGKILL -- for that, the heartbeat file (last write = last update reached)
    # is the only way to learn anything post-mortem.
    faulthandler.enable()
    try:
        faulthandler.register(signal.SIGTERM, all_threads=True)
    except (ValueError, AttributeError):
        pass  # not available on this platform; faulthandler.enable() still covers fatal signals

    # Header written up front, then each row appended as it's produced below, and the
    # checkpoint saved every cfg.checkpoint_interval updates (plus always on the final
    # one) -- a full run is ~3900 updates and can take many hours unattended, so both
    # need to be durable against a crash/kill/sleep partway through, not only written
    # once at the very end (which is what this loop did before -- losing everything on
    # an interruption at update 3900 of 3906, say).
    pd.DataFrame(columns=_LOG_COLUMNS).to_csv(log_path, index=False)

    try:
        for update in range(n_updates):
            t0 = time.time()
            heartbeat_path.write_text(f"update {update}/{n_updates} starting rollout at {t0}\n")

            transitions = collect_rollout(env, policy, cfg.n_steps)
            advantages, returns = compute_gae(transitions, cfg.gamma, cfg.gae_lambda)

            # --- instrumentation ---
            failure_rate = env.n_extraction_failures / max(env.n_steps_total, 1)
            print(f"    [diag] adv mean={advantages.mean():.4f} std={advantages.std():.4f} "
                  f"| extraction failure rate (cumulative)={failure_rate:.1%}")

            stats = ppo_update(policy, optimizer, transitions, advantages, returns, cfg)

            row = {
                "update": update,
                "timesteps": (update + 1) * cfg.n_steps,
                "mean_reward": float(np.mean([t.reward for t in transitions])),
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "wall_time_s": time.time() - t0,
            }
            log_rows.append(row)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=False, index=False)
            heartbeat_path.write_text(f"update {update}/{n_updates} completed at {time.time()}\n")

            if update % cfg.log_interval == 0:
                print(f"[{run_name}] update {update}/{n_updates} "
                      f"reward={row['mean_reward']:.3f} "
                      f"entropy={stats['entropy']:.3f} "
                      f"({row['wall_time_s']:.1f}s)")

            if (update + 1) % cfg.checkpoint_interval == 0 or update == n_updates - 1:
                torch.save(policy.state_dict(), checkpoint_path)
    except Exception:
        crash_path.write_text(traceback.format_exc())
        print(f"[{run_name}] CRASHED -- traceback written to {crash_path}", file=sys.stderr, flush=True)
        raise

    return pd.DataFrame(log_rows)
