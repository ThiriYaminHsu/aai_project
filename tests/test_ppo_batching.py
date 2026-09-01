"""Correctness check for the batched PPO update (ppo.evaluate_actions_batched)
against the original one-graph-at-a-time reference implementation
(ppo.evaluate_actions_loop), on a real rollout of variable-sized observation
graphs. Exists specifically because a batching rewrite that silently mis-groups
per-graph action logits/log-probs would still "train" and produce plausible-
looking curves while computing the wrong PPO loss -- this is the guard against
that, not just against a crash.
"""

import torch

from circopt_adder.config import DEVICE, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import make_pure_random_circuit_generator
from circopt_adder.model import ActorCriticGNN
from circopt_adder.ppo import collect_rollout, evaluate_actions_batched, evaluate_actions_loop


def _collect_mixed_size_rollout(n_steps=40, seed=123):
    """pure_random on purpose: varying circuit size across episodes means the
    resulting graphs genuinely differ in node/action count from step to step,
    which is exactly the case a batching bug would show up on -- fixed-size
    graphs wouldn't exercise the per-graph offset logic at all."""
    set_seed(seed)
    cfg = Config()
    cfg.pure_random_min_qubits, cfg.pure_random_max_qubits = 2, 6
    cfg.pure_random_min_gates, cfg.pure_random_max_gates = 10, 40
    cfg.max_episode_steps = 8  # short episodes -> more distinct circuits per rollout
    gen = make_pure_random_circuit_generator(
        cfg.pure_random_min_qubits, cfg.pure_random_max_qubits,
        cfg.pure_random_min_gates, cfg.pure_random_max_gates, seed=seed,
    )
    env = ZXOptEnv(gen, cfg)
    policy = ActorCriticGNN(cfg).to(DEVICE)
    policy.eval()
    transitions = collect_rollout(env, policy, n_steps)
    return policy, transitions


def test_batched_matches_loop_forward():
    policy, transitions = _collect_mixed_size_rollout()
    data_list = [t.data for t in transitions]
    action_idxs = [t.action_idx for t in transitions]

    # sanity: this rollout actually contains graphs of different sizes, otherwise
    # the test wouldn't be exercising the per-graph offset/grouping logic at all
    n_action_nodes = [int(d.action_node_mask.sum()) for d in data_list]
    assert len(set(n_action_nodes)) > 1, "rollout has no size variation -- test is not meaningful"

    with torch.no_grad():
        log_probs_loop, entropies_loop, values_loop = evaluate_actions_loop(policy, data_list, action_idxs)
        log_probs_batched, entropies_batched, values_batched = evaluate_actions_batched(policy, data_list, action_idxs)

    assert torch.allclose(log_probs_loop, log_probs_batched, atol=1e-4, rtol=1e-4)
    assert torch.allclose(entropies_loop, entropies_batched, atol=1e-4, rtol=1e-4)
    assert torch.allclose(values_loop, values_batched, atol=1e-4, rtol=1e-4)


def test_batched_matches_loop_gradients():
    """Forward-value agreement alone wouldn't catch a batching bug that only
    breaks how gradients are routed back through the shared GNN trunk (e.g. if
    autograd ends up mixing contributions across graphs) -- so also check that
    backpropagating a simple scalar loss produces matching parameter gradients."""
    policy, transitions = _collect_mixed_size_rollout(seed=7)
    data_list = [t.data for t in transitions]
    action_idxs = [t.action_idx for t in transitions]

    policy.zero_grad()
    log_probs_loop, entropies_loop, values_loop = evaluate_actions_loop(policy, data_list, action_idxs)
    (log_probs_loop.sum() + entropies_loop.sum() + values_loop.sum()).backward()
    grads_loop = [p.grad.clone() for p in policy.parameters()]

    policy.zero_grad()
    log_probs_batched, entropies_batched, values_batched = evaluate_actions_batched(policy, data_list, action_idxs)
    (log_probs_batched.sum() + entropies_batched.sum() + values_batched.sum()).backward()
    grads_batched = [p.grad.clone() for p in policy.parameters()]

    for g_loop, g_batched in zip(grads_loop, grads_batched):
        assert torch.allclose(g_loop, g_batched, atol=1e-3, rtol=1e-3)
