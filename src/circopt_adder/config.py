"""Experiment configuration.

Migrated from Section 2 of the dissertation notebook (circuit-generation parameters,
environment settings, GNN architecture sizes, PPO hyperparameters, and training-run
settings).
"""

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Seed python/numpy/torch RNGs. Note this does not reach pyzx's circuit
    generators (zx.generate.cliffordT takes no seed argument), so random-circuit
    generation is only reproducible up to pyzx's own global RNG state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    # --- circuit generation ---
    n_qubits: int = 5                 # for random-circuit training regime (Agent R)
    n_gates_random: int = 70          # Riu et al. Sec. 4/Table 1: 70 gates + CNOT prob. 1/3
                                       # is their circuit-generation setup specifically for the
                                       # twoqubits reward objective (this project's reward_mode
                                       # default) -- distinct from their total_gates-objective
                                       # setup (5 qubits/60 gates/uniform gate probabilities)
    pure_random_min_qubits: int = 2   # for pure-random training regime (Agent P)
    pure_random_max_qubits: int = 15
    pure_random_min_gates: int = 10
    pure_random_max_gates: int = 200
    adder_min_bits: int = 2           # smallest ripple-carry adder used in training
    adder_max_bits: int = 5           # largest ripple-carry adder used in training
    adder_heldout_interp_bits: Tuple[int, ...] = (4,)       # withheld from training, within range
    adder_heldout_extrap_bits: Tuple[int, ...] = (7, 8, 9)  # never seen at any bit-width during training

    # --- environment ---
    max_episode_steps: int = 35    # matches Riu et al.'s max_episode_len
    reward_mode: str = "twoqubits"  # "twoqubits" (paper default), "total_gates", or "t_count"
    max_compression: int = 10  # Riu et al. Sec 3.3.2 / zx_env.py: fixed reward normalizer

    # --- GNN architecture (Table/Fig 4 of Riu et al.) ---
    gnn_layers: int = 5
    gnn_channels: int = 32
    actor_node_feat_dim: int = 18      # 8 phase one-hot + 3 boundary/gadget flags + 7 action-type flags (LC, PIV, PIVB, PIVG, STOP, GF, ID)
    critic_node_feat_dim: int = 11     # no action-type flags needed
    edge_feat_dim: int = 9             # 2 wire-type (simple/hadamard) + 7 action-node edge types

    # --- PPO hyperparameters (Table 1, Riu et al.) ---
    n_steps: int = 512
    n_envs: int = 8
    learning_rate: float = 2e-4
    n_epochs: int = 8
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_eps: float = 0.1
    max_grad_norm: float = 0.5

    # --- training run ---
    total_timesteps: int = 2_000_000   # scale down for a quick smoke test; paper uses ~16h wall-clock
    log_interval: int = 10
    checkpoint_interval: int = 100     # save the checkpoint every N updates (not just at
                                        # the end) -- a full run is ~3900 updates and can
                                        # take many hours unattended; the final update
                                        # always saves too regardless of this interval
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"

    seed: int = SEED
