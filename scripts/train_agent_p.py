"""Train Agent P (pure-random training regime).

Not in Riu et al. -- the third arm of this dissertation's random-vs-structured
ablation, added alongside Agent R (fixed-size, fixed-recipe random circuits) and
Agent A (adder-specific). Identical config to Agent R except for the generator: both
circuit size (n_qubits, n_gates) and gate-type probabilities are left unconstrained
per episode, rather than fixed at Riu et al.'s tuned values. See
docs/random_vs_structured_composition_analysis.md for the composition analysis
motivating this as a distinct regime from Agent R's.

Requires the package to be installed (`pip install -e .` from the repo root).

Usage:
    python scripts/train_agent_p.py [--smoke-test]
"""

import argparse

from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import make_pure_random_circuit_generator
from circopt_adder.model import ActorCriticGNN
from circopt_adder.ppo import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Drastically reduce total_timesteps before committing to a full run.")
    args = parser.parse_args()

    set_seed(SEED)
    print("Using device:", DEVICE)

    cfg = Config()
    if args.smoke_test:
        cfg.total_timesteps = 10_000
        cfg.log_interval = 1

    generator = make_pure_random_circuit_generator(
        min_qubits=cfg.pure_random_min_qubits, max_qubits=cfg.pure_random_max_qubits,
        min_gates=cfg.pure_random_min_gates, max_gates=cfg.pure_random_max_gates,
        seed=SEED,
    )
    env = ZXOptEnv(generator, cfg)
    policy = ActorCriticGNN(cfg).to(DEVICE)

    train(env, policy, cfg, run_name="agent_P_pure_random")


if __name__ == "__main__":
    main()
