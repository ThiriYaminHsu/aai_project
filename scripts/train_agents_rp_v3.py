"""R/P v3: the entry-61 winning recipe (128 channels; deduplicated labels,
which R/P datasets satisfy by construction: one winning trajectory per
circuit) applied to the random families at fuller budgets, judged against
the untrained-network control that beat every previous attempt (entries
52/57). Expectation stated up front: the diversity-loss mechanism is
untouched by this recipe, so the honest bar is whether 128ch+dedup changes
the distilled-vs-untrained ordering at all.

Usage: python scripts/train_agents_rp_v3.py
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.generators import make_pure_random_circuit_generator, make_random_circuit_generator
from circopt_adder.model import ActorCriticGNN
from scripts.expert_iteration_agents_rp import collect_winners, evaluate_family
import scripts.expert_iteration_agents_rp as rp
from scripts.expert_iteration_agent_a import distill

rp.N_TRAIN, rp.N_ROLLOUTS, rp.N_EVAL = 150, 15, 30
CHANNELS, EPOCHS = 128, 150

def main():
    set_seed(SEED)
    cfg = Config(); cfg.gnn_channels = CHANNELS
    untrained = ActorCriticGNN(cfg).to(DEVICE); untrained.eval()
    fams = {
        "structured_random": make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random, seed=SEED + 5),
        "pure_random": make_pure_random_circuit_generator(
            cfg.pure_random_min_qubits, cfg.pure_random_max_qubits,
            cfg.pure_random_min_gates, cfg.pure_random_max_gates, seed=SEED + 5),
    }
    for name, gen in fams.items():
        print(f"=== {name} v3 (128ch) ===", flush=True)
        train_c = [gen() for _ in range(rp.N_TRAIN)]
        eval_c = [gen() for _ in range(rp.N_EVAL)]
        dataset = collect_winners(train_c, cfg)
        policy = distill(dataset, cfg, epochs=EPOCHS)
        evaluate_family(name, eval_c, policy, untrained, cfg)
        out = REPO_ROOT / "results" / "checkpoints" / f"agent_v3_128_{name}.pt"
        torch.save(policy.state_dict(), out)
        print(f"Saved {out}", flush=True)

if __name__ == "__main__":
    main()
