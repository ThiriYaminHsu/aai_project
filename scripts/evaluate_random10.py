"""Re-evaluate ONLY the two random evaluation families with 10 circuits
each (same eval seed SEED+1, so circuits 0-4 are identical to the
5-circuit run and 5-9 are new), under the exact evaluate_final_v3
protocol. Merges with the adder and real-world rows of
evaluation_results_final_v3.csv into evaluation_results_final_v4.csv,
the new canonical Chapter 4 dataset.

Usage: python3 scripts/evaluate_random10.py
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd, torch
from circopt_adder.baselines import (baseline_basic_optimization,
    baseline_full_reduce, baseline_teleport_reduce)
from circopt_adder.config import DEVICE, SEED, Config, set_seed
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import (_light_preprocess,
    make_pure_random_circuit_generator, make_random_circuit_generator)
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import gate_count, t_count, two_qubit_gate_count

N_SAMPLES = 10
BASELINES = [("basic_opt", baseline_basic_optimization),
             ("full_reduce", baseline_full_reduce),
             ("teleport_reduce", baseline_teleport_reduce)]

def load_policy(name):
    cfg = Config(); cfg.gnn_channels = 128
    p = ActorCriticGNN(cfg).to(DEVICE)
    p.load_state_dict(torch.load(REPO_ROOT / cfg.checkpoint_dir / name, map_location=DEVICE))
    p.eval(); return p

def best_of_n(policy, circuit, cfg):
    best_2q, best = None, None
    for i in range(N_SAMPLES):
        env = ZXOptEnv(lambda c=circuit: c, cfg)
        obs, _ = env.reset()
        for _ in range(cfg.max_episode_steps):
            with torch.no_grad():
                logits, _ = policy.forward(obs.to(DEVICE))
                a = int(torch.argmax(logits).item()) if i == 0 else torch.distributions.Categorical(logits=logits).sample().item()
            obs2, r, term, trunc, info = env.step(a)
            if term or trunc: break
            obs = obs2
        c = env.best_circuit; m = two_qubit_gate_count(c)
        if best_2q is None or m < best_2q: best_2q, best = m, c
    return best

def main():
    set_seed(SEED)
    cfg = Config(); cfg.gnn_channels = 128
    policies = {"agent_R_v2": load_policy("agent_v3_128_structured_random.pt"),
                "agent_P_v2": load_policy("agent_v3_128_pure_random.pt"),
                "agent_A_v2": load_policy("agent_A_v4_correct.pt")}
    circuits = {}
    g = make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random, seed=SEED + 1)
    for i in range(10): circuits[f"random_{i}"] = g()
    g = make_pure_random_circuit_generator(cfg.pure_random_min_qubits,
        cfg.pure_random_max_qubits, cfg.pure_random_min_gates,
        cfg.pure_random_max_gates, seed=SEED + 1)
    for i in range(10): circuits[f"purerandom_{i}"] = g()

    rows = []
    for idx, (label, circuit) in enumerate(circuits.items()):
        init_g, init_2q, init_t = gate_count(circuit), two_qubit_gate_count(circuit), t_count(circuit)
        for mname, fn in BASELINES:
            out = fn(circuit)
            rows.append({"circuit": label, "method": mname,
                "initial_gates": init_g, "final_gates": gate_count(out),
                "initial_2q": init_2q, "final_2q": two_qubit_gate_count(out),
                "initial_t": init_t, "final_t": t_count(out)})
        prep = _light_preprocess(circuit.to_graph() if hasattr(circuit, "to_graph") else circuit)
        for pname, policy in policies.items():
            out = best_of_n(policy, prep, cfg)
            rows.append({"circuit": label, "method": pname,
                "initial_gates": init_g, "final_gates": gate_count(out),
                "initial_2q": init_2q, "final_2q": two_qubit_gate_count(out),
                "initial_t": init_t, "final_t": t_count(out)})
        print(f"[{idx+1}/20] {label} done", flush=True)

    new = pd.DataFrame(rows)
    old = pd.read_csv(REPO_ROOT / "results/logs/evaluation_results_final_v3.csv")
    keep = old[~(old["circuit"].str.startswith("random") | old["circuit"].str.startswith("purerandom"))]
    merged = pd.concat([new, keep], ignore_index=True)
    out_csv = REPO_ROOT / "results/logs/evaluation_results_final_v4.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
