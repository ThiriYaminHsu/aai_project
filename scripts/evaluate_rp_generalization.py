"""Generalisation test: how do trained Agent R and Agent P behave on
structured-random and pure-random circuits sized smaller and larger than
each agent's own training distribution?

Training ranges (Chapter 3, sec:training-data):
  Agent R: fixed at 5 qubits, 70 gates (a single point, not a range)
  Agent P: 2-15 qubits, 10-200 gates (drawn uniformly per episode)

Two 1-D sweeps, each varying one size axis while holding the other fixed at
Agent R's training value, so results are directly readable as "smaller than
training" / "within Agent P's range" / "larger than both agents' training":
  qubit sweep:  qubits in {2,3,5,8,12,15,20,30}, gates held at 70
  gate sweep:   gates in {10,30,70,100,150,200,300,500}, qubits held at 5

At every point, circuits are generated in both the structured_random recipe
(Agent R's tuned 17/24/25/34 T/S/HSH/CNOT split) and the pure_random recipe
(uniform 25% split, Agent P's), n=3 circuits each, run through
_light_preprocess exactly as training circuits are (matching what both
agents actually learned to act on).

Uses LegacyRPEnv (legacy_env_rp.py), not the current ZXOptEnv: env.py has
since grown three more action types (gadget fusion, identity removal, the
two non-Clifford pivot variants) that did not exist, in whole or in part,
when R/P were trained -- see that module's docstring for why running their
checkpoints through the current action space would be invalid, not just
differently-scoped.

Usage:
    python scripts/evaluate_rp_generalization.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyzx as zx
import torch

from circopt_adder.baselines import baseline_basic_plus_teleport_reduce, baseline_teleport_reduce
from circopt_adder.config import DEVICE, Config
from circopt_adder.generators import _light_preprocess
from circopt_adder.legacy_env_rp import LegacyRPEnv
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import gate_count, t_count, two_qubit_gate_count

N_PER_CELL = 3
QUBIT_SWEEP = [2, 3, 5, 8, 12, 15, 20, 30]
GATE_SWEEP = [10, 30, 70, 100, 150, 200, 300, 500]
FIXED_GATES_FOR_QUBIT_SWEEP = 70
FIXED_QUBITS_FOR_GATE_SWEEP = 5

AGENT_R_QUBITS, AGENT_R_GATES = 5, 70
AGENT_P_QUBIT_RANGE, AGENT_P_GATE_RANGE = (2, 15), (10, 200)


def make_circuit(n_qubits: int, n_gates: int, family: str) -> "zx.Circuit":
    if family == "structured_random":
        g = zx.generate.cliffordT(n_qubits, n_gates, p_t=0.17, p_s=0.24, p_hsh=0.25)
    else:
        g = zx.generate.cliffordT(n_qubits, n_gates)
    return _light_preprocess(g)


def run_agent_eval_legacy(policy: ActorCriticGNN, circuit: "zx.Circuit", cfg: Config, deterministic: bool):
    env = LegacyRPEnv(lambda: circuit, cfg)
    obs, _ = env.reset()
    for _ in range(cfg.max_episode_steps):
        with torch.no_grad():
            logits, _ = policy.forward(obs.to(DEVICE))
            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                action_idx = torch.distributions.Categorical(logits=logits).sample().item()
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        if terminated or truncated:
            break
        obs = next_obs
    return env.best_circuit


def run_agent_eval_best_of_n(policy: ActorCriticGNN, circuit: "zx.Circuit", cfg: Config, n_samples: int = 10):
    best_circuit, best_2q = None, None
    for i in range(n_samples):
        c = run_agent_eval_legacy(policy, circuit, cfg, deterministic=(i == 0))
        m = two_qubit_gate_count(c)
        if best_2q is None or m < best_2q:
            best_2q, best_circuit = m, c
    return best_circuit


def main() -> None:
    cfg = Config()
    legacy_cfg = Config()
    legacy_cfg.actor_node_feat_dim = 16
    legacy_cfg.edge_feat_dim = 6

    policy_r = ActorCriticGNN(legacy_cfg).to(DEVICE)
    policy_r.load_state_dict(torch.load(REPO_ROOT / cfg.checkpoint_dir / "agent_R_random.pt", map_location=DEVICE))
    policy_r.eval()

    policy_p = ActorCriticGNN(legacy_cfg).to(DEVICE)
    policy_p.load_state_dict(torch.load(REPO_ROOT / cfg.checkpoint_dir / "agent_P_pure_random.pt", map_location=DEVICE))
    policy_p.eval()

    rows = []

    def evaluate_point(sweep: str, n_qubits: int, n_gates: int, family: str):
        for i in range(N_PER_CELL):
            circuit = make_circuit(n_qubits, n_gates, family)
            init_2q = two_qubit_gate_count(circuit)
            init_gates = gate_count(circuit)
            if init_2q == 0:
                continue  # degenerate circuit at this size, skip (nothing to reduce)

            for method_label, fn in [("teleport_reduce", baseline_teleport_reduce),
                                      ("basic_opt+teleport_reduce", baseline_basic_plus_teleport_reduce)]:
                try:
                    c_opt = fn(circuit)
                    rows.append({"sweep": sweep, "n_qubits": n_qubits, "n_gates": n_gates, "family": family,
                                 "method": method_label, "init_2q": init_2q, "final_2q": two_qubit_gate_count(c_opt),
                                 "init_gates": init_gates, "final_gates": gate_count(c_opt)})
                except Exception:
                    pass

            for label, policy in [("agent_R", policy_r), ("agent_P", policy_p)]:
                c_opt = run_agent_eval_best_of_n(policy, circuit, legacy_cfg, n_samples=10)
                rows.append({"sweep": sweep, "n_qubits": n_qubits, "n_gates": n_gates, "family": family,
                             "method": label, "init_2q": init_2q, "final_2q": two_qubit_gate_count(c_opt),
                             "init_gates": init_gates, "final_gates": gate_count(c_opt)})

    for n_qubits in QUBIT_SWEEP:
        for family in ["structured_random", "pure_random"]:
            print(f"[qubit sweep] {family} @ {n_qubits}q / {FIXED_GATES_FOR_QUBIT_SWEEP}g")
            evaluate_point("qubit_sweep", n_qubits, FIXED_GATES_FOR_QUBIT_SWEEP, family)

    for n_gates in GATE_SWEEP:
        for family in ["structured_random", "pure_random"]:
            print(f"[gate sweep] {family} @ {FIXED_QUBITS_FOR_GATE_SWEEP}q / {n_gates}g")
            evaluate_point("gate_sweep", FIXED_QUBITS_FOR_GATE_SWEEP, n_gates, family)

    df = pd.DataFrame(rows)
    df["twoq_red_pct"] = 100.0 * (df["init_2q"] - df["final_2q"]) / df["init_2q"]

    log_dir = REPO_ROOT / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv = log_dir / "rp_generalization_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    methods = ["agent_R", "agent_P", "teleport_reduce", "basic_opt+teleport_reduce"]
    colors = {"agent_R": "#c1440e", "agent_P": "#0f6e77", "teleport_reduce": "#888888", "basic_opt+teleport_reduce": "#b8a13a"}
    markers = {"agent_R": "o", "agent_P": "s", "teleport_reduce": "x", "basic_opt+teleport_reduce": "+"}

    for col, family in enumerate(["structured_random", "pure_random"]):
        ax = axes[0][col]
        sub = df[(df.sweep == "qubit_sweep") & (df.family == family)]
        piv = sub.groupby(["method", "n_qubits"])["twoq_red_pct"].mean().unstack(0)
        for m in methods:
            if m in piv.columns:
                ax.plot(piv.index, piv[m], marker=markers[m], color=colors[m], label=m)
        ax.axvspan(AGENT_P_QUBIT_RANGE[0], AGENT_P_QUBIT_RANGE[1], color="#0f6e77", alpha=0.08, label="Agent P training range")
        ax.axvline(AGENT_R_QUBITS, color="#c1440e", linestyle="--", alpha=0.5, label="Agent R training point")
        ax.set_xlabel("qubit count (gates fixed=70)")
        ax.set_ylabel("mean % reduction, 2Q gates")
        ax.set_title(f"{family}: qubit-count sweep")
        ax.legend(fontsize=7)

        ax = axes[1][col]
        sub = df[(df.sweep == "gate_sweep") & (df.family == family)]
        piv = sub.groupby(["method", "n_gates"])["twoq_red_pct"].mean().unstack(0)
        for m in methods:
            if m in piv.columns:
                ax.plot(piv.index, piv[m], marker=markers[m], color=colors[m], label=m)
        ax.axvspan(AGENT_P_GATE_RANGE[0], AGENT_P_GATE_RANGE[1], color="#0f6e77", alpha=0.08, label="Agent P training range")
        ax.axvline(AGENT_R_GATES, color="#c1440e", linestyle="--", alpha=0.5, label="Agent R training point")
        ax.set_xlabel("gate count (qubits fixed=5)")
        ax.set_ylabel("mean % reduction, 2Q gates")
        ax.set_title(f"{family}: gate-count sweep")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_png = REPO_ROOT / "results" / "figures" / "rp_generalization.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

    print("\n=== mean %% 2Q reduction by sweep/family/method ===")
    print(df.groupby(["sweep", "family", "method"])["twoq_red_pct"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
