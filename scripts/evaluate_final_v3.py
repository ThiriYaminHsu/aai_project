"""Chapter-4 Table 4.1/4.2 refresh: the three FINAL second-phase agents
(R v2 = agent_v3_128_structured_random, P v2 = agent_v3_128_pure_random,
A v2 = agent_A_v3_1_128dedup) under the exact evaluate_final.py protocol:
same seeded evaluation circuits, same best-of-10 (1 argmax + 9 sampled),
same three atomic baselines re-run for one coherent CSV.

Differences from evaluate_final.py:
- Agents run through the CURRENT ZXOptEnv (18/9-dim, six action types)
  at 128 GNN channels, matching how the v3/v3.1 checkpoints were trained,
  instead of LegacyRPEnv.
- Real-world set restricted to the arithmetic subset used in Chapter 4
  is NOT done here: all 33 are evaluated (the arithmetic subset is a
  post-hoc filter, as in the current chapter pipeline).

Honesty note for the write-up: for A v2, adder bit-widths 7/8/9 are
TRAINING widths (v3/v3.1 trained on {2,3,5,6,7,8,9}); only the 4-bit
circuit in this set is genuinely held out (10-bit is reported in the
phase-2 section). The table caption must say so.

Usage:
    python scripts/evaluate_final_v2.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import torch

from benchmarks.fetch_paper_circuits import load_all as load_paper_benchmark_circuits
from circopt_adder.baselines import (
    baseline_basic_optimization,
    baseline_full_reduce,
    baseline_teleport_reduce,
)
from circopt_adder.config import DEVICE, Config
from circopt_adder.env import ZXOptEnv
from circopt_adder.generators import _light_preprocess
from circopt_adder.model import ActorCriticGNN
from circopt_adder.zx_utils import gate_count, t_count, two_qubit_gate_count
from scripts.evaluate_final import build_eval_circuits

N_SAMPLES = 10

BASELINES = [
    ("basic_opt", baseline_basic_optimization),
    ("full_reduce", baseline_full_reduce),
    ("teleport_reduce", baseline_teleport_reduce),
]


def _load_v2_policy(path: Path) -> ActorCriticGNN:
    cfg = Config()
    cfg.gnn_channels = 128
    policy = ActorCriticGNN(cfg).to(DEVICE)
    policy.load_state_dict(torch.load(path, map_location=DEVICE))
    policy.eval()
    return policy


def run_agent_best_of_n(policy: ActorCriticGNN, circuit, cfg: Config):
    best_2q, best = None, None
    for i in range(N_SAMPLES):
        env = ZXOptEnv(lambda c=circuit: c, cfg)
        obs, _ = env.reset()
        for _ in range(cfg.max_episode_steps):
            with torch.no_grad():
                logits, _ = policy.forward(obs.to(DEVICE))
                a = int(torch.argmax(logits).item()) if i == 0 else torch.distributions.Categorical(logits=logits).sample().item()
            obs2, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
            obs = obs2
        c = env.best_circuit
        m = two_qubit_gate_count(c)
        if best_2q is None or m < best_2q:
            best_2q, best = m, c
    return best


def main() -> None:
    cfg = Config()
    cfg.gnn_channels = 128

    ckpt_dir = REPO_ROOT / cfg.checkpoint_dir
    policies = {
        "agent_R_v2": _load_v2_policy(ckpt_dir / "agent_v3_128_structured_random.pt"),
        "agent_P_v2": _load_v2_policy(ckpt_dir / "agent_v3_128_pure_random.pt"),
        "agent_A_v2": _load_v2_policy(ckpt_dir / "agent_A_v4_correct.pt"),
    }

    circuits = build_eval_circuits(Config())
    rows = []
    for idx, (label, circuit) in enumerate(circuits.items()):
        init_g, init_2q, init_t = gate_count(circuit), two_qubit_gate_count(circuit), t_count(circuit)
        for mname, fn in BASELINES:
            try:
                out = fn(circuit)
                rows.append({"circuit": label, "method": mname,
                             "initial_gates": init_g, "final_gates": gate_count(out),
                             "initial_2q": init_2q, "final_2q": two_qubit_gate_count(out),
                             "initial_t": init_t, "final_t": t_count(out)})
            except Exception as e:
                rows.append({"circuit": label, "method": mname, "error": str(e)})
        # Agents receive the preprocessed circuit, exactly as in training
        # and in the documented deployment pipeline (Section 3.2's
        # preprocessing subsection); ZXOptEnv.reset() does NOT preprocess.
        # Reductions stay measured against the RAW initial counts, so the
        # agent rows report the full preprocess+agent pipeline.
        try:
            prep = _light_preprocess(circuit.to_graph() if hasattr(circuit, "to_graph") else circuit)
        except Exception:
            prep = circuit
        for pname, policy in policies.items():
            try:
                out = run_agent_best_of_n(policy, prep, cfg)
                rows.append({"circuit": label, "method": pname,
                             "initial_gates": init_g, "final_gates": gate_count(out),
                             "initial_2q": init_2q, "final_2q": two_qubit_gate_count(out),
                             "initial_t": init_t, "final_t": t_count(out)})
            except Exception as e:
                rows.append({"circuit": label, "method": pname, "error": str(e)})
        print(f"[{idx + 1}/{len(circuits)}] {label} done", flush=True)

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / cfg.log_dir / "evaluation_results_final_v3.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    def fam(s):
        if s.startswith("purerandom"): return "pure_random"
        if s.startswith("random"): return "structured_random"
        if s.startswith("adder"): return "adder_heldout"
        return "real_world"
    ok = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    ok["family"] = ok["circuit"].apply(fam)
    ARITH = ("add", "qcla", "mod")
    ok["family2"] = ok.apply(
        lambda r: "real_world_arith" if r["family"] == "real_world" and any(k in r["circuit"] for k in ARITH) else r["family"], axis=1)
    ok["twoq_red"] = 100 * (ok["initial_2q"] - ok["final_2q"]) / ok["initial_2q"]
    ok["gate_red"] = 100 * (ok["initial_gates"] - ok["final_gates"]) / ok["initial_gates"]
    order = ["structured_random", "pure_random", "adder_heldout", "real_world_arith"]
    print("\n=== 2Q reduction (arithmetic subset as real-world) ===")
    print(ok.pivot_table(index="method", columns="family2", values="twoq_red", aggfunc="mean")[order].round(1).to_string())
    print("\n=== total-gate reduction ===")
    print(ok.pivot_table(index="method", columns="family2", values="gate_red", aggfunc="mean")[order].round(1).to_string())
    print("\n=== adder per-circuit 2Q detail ===")
    ad = ok[ok["family"] == "adder_heldout"]
    print(ad.pivot_table(index="circuit", columns="method", values="final_2q").to_string())


if __name__ == "__main__":
    main()
