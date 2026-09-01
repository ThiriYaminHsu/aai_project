# circopt-rl-zx: Search Distillation for ZX-Calculus Quantum Circuit Optimisation

Code, checkpoints, and evaluation data for the MSc dissertation
*"Applying reinforcement learning to ZX-calculus quantum circuit optimisation:
does training-circuit structure decide what a policy learns?"*
(WMG, University of Warwick, 2026).

The platform is inspired by Riu et al. (2025), *Reinforcement Learning Based
Quantum Circuit Optimization via ZX-Calculus* (Quantum 9, 1758); every deliberate
deviation is documented in `docs/deviations_from_paper.md`.

## Start here

**`reproduce_final_agents.ipynb`** reproduces the dissertation's final agents end to
end: the verified Cuccaro adder training data, the search-distillation pipeline, the
Chapter 4 best-of-10 evaluation (Table 4.1 reproduces exactly), and the certified
pivot-gadget motif. Quick path (released checkpoints) ~30 min; full retraining
(`RETRAIN = True`) ~30 min for Agent A and ~20 min each for Agents R and P
(Apple M5, CPU only).

```
pip install -r requirements.txt
python3 tests/test_adder_correctness.py     # exhaustive adder verification
jupyter lab reproduce_final_agents.ipynb
```

## Layout

| Path | Contents |
|---|---|
| `src/circopt_adder/` | Environment, GATv2 policy, PPO loop, generators, baselines |
| `scripts/train_agent_{r,p,a}.py` | PPO generation (the diagnosed-failure generation) |
| `scripts/train_agent_a_v4_correct.py` | Agent A by search distillation (final agent) |
| `scripts/train_agents_rp_v3.py` | Agents R and P by search distillation (final agents) |
| `scripts/evaluate_final_v3.py` | Chapter 4 evaluation (best-of-10 protocol) |
| `scripts/motif_experiments.py` | Exhaustive pivot-gadget optimality certificates |
| `scripts/plot_*.py` | Dissertation figure generators |
| `benchmarks/` | The 11 real-world arithmetic circuits (from the source paper's suite) |
| `results/checkpoints/` | Final agents (`agent_A_v4_correct.pt`, `agent_v3_128_*.pt`) and PPO agents |
| `results/logs/` | Canonical evaluation data (`evaluation_results_final_v3.csv`) |
| `tests/` | Adder correctness (exhaustive unitary simulation), env and PPO tests |
| `docs/project_log.md` | Dated experiment log for the full project |

## Headline result

Trained only on Cuccaro ripple-carry adders, the final Agent A improves every
adder bit-width 2-10 past `basic_optimization` by exactly one two-qubit gate,
including two fully held-out widths, via a single certified pivot-gadget motif;
the identically trained random-family agents only tie an untrained-network
control. Training-circuit structure decides whether learning helps at all.
