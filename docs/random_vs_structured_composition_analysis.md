# Random vs. structured circuit composition analysis

Companion analysis to the Agent R / Agent A ablation: how similar, in raw gate-type
makeup, are the random Clifford+T circuits used to train Agent R to the structured
circuits (ripple-carry adders, Riu et al.'s benchmark suite) used elsewhere in this
project? Produced by `scripts/analyze_random_vs_structured.py` and
`scripts/visualize_circuit_structures.py`; this document records what those scripts
found, so the result can be cited without re-running them.

## Method

Four circuit families, 100 circuits each except `riu_benchmarks`:

| Family | Generator | Size | Gate-type recipe |
|---|---|---|---|
| `structured_random` | `generators.make_random_circuit_generator` (Agent R's actual training generator) | fixed: 5 qubits, ~60 gates | fixed: T=17%, S=24%, HSH=25%, CNOT=34% (Riu et al.'s tuned recipe) |
| `pure_random` | `analyze_random_vs_structured.make_pure_random_circuit_generator` (new, this analysis) | random per circuit: 2-15 qubits, 10-200 gates | uniform: T=S=HSH=CNOT=25% (pyzx's default when no probabilities are given) |
| `cuccaro_adder` | `generators.make_adder_generator` (Agent A's actual training generator) | random bit-width per circuit, uniform in [2,9] | none -- deterministic Cuccaro ripple-carry construction, Toffolis pre-decomposed into the standard 7-T-gate template |
| `riu_benchmarks` | `benchmarks/fetch_paper_circuits.py` (Riu et al. 2025's published suite) | fixed real-world set, n=33 (not resampled -- see note below) | none -- real circuits: Toffoli cascades, QFTs, multipliers, mod-arithmetic |

`riu_benchmarks` was deliberately left at its natural n=33 rather than padded to 100 by
resampling with replacement: it is a fixed, finite, real-world benchmark corpus, and
bootstrapping duplicate rows into it would manufacture statistical weight that isn't
there.

For each circuit we count, using the same helpers `scripts/evaluate.py` relies on
(`circopt_adder.zx_utils.t_count`, `two_qubit_gate_count`): T gates, native
Toffoli/CCZ gates, 2-qubit gates (CNOT/CZ/SWAP/CRZ), and "other Clifford" (H, S, X,
...) -- each also expressed as a fraction of that circuit's total gate count. Full
per-circuit data: `results/logs/circuit_composition.csv` (333 rows). Figures:
`results/figures/composition_*.png`.

## Composition results

Mean gate-type share (± std), from `results/logs/circuit_composition.csv`:

| Family | T share | Toffoli/CCZ share | 2-qubit share | Other-Clifford share |
|---|---|---|---|---|
| `structured_random` | 0.059 ± 0.033 | 0.000 | 0.438 ± 0.049 | 0.503 ± 0.053 |
| `pure_random` | 0.070 ± 0.037 | 0.000 | 0.429 ± 0.137 | 0.501 ± 0.125 |
| `cuccaro_adder` | 0.412 ± 0.000 | 0.000 | 0.471 ± 0.000 | 0.118 ± 0.000 |
| `riu_benchmarks` | 0.019 ± 0.063 | 0.259 ± 0.141 | 0.169 ± 0.170 | 0.553 ± 0.121 |

Size, for context (gate count / qubit count, min-median-max):

| Family | gate count | qubits |
|---|---|---|
| `structured_random` | 32 - 53 - 69 | 5 (fixed) |
| `pure_random` | 6 - 91 - 233 | 2 - 9 - 15 |
| `cuccaro_adder` | 68 - 170 - 306 | 5 - 11 - 19 |
| `riu_benchmarks` | 10 - 79 - 1562 | 5 - 15 - 36 |

Three findings:

1. **The two random families land in almost the same place.** `structured_random`'s
   fixed 17/24/25/34 recipe and `pure_random`'s uniform 25/25/25/25 recipe -- despite
   having no size constraint in common -- produce circuits with nearly identical mean
   T-share (5.9% vs 7.0%) and 2-qubit-share (43.8% vs 42.9%). `pure_random` has much
   higher *variance* (std 0.137 vs 0.049 on 2-qubit-share) because its size is
   unconstrained, but the central tendency barely moves. This means the narrow,
   tightly-clustered composition profile seen in the fixed-recipe family is mostly a
   property of injecting T/S/H/CNOT gates independently at random -- not an artifact of
   Riu et al.'s specific probability tuning.
2. **`cuccaro_adder` is a T-heavy outlier by construction, not variance.** Every
   Cuccaro adder has exactly 41.2% T-gates and 47.1% 2-qubit gates, with zero variance
   across bit-widths 2-9 -- a direct consequence of building every Toffoli from the
   same fixed 7-T-gate template (`generators._add_toffoli`), so the T:2q:other ratio
   within one Toffoli gadget is invariant, and MAJ/UMA blocks are just repeated copies
   of that gadget. Structured, in the most literal sense: one fixed motif, tiled.
3. **`riu_benchmarks` is structured but not T-heavy -- it's Toffoli-heavy.** 29 of 33
   circuits contain native, undecomposed Toffoli/CCZ gates (mean Toffoli-share 25.9%);
   only 5 contain any T gates at all, and only as a small share. This is the opposite
   profile from `cuccaro_adder` despite both being "structured, non-random" circuits:
   real benchmark suites don't converge on one gate-type signature, they just
   consistently avoid the random families' profile (near-zero Toffoli, moderate
   T-share). `riu_benchmarks` is also the only family with real size heterogeneity
   (10 to 1562 gates, 5 to 36 qubits) since it's a grab-bag of genuinely different
   algorithms (Toffoli cascades, QFTs, multipliers, mod-arithmetic) rather than one
   generator swept over a size parameter.

Net read: random and structured are not points on a continuum that a probability knob
or a size knob would bridge. `pure_random` tried both (freed the size, freed the
recipe) and still landed on top of `structured_random`. What actually separates the
populations is the *generative principle* -- independent per-gate injection vs.
composition from fixed algorithmic sub-circuits -- which is a structural difference,
not a statistical one, and is why the composition scatter plot
(`composition_scatter.png`) shows two random clouds overlapping tightly in one region
and two structured families landing in disjoint regions elsewhere.

## Structural (diagram-level) comparison

`scripts/visualize_circuit_structures.py` draws actual ZX-diagrams for representative
circuits per family (`results/figures/structures_*.png`), confirming the same
conclusion visually:

- `structured_random` / `pure_random`: dense, unstructured mixes of Z/X spiders with
  scattered phase labels and crossing wires -- no repeating pattern, consistent with
  independent per-gate sampling. The two families are visually indistinguishable from
  each other.
- `cuccaro_adder`: the same MAJ/UMA block motif visibly repeats and scales with qubit
  count across bit-widths -- a clean, self-similar structure. (Note: since gate count
  is fully determined by bit-width, picking "smallest by gate count" without
  deduplication just re-draws the same bit-width 5 times; the script deduplicates on
  gate count first to show 5 *distinct* bit-widths instead.)
- `riu_benchmarks`: genuine Toffoli-cascade circuits, rendered with H-box vertices
  (pyzx's representation of the Toffoli's AND-like structure) in a visually distinct
  "staircase" pattern.

## Files

- `scripts/analyze_random_vs_structured.py` -- generates the four populations, writes
  `results/logs/circuit_composition.csv`, and produces
  `composition_stacked_mean.png` / `composition_distributions.png` / `composition_scatter.png`.
- `scripts/visualize_circuit_structures.py` -- draws 5 example circuit diagrams per
  family into `structures_{family}.png`.
- `results/logs/circuit_composition.csv` -- full per-circuit metrics (333 rows: name,
  family, n_qubits, gate_count, t_count/frac, toffoli_count/frac, twoq_count/frac,
  other_clifford_count/frac, clifford_count, h_count, s_count).

Re-run either script directly (`python scripts/analyze_random_vs_structured.py`) to
regenerate with a different seed or population size; both scripts are self-contained
given the package install (`pip install -e .`).
