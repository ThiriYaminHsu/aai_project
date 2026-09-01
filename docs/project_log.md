# Project log

Chronological record of work done on this project in AI-assisted sessions, kept so
the Methodology and Results chapters can be written from an accurate account of what
was actually done, in what order, and why -- rather than reconstructed from memory
afterwards. Each entry gives: what was being tried, the concrete method/parameters,
the decisions made and their rationale, what was found, and which files it produced.
Dates are taken from git commit timestamps and file modification times, not estimated.

Repo: `circopt-rl-zx-dissertation`. Environment: Python 3.14, pyzx, PyTorch,
torch-geometric, on macOS (Darwin), no GPU (all runs on CPU).

---

## 2026-08-17 -- Random-vs-structured circuit composition analysis

### 1. Initial composition analysis (random vs. adders vs. paper benchmarks)

**Goal.** Characterise, quantitatively, how similar the random Clifford+T circuits
used to train "Agent R" (replicating Riu et al. 2025's training regime) are to
structured circuits (ripple-carry adders; Riu et al.'s benchmark suite), by gate-type
composition -- T-count, Toffoli-count, 2-qubit-gate count, Clifford count.

**Method.**
- Wrote `scripts/analyze_random_vs_structured.py`. For each circuit, counted gates by
  type using the project's existing helpers (`circopt_adder.zx_utils.t_count`,
  `two_qubit_gate_count`) plus a direct `Counter` over `circuit.gates` for
  native `Tof`/`CCZ` gates, so the numbers are consistent with what `evaluate.py`
  already reports elsewhere in the project.
- Three families generated initially:
  - **random**: 100 circuits from `generators.make_random_circuit_generator(n_qubits=5,
    n_gates=60, seed=42)` -- i.e. Agent R's actual training generator (fixed size,
    fixed gate-type recipe T=17%/S=24%/HSH=25%/CNOT=34%, matching Riu et al.'s
    released code, not their paper's prose summary).
  - **adder**: 8 circuits, one `ripple_carry_adder(n)` per bit-width `n` in `2..9`.
  - **paper**: Riu et al.'s full benchmark suite, downloaded via the project's
    existing `benchmarks/fetch_paper_circuits.py` (33 circuits: Toffoli cascades,
    QFTs, multipliers, mod-arithmetic).
- Hit an SSL certificate error downloading the benchmark suite (`urlopen` had no
  local CA bundle wired up on this machine's Python install). Fixed by pointing
  `ssl._create_default_https_context` at `certifi.where()` before the download call
  -- a local-machine fix, not a project code change.
- Produced 3 figures (dataviz-skill palette, categorical colour per family):
  `composition_stacked_mean.png` (mean gate-type share per family, stacked bar),
  `composition_distributions.png` (box+strip plots of T-share and 2-qubit-share per
  family), `composition_scatter.png` (T-share vs. 2-qubit-share per circuit,
  coloured by family -- the direct "how similar are these populations" view).
- Wrote `results/logs/circuit_composition.csv` (one row per circuit: name, family,
  n_qubits, gate_count, t_count/frac, toffoli_count/frac, twoq_count/frac,
  other_clifford_count/frac, clifford_count, h_count, s_count).

**Result (first pass, n=141: 100 random + 8 adder + 33 paper).** Random circuits
formed a tight, narrow cloud (T-share ~6%, 2q-share ~45%, low variance); adders had a
fixed, zero-variance T-share of 41.2% (a direct consequence of every Toffoli being
built from the same fixed 7-T-gate template); paper benchmarks were the most
different from random -- mostly zero T-count but native, undecomposed Toffoli/CCZ
gates instead (since some of Riu et al.'s benchmark QASM/qc files contain literal
`ccx` gates that pyzx loads as `Tof` gates rather than decomposing on load).
Essentially no overlap between the random cloud and either structured family.

### 2. Renamed families to specific names

**Trigger.** Family labels "adder" and "paper" were too generic.
Renamed **adder → cuccaro_adder** (after Cuccaro et al. 2004, whose construction
`generators.ripple_carry_adder` implements) and **paper → riu_benchmarks** (after
Riu, Nogué, Vilaplana, Garcia-Saez & Estarellas 2025, whose suite it is). Same data,
same conclusions, clearer provenance in every plot legend/CSV column.

### 3. Scaled random population to n=1000

Increased the random family from 100 to 1000 circuits (`N_RANDOM = 1000`) to check
the tight-cloud finding wasn't a small-sample artefact. Ran in ~3s on CPU. Result
unchanged and sharper: random cloud mean T-share 6.0% ± 2.8%, 2q-share unchanged,
still no overlap with either structured family.

### 4. Added a fourth family: pure_random (unconstrained random circuits)

**Motivation.** The "random" family's tight cloud could in principle be an artefact
of Riu et al.'s specific, tuned sampling recipe (T=17%/S=24%/HSH=25%/CNOT=34%) rather
than a property of random Clifford+T circuits in general. Added a second random
generator with *nothing* fixed, to test that directly.

**Method.** New function `make_pure_random_circuit_generator(min_qubits, max_qubits,
min_gates, max_gates, seed)`: draws `n_qubits` and `n_gates` uniformly at random per
circuit (initially range 2-15 qubits, 10-200 gates), and passes no `p_t`/`p_s`/`p_hsh`
to `zx.generate.cliffordT`, so pyzx falls back to a uniform 25/25/25/25 split over
T/S/HSH/CNOT rather than the tuned recipe. 100 circuits generated this way.
Since 4 overlapping-colour families exceed the dataviz palette's all-pairs-safe
3-colour limit for scatter plots, `composition_scatter.png` was changed from one
overlaid scatter to 4 faceted small-multiple panels (one colour per panel, shared
axes) -- avoids the CVD-safety problem entirely rather than working around it.

**Result.** `pure_random` was visibly more *spread out* than the fixed-recipe
`random` family (2q-share std 0.137 vs. 0.049) but landed in essentially the *same
region* of the T-share/2q-share plane (mean T-share 7.0% vs. 6.0%). Conclusion: the
narrow cloud is not an artefact of Riu et al.'s specific probability tuning -- any
Clifford+T circuit built by independent per-gate random injection lands in a
characteristically different region from circuits built from real algorithmic
structure, regardless of the exact injection probabilities or circuit size range.

### 5. Equalised sample sizes across families; renamed random → structured_random

**Trigger.** Wanted every family sampled at the same size (n=100) for a fair
comparison, and a name for the fixed-recipe family that reflects *why* it's
different from `pure_random` (a fixed recipe is itself a structure being imposed).

**Decisions.**
- `structured_random` (renamed from `random`) and `pure_random`: both set to n=100
  (structured_random was reduced from 1000 → 100 to match).
- `cuccaro_adder`: switched from "one deterministic circuit per bit-width 2-9" (n=8,
  with duplicates impossible) to **100 draws from `generators.make_adder_generator(2,
  9, seed=42)`**, which samples a random bit-width per call -- this is also literally
  Agent A's actual training generator, so the comparison population now mirrors a real
  training regime the same way `structured_random` mirrors Agent R's.
- `riu_benchmarks`: **left at its natural n=33** rather than padded to 100 by
  resampling with replacement. Explicitly asked and decided: it is a fixed,
  finite, real-world benchmark corpus; bootstrapping duplicate rows into it would
  manufacture statistical weight that isn't there. (This is a deliberate methodological
  choice worth stating explicitly in the dissertation: sample-size parity was enforced
  for the two generated-on-demand families but not forced onto a fixed real dataset.)
- Family display labels updated to show each one's specific recipe/parameterisation
  inline (e.g. `structured_random (T17/S24/H25)`, `cuccaro_adder (n_bits~U[2,9])`,
  `riu_benchmarks (fixed, n=33)`) rather than a generic name.

**Bug fixed along the way.** The first attempt at showing per-family recipe details
on chart x-axis labels used two-line labels that were too wide for the column
spacing and visually overlapped between adjacent categories. Fixed by shortening the
label text and widening the figures (and, separately, fixed a `set_xticklabels`
`UserWarning` by calling `set_xticks` before `set_xticklabels`).

**Result (n=333: 100+100+100+33), current numbers on disk in
`results/logs/circuit_composition.csv`:**

| family | n_qubits (mean±sd) | gate_count (mean±sd) | T-share | Toffoli-share | 2q-share | other-Clifford-share |
|---|---|---|---|---|---|---|
| structured_random | 5.00 ± 0.00 | 53.3 ± 7.8 | 0.056 ± 0.029 | 0.000 | 0.446 ± 0.051 | 0.497 ± 0.050 |
| pure_random | 8.75 ± 3.84 | 98.8 ± 59.6 | 0.077 ± 0.036 | 0.000 | 0.409 ± 0.139 | 0.514 ± 0.127 |
| cuccaro_adder | 11.56 ± 4.27 | 179.5 ± 72.5 | 0.412 ± 0.000 | 0.000 | 0.471 ± 0.000 | 0.118 ± 0.000 |
| riu_benchmarks | 15.36 ± 8.47 | 160.5 ± 281.4 | 0.019 ± 0.063 | 0.259 ± 0.141 | 0.169 ± 0.170 | 0.553 ± 0.121 |

Of the 33 `riu_benchmarks` circuits, 29 contain native (undecomposed) Toffoli/CCZ
gates and only 5 contain any T gates at all. `cuccaro_adder` has exactly zero
variance in its gate-type shares across all 100 draws (different bit-widths, same
ratios) -- a direct consequence of every Toffoli being built from one fixed 7-T-gate
template, so the MAJ/UMA blocks are just repeated copies of that template regardless
of scale.

**Note on reproducibility.** `zx.generate.cliffordT` draws from pyzx's own global RNG
state rather than an independently-seeded stream (documented in
`generators.make_random_circuit_generator`'s own docstring), so re-running this
script with the same `seed=42` does **not** reproduce bit-identical circuits between
runs -- only the same *statistical* population. The table above is one such run's
numbers, not a fixed ground truth; a re-run will show the same qualitative pattern
(tight random cloud, zero-variance adder T-share, Toffoli-heavy paper benchmarks)
with slightly different exact decimals. Worth stating explicitly as a methodology
limitation.

### 6. Diagram-level structural comparison (5 example circuits per family)

**Goal.** Compare *what the circuits actually look like*, not just aggregate
statistics, to visually corroborate the composition-analysis conclusion.

**Method.** New script `scripts/visualize_circuit_structures.py`. Refactored
`analyze_random_vs_structured.build_dataset()` to also return `{circuit_name:
zx.Circuit}` (previously it only returned the metrics dataframe) so the exact same
population already analysed could be drawn, not a freshly-regenerated one. For each
family, picked circuits using `sort_values('gate_count').drop_duplicates(subset=
'gate_count').head(5)` -- deduplicated on gate count *first*.

**Bug caught and fixed before shipping.** The first version picked the 5 smallest by
gate count without deduplication; for `cuccaro_adder`, gate count is fully
determined by bit-width, so this just re-drew 5 copies of the same (smallest)
bit-width. Fixed by deduplicating on `gate_count` before taking the 5 smallest, which
gives 5 genuinely distinct bit-widths instead -- a much more informative comparison
(shows the MAJ/UMA motif visibly repeating and scaling with qubit count).

Circuits rendered with `zx.draw()` (pyzx's own ZX-diagram renderer, `labels=False`,
figure size scaled to `n_gates`/`n_qubits`), composited per-family into one PNG panel
each via Pillow (`results/figures/structures_{family}.png`), with per-circuit
captions (name, qubit count, gate count).

**Result.** `structured_random`/`pure_random`: dense, unstructured mixes of Z/X
spiders with scattered phase labels and crossing wires, no repeating pattern, and
visually indistinguishable from each other. `cuccaro_adder`: the same MAJ/UMA block
motif visibly repeats and scales across bit-widths 5/7/9/11/13 qubits. `riu_benchmarks`:
genuine Toffoli-cascade circuits, rendered with yellow H-box vertices (pyzx's
representation of the Toffoli's AND-like structure) in a distinct "staircase"
pattern -- visually confirms the Toffoli-heavy finding from the composition numbers.

### 7. Wrote up the analysis

Wrote `docs/random_vs_structured_composition_analysis.md`: methodology table, the
composition-results table above, three stated findings (random families converge to
near-identical composition regardless of size/recipe constraints; `cuccaro_adder` is
a zero-variance T-heavy outlier by construction; `riu_benchmarks` is structured via
Toffoli-heaviness rather than T-heaviness), the diagram-level confirmation, and a
file index. Added a one-line pointer to it from `README.md`'s structure listing.

### 8. Committed and pushed

One commit (`9a6eba6`, 2026-08-17 15:04), everything together per an explicit
decision at the time (rather than splitting): `README.md`, the new analysis doc, the
2 new scripts, and 8 new result figures, bundled with an unrelated already-staged
`CITATION.cff` author-name fix (a personal edit made directly in the IDE, not
part of this work, but bundled in on request rather than left as a stray staged
change). Pushed to `origin/main` (`9c9dc1a..9a6eba6`).

---

## 2026-08-17 -- Three-way training-regime ablation (Agent R / Agent P / Agent A)

### 9. Research question and test design

**Hypothesis under test:** training an RL circuit-optimisation agent on different
circuit-generation regimes (random; structured-but-random-recipe; genuinely
structured/arithmetic) produces agents with different circuit-optimisation
performance.

**Design decision.** `riu_benchmarks` is **validation-only** -- no agent trains on
it. Its specific purpose is to let the trained agents' results be compared directly
against Riu et al. (2025)'s own reported numbers (Table 2 of their paper): since
these are the exact same benchmark circuits the original paper evaluated on, running
Agent R/P/A against them is a like-for-like check against the paper's published
results, not just a generic held-out test set. This is distinct from its other role
in the composition analysis above (§1-8), where it served as a real-world reference
population to compare gate-type composition against, and from `evaluate.py`'s
`random`/`pure_random`/adder-interpolation/adder-extrapolation sets, which test
generalisation within the project's own training distributions rather than against
the paper. The three *training* regimes are: `structured_random` (Agent R,
pre-existing), `pure_random` (new: Agent P), `cuccaro_adder` (Agent A, pre-existing).
This mirrors the composition-analysis families exactly, so the same generator code
is reused for both purposes.

### 10. Built Agent P's training regime

**Method.**
- Moved `make_pure_random_circuit_generator` out of the analysis script and into
  `src/circopt_adder/generators.py` (previously it lived only in
  `analyze_random_vs_structured.py`) -- its proper, reusable home alongside
  `make_random_circuit_generator` and `make_adder_generator`, since it's now used by
  both a training script and the analysis script. `analyze_random_vs_structured.py`
  updated to import it from there instead of defining its own copy.
- Added `Config` fields: `pure_random_min_qubits=2`, `pure_random_max_qubits=15`,
  `pure_random_min_gates=10`, `pure_random_max_gates=200` (`src/circopt_adder/config.py`).
- New `scripts/train_agent_p.py`, structurally identical to `train_agent_r.py`/
  `train_agent_a.py` (same PPO call, same `--smoke-test` flag), differing only in
  which generator it passes to `ZXOptEnv`. Checkpoint/run name: `agent_P_pure_random`.
- Extended `scripts/evaluate.py`: loads all three checkpoints (`agent_R_random.pt`,
  `agent_P_pure_random.pt`, `agent_A_adder.pt`); added a `pure_random` evaluation set
  (5 circuits from the same generator/range, `seed=SEED+1`) alongside the existing
  fixed-recipe `random` sanity-check set; updated `_categorize`, `build_result_tables`,
  and `plot_comparison` to handle 5 evaluation buckets instead of 4 (random,
  pure_random, adder-interpolation, adder-extrapolation, paper/validation-only).

**Verification before running anything.** Reran the composition-analysis script
(confirms the generator-relocation refactor didn't change behaviour); ran
`train_agent_p.py --help` (argparse sanity check); `ast.parse` syntax-checked every
edited file; a dry run of `evaluate.py`'s `build_eval_circuits()` (confirmed the
5-way category split: 5 random / 5 pure_random / 4 adder / 33 paper); full existing
test suite (`pytest tests/`, 8 passed).

### 11. Environment bug found and fixed: stale pyzx version

**What happened.** Launched smoke-test training runs (`--smoke-test`, 10,000
timesteps, ≈19 PPO updates at `n_steps=512`) for all three agents in parallel.
Agent P crashed almost immediately with:
```
NameError: name 'zx' is not defined
  File ".../site-packages/pyzx/simplify.py", line 450, in to_graph_like
    z1 = g.add_vertex(ty=zx.VertexType.Z)
```
i.e. inside **pyzx's own installed code**, not this project's. Agent R crashed with
the identical traceback shortly after, at PPO update 6/19 -- proving the bug was not
specific to `pure_random`'s circuit shapes, just a rarely-hit code path that
`structured_random`'s narrower circuit distribution happened to reach later.

**Root cause.** `pip show pyzx` reported version **0.7.0** installed, despite both
`pyproject.toml` and `requirements.txt` pinning `pyzx>=0.10.5`. `docs/
deviations_from_paper.md` explicitly documents that this project's baseline
substitutions (`clifford_simp`, `teleport_reduce`) are reasoned about *assuming*
pyzx 0.10.5's available API -- so running on 0.7.0 was silently out of spec for the
whole project, not just this run.

**Fix.** `pip3 install --upgrade "pyzx>=0.10.5"` → installed 0.10.5 (also upgraded
`lark` 1.1.9 → 1.3.1 as a transitive dependency). Verified: full test suite still
passes (8/8); a targeted stress test pushed 20 circuits from each of the three
generators through `circuit_to_graphlike` (the function that crashed) with no
errors. Stopped the still-running Agent A smoke test proactively (it would have hit
the same bug eventually) rather than letting it fail on its own, then relaunched all
three smoke tests from scratch.

**Methodology note.** This is worth stating plainly in the dissertation: an
environment/dependency drift (not a bug in the project's own code) caused training
failures that were circuit-distribution-dependent in *when* they surfaced but not in
*why* -- a reminder that apparent "some circuits are more fragile than others"
symptoms are worth root-causing against the dependency stack before concluding
anything about the circuits themselves.

### 12. Smoke-test training runs (all three agents)

**Method.** All three launched with `--smoke-test` (`total_timesteps=10_000`,
`log_interval=1`, all other `Config` defaults: `n_qubits=5` for Agent R,
`n_steps=512`, `n_envs=8` (unused by the current single-env PPO loop),
`learning_rate=2e-4`, `n_epochs=8`, `minibatch_size=512`, `gamma=0.99`,
`gae_lambda=0.95`, `vf_coef=0.5`, `ent_coef=0.01`, `clip_eps=0.1`,
`max_grad_norm=0.5`, `max_episode_steps=35`, `reward_mode="twoqubits"`, GATv2
actor-critic, 5 layers, 32 channels, `seed=42`). Run on CPU (no CUDA available);
wall-clock ≈7-8s/update for Agent R, ≈12-15s/update for Agent P and Agent A
(circuits ~2-3x larger on average) → roughly 15-16 minutes per agent for the full
19-update smoke test.

**Result.** All three completed 19/19 PPO updates without error and saved
checkpoints + per-update training logs:
- `results/checkpoints/agent_R_random.pt` (68.6 KB), `agent_R_random_train_log.csv`
- `results/checkpoints/agent_P_pure_random.pt` (68.8 KB), `agent_P_pure_random_train_log.csv`
- `results/checkpoints/agent_A_adder.pt` (68.6 KB), `agent_A_adder_train_log.csv`

Mean episode reward stayed noisy/negative across all three (expected -- 19 updates
is far too little training signal to show learning; this run's purpose was pipeline
validation, not a real result).

### 13. Ran full evaluation on the smoke-test checkpoints, and diagnosed why the numbers looked flat

**Method.** Ran `scripts/evaluate.py` against the three smoke-test checkpoints
(baselines: `full_reduce`, `basic_opt+full_reduce`; evaluated on the 5-way circuit
set from §10). Separately, wrote and ran a small standalone diagnostic: stepped one
fixed circuit through all three policies deterministically and logged the exact
action sequence taken and the best gate count found at each step.

**Result / finding.** All three policies picked nearly identical action sequences
(mostly alternating `PIV`/`LC` in the same order) and never improved on the gate
count already achieved by the environment's own one-shot `extract_and_cleanup` step
inside `reset()` -- e.g. on one 48-gate test circuit, all three reported
`steps_taken=11`, `final best=48` (no reduction at all from the learned policy
itself). This explains why the evaluation summary showed nearly identical
% -reduction numbers across Agent R/P/A: those numbers came almost entirely from the
shared `reset()`-time baseline cleanup (identical code path for all three), not from
anything the GNN policies had learned to do differently.

**Conclusion for methodology/results write-up.** The smoke test validates that the
full pipeline (generator → `ZXOptEnv` → PPO → checkpoint → `evaluate.py`) runs
end-to-end without error across all three training regimes, but 19 PPO updates
(10,000 timesteps) is nowhere near enough for a randomly-initialised GNN policy to
diverge from near-initialisation behaviour. **The three-regime hypothesis has not
yet been tested** -- that requires the full `total_timesteps=2_000_000` runs (or at
minimum several orders of magnitude more updates than 19) before a performance
comparison between Agent R/P/A is meaningful. This is an important distinction to
preserve in the dissertation: what was validated here is *pipeline correctness*, not
the *research hypothesis*.

**Status as of this log entry:** full training runs have not yet been started.

---

## 2026-08-23 -- Dissertation LaTeX template

Not part of the research methodology -- infrastructure for writing the dissertation
itself, logged here for completeness/traceability only.

### 14. Converted the WMG dissertation template from .docx to LaTeX

**Source.** `../Writing/25-26-wmg-ft-msc-dissertation-template-v2.docx` (the official
WMG MSc dissertation template, already partially populated with this project's own
chapter headings and a 31-entry reference list).

**Method.** Converted the .docx to HTML via macOS `textutil` to read its content
(no `pandoc`/`python-docx` available locally). Unzipped the .docx directly (it is a
zip of OOXML files) and inspected `word/document.xml`'s `<w:sectPr>` elements to
recover exact page geometry that the HTML conversion loses: A4 paper; 1.5cm margins
on the main body (`w:pgMar w:top/right/bottom/left="851"` twips = exactly 1.5cm);
2.54cm margins on the pro-forma page only; lower-roman page numbering for front
matter (Declaration through List of figures); arabic numbering restarting at 1 from
Chapter 1; blank headers and a page-number-only footer (`word/header1.xml`,
`header2.xml` empty; `word/footer1.xml` contains only a `PAGE` field).

**Attempted, but could not, verify by compiling.** Tried installing a minimal TeX
distribution (`brew install --cask basictex`) to test-compile the result; the
installer requires an interactive `sudo` password this environment can't supply, so
no local LaTeX toolchain is available. The LaTeX below was written by hand using
only long-established, universally-bundled packages and was checked with static
sanity passes (brace-balance, `\begin`/`\end` pairing, unescaped-special-character
scan, and a duplicate/count check on the 31 BibTeX entries) -- but has **not been
compiled**, and should be verified (e.g. via Overleaf, which needs no local install)
before being trusted for submission-grade formatting.

**Output**, under `../Writing/latex/`: `main.tex` (book class, `geometry`, `times`,
`natbib`+`plainnat.bst` for Harvard-style citations, `\frontmatter`/`\mainmatter`
page-numbering scheme reconstructed from the section-properties above);
`frontmatter/{proforma,titlepage,declaration,abstract,acknowledgements}.tex`;
`chapters/chapter01_introduction.tex` through `chapter07_conclusion.tex` (headings
transcribed exactly from the docx -- the real headings the student had already
chosen for Chapters 1-2 preserved verbatim, the still-generic "Level 2/3/4 heading"
placeholder scaffolding for Chapters 3-7 preserved as-is rather than invented over);
`appendices/appendixA.tex`, `appendixB.tex`; `references.bib` (all 31 references
from the docx transcribed to BibTeX, `\nocite{*}` used so they all print despite no
in-text `\citep`/`\citet` calls existing yet in the still-placeholder chapters).

**Judgement calls made where the source was ambiguous** (worth citing in a
methodology/limitations note if this template is mentioned in the dissertation):
- Chapter 2's heading nesting depth (which headings are `\section` vs `\subsection`
  vs `\subsubsection`) could not be recovered from the lossy HTML conversion, since
  Word's named heading styles didn't survive it; the nesting used was inferred from
  topic logic (Clifford/Non-Clifford Gates as subtopics of Quantum Circuits and
  Quantum Gates) and flagged with an explicit comment in the `.tex` source.
  Chapter 6 preserves an apparent quirk in the source (a "Level 3 heading" nested
  directly under the chapter's Introduction, before any "Level 2 heading" appears)
  rather than silently correcting it, since it's unclear whether that's deliberate.
- Exact roman-numeral page number on the Declaration page couldn't be reconciled
  between two conflicting signals in the docx's own XML (`w:pgNumType` implies "i",
  the typed-out Table-of-Contents text says "ii"); the typed ToC text was treated as
  the authoritative statement of intent and the numbering built to match it exactly.
- The duplicated "name/department of collaborating partner" and repeated
  "WMG / University of Warwick" lines on the source title page (present twice
  consecutively) were treated as a docx artefact and collapsed to one occurrence.

**Not yet done:** `../Writing/latex/README.md` (compile instructions and a written-out
version of the judgement-call list above) was in progress when this log was
requested instead.

---

## 2026-08-24 -- Optimization-target clarification

### 15. Confirmed the training objective is two-qubit gate count, and fixed a metric bug in evaluate.py

**Trigger.** Decided to focus training explicitly on reducing two-qubit gate count
rather than T-count.

**Finding.** `Config.reward_mode` already defaults to `"twoqubits"`
(`src/circopt_adder/config.py`), and `ZXOptEnv._metric()` (`src/circopt_adder/env.py`)
already computes `two_qubit_gate_count(circuit)` for that mode -- matching Riu et
al.'s own hardcoded default (`gate_type="twoqubits"` in their released `zx_env.py`).
Checked all three training scripts (`train_agent_r.py`, `train_agent_p.py`,
`train_agent_a.py`): none override `reward_mode`. **So no training-side change was
needed** -- the reward signal for all three smoke-test runs in §12 already optimised
two-qubit gate count, not T-count.

**Bug found and fixed.** `scripts/evaluate.py`'s `plot_comparison()` only branched on
`reward_mode == "t_count"` vs. everything else, so for the actual default
(`"twoqubits"`) it silently computed and plotted the **total gate count** reduction
(`initial_gates`/`final_gates`) instead of the two-qubit-gate-count reduction
(`initial_2q`/`final_2q`) -- the y-axis label was wrong too (`"gate count"` instead of
`"2-qubit gate count"`). This only affected the `comparison_barplot.png` plot; the
underlying per-circuit `evaluation_results.csv` and the printed summary tables
(`build_result_tables`) already computed all three reduction metrics unconditionally
and were unaffected. Fixed by replacing the two-way branch with a
`reward_mode -> (init_col, metric_col)` lookup covering all three modes
(`_METRIC_COLS`, `_METRIC_NAMES` module-level dicts in `evaluate.py`), and reran
`scripts/evaluate.py` to regenerate `comparison_barplot.png` against the existing
smoke-test checkpoints with the corrected metric.

**Methodology note.** Worth stating in the dissertation as the confirmed optimisation
target: all three agents (R/P/A) are trained to minimise two-qubit gate count per
episode (Riu et al.'s `twoqubits` reward mode), not T-count or total gate count --
`total_gates` and `t_count` remain available as `Config.reward_mode` alternatives for
future ablations but were not used for any run so far.

### 16. Found and fixed a gate-basis mismatch in the paper-benchmark evaluation

**Trigger.** Inspecting the just-regenerated `comparison_barplot.png` (§15): the
`paper` (riu_benchmarks) eval set showed both deterministic baselines "reducing"
2-qubit gate count by **-4000% to -6000%** (i.e. massively *increasing* it, with huge
error bars), and no visible agent bars for that eval set at all.

**Root cause.** Some `riu_benchmarks` circuits load with native Toffoli/CCZ gates
still intact (3-qubit gates, not counted as "2-qubit" or "T" by
`two_qubit_gate_count`/`t_count`). Every downstream method in the evaluation
(`full_reduce`, `basic_opt+full_reduce`, and the RL agents via `ZXOptEnv.reset()`)
converts to graph-like ZX form and re-extracts, which decomposes every Toffoli into
elementary Clifford+T gates (verified: one `TOF` gate → 7 T + 6 CNOT + 2 H via
`Circuit.to_basic_gates()`). So "initial" counts were measured on the native-Toffoli
circuit and "final" counts on the fully-decomposed one -- two different gate bases,
not a real regression. Concretely, on `paper_gf2^7_mult`: `initial_gates=153,
initial_2q=6, initial_t=0` (Toffoli-native) → after `full_reduce`,
`final_gates=1887, final_2q=1220, final_t=125` (decomposed) -- a "-20,233% reduction"
that is entirely a basis-mismatch artefact, not the baseline making the circuit
worse. Affects at least 17/33 `riu_benchmarks` circuits (`initial_2q < 10`).
Same inspection also reconfirmed §13's finding: all three agents produced
byte-identical output on this circuit (615 gates/300 2q/301 T each) -- the
smoke-test checkpoints still haven't diverged.

**Decision (asked, not assumed).** Two options were possible: (a) decompose every
`riu_benchmarks` circuit to Clifford+T basis immediately after loading, so "initial"
and "final" are measured on the same basis; (b) leave measurement as-is and document
the reduction-% metric as not meaningful for Toffoli-native circuits. Chose **(a)**,
since it matches how the rest of this pipeline already treats circuits (Clifford+T
throughout -- see `docs/deviations_from_paper.md`'s note that the action space only
handles the Clifford+T fragment) rather than introducing a second, inconsistent
convention just for this evaluation.

**Fix.** In `scripts/evaluate.py`'s `build_eval_circuits()`, every loaded paper
benchmark circuit is now passed through `.to_basic_gates()` before being added to
`eval_circuits`, so `initial_gates`/`initial_2q`/`initial_t` are measured on the same
elementary-gate basis that `final_*` will already be in after any processing.
**Scoped deliberately to `evaluate.py` only** -- `benchmarks/fetch_paper_circuits.py`
and `scripts/analyze_random_vs_structured.py` were left untouched, since the
composition analysis (§1-8) *intentionally* measures these circuits' native,
as-published gate composition (including native Toffoli/CCZ), and that finding
("29/33 riu_benchmarks circuits contain native Toffoli/CCZ gates") is itself a
correct, already-published result that decomposing at the source would have erased.

**Confirmed.** Re-ran `evaluate.py` (took noticeably longer than before: decomposed
paper circuits are much bigger -- `paper_gf2^7_mult` went from 153 gates as-loaded to
741 after `to_basic_gates()` -- so RL episodes over them take longer). Same circuit
now: `initial_2q=300` on all methods (consistent basis), `full_reduce` →
`final_2q=1220`, `basic_opt+full_reduce` → `final_2q=1175`, all three agents →
`final_2q=300` (unchanged) with `final_t` down from 343 to 301.
`comparison_barplot.png` is now on a sane, interpretable scale (-130% to +30%,
previously -6000% to +30%) across every eval set.

**Unexpected finding, worth reporting as-is rather than smoothing over.** With the
metric now trustworthy, the deterministic pyzx baselines (`full_reduce`,
`basic_opt+full_reduce`) *consistently increase* 2-qubit gate count relative to a
Clifford+T-decomposed starting point, across every eval set (`random` ≈ -20%,
`interpolation`/`extrapolation` adders ≈ -55% to -60%, `paper` ≈ -75% to -95%) --
i.e. pyzx's own graph-like simplify-and-re-extract routines, run through this
project's `baselines.py` wrappers, expand 2-qubit gate count rather than shrink it on
every circuit family tested here. The (still smoke-test, undertrained) RL agents
never make things worse and show small positive reductions (0-30%) with wide/tight
error bars depending on eval set -- but per §13, agent behaviour is still
near-identical across R/P/A at this training stage, so this is a "the environment's
reset-time cleanup plus deterministic baselines don't obviously help 2-qubit count"
observation, not yet a "the agents learned something regime-specific" one. Worth
double-checking against Riu et al.'s own reported baseline numbers once full training
runs exist, since a baseline that reliably makes things worse on this exact metric
would be a real, citable methodological point either way (an artefact of this
project's baseline substitutions per `docs/deviations_from_paper.md`, or a genuine
property of `full_reduce` on Clifford+T-heavy circuits).

### 17. Cross-checked §16's fix and finding against Riu et al. (2025) directly

**Trigger.** The user supplied the actual paper PDF, closing the open item at the
end of §16.

**Cross-check 1 (validates the `.to_basic_gates()` fix).** Riu et al.'s Table 2
reports, per benchmark circuit, an "Original" (Qubits, Total Gates, 2Q) column --
the exact same quantity as this project's `initial_gates`/`initial_2q`. Transcribed
all 33 rows and joined against `results/logs/evaluation_results.csv`'s post-fix
`initial_gates`/`initial_2q` by circuit name: **33/33 exact matches on 2-qubit gate
count, 32/33 exact matches on total gate count** (the one miss, `tof_3.qasm`: ours
57 vs. paper's 39 total gates, but 2Q matches exactly at 18 -- almost certainly a
trivial single-qubit gate-cancellation cleanup `.to_basic_gates()` doesn't run,
which doesn't touch CNOT count). This confirms the paper's own "Original" circuit
statistics are measured on the Clifford+T-decomposed circuit, exactly matching the
design decision made in §16 (option (a), chosen over leaving native Toffoli gates
in place) -- not just a reasonable choice, but the one the paper itself uses.

**Cross-check 2 (validates the finding, not just the fix).** The "unexpected
finding" flagged at the end of §16 -- that `full_reduce` consistently *increases*
2-qubit gate count -- turns out to be explicitly documented in the paper, not
project-specific: Section 2.4.1 states "it is crucial to understand the trade-off
between removing a spider and altering the connectivity of the diagram, as this may
imply a reduction in the total number of gates but also increase the number of
two-qubit gates (Figure 2b)", and their own Figure 2b (p.7) shows `full-reduce+`
(their basic-opt-postprocessed full_reduce) landing in the "unsuccessful
compression" (red-shaded) region for 2-qubit gate count on 10-qubit Clifford+T
circuits with as few as ~20 initial 2-qubit gates -- the same qualitative behaviour
observed here. **Conclusion: this is a genuine, well-known property of
`full_reduce` on Clifford+T circuits, not an artefact of this project's baseline
substitutions.** Safe to cite directly in Results/Discussion with Riu et al. (2025)
Section 2.4.1 and Figure 2b as corroborating evidence, rather than needing to
re-derive or caveat it as project-specific.

**Also useful for Methodology, from the same paper.** Table 1 (p.12) gives the exact
PPO hyperparameters this project's `Config` defaults were migrated from (matches:
`n_steps=512`, `n_envs=8`, `learning_rate=2e-4`, `n_epochs=8`, `minibatch_size=512`,
`gamma=0.99`, `gae_lambda=0.95`, `vf_coeff=0.5`, `ent_coeff=0.01`, `clip_eps=0.1`).
Section 3.3.2 confirms the two reward targets used in this project (`total_gates`,
`twoqubits`) are exactly the paper's own two training objectives, and states the
paper's own training circuit parameters for the two-qubit-gate objective
specifically: "the agent trains on circuits of 5 qubits and 70 gates, with increased
probability of including CNOT gates to 1/3" -- notably *different* from this
project's `Config.n_gates_random=60` (which matches the paper's *other*,
total-gate-count objective's circuit parameters, not the two-qubit one) and with no
CNOT-probability boost applied here. Worth a methodology note: since §15 confirmed
`twoqubits` is this project's actual training/reward target, `Config.n_gates_random`
and the training generator's gate-probabilities (`p_t=0.17, p_s=0.24, p_hsh=0.25`,
i.e. CNOT gets the remaining ~34%) were **not** re-tuned to match Riu et al.'s
`twoqubits`-specific circuit generation (70 gates, CNOT probability 1/3) -- a
deviation from the paper worth deciding on and recording, not silently carrying
forward.

### 18. Fixed the training-circuit size to match Riu et al.'s twoqubits objective

**Trigger.** §17's finding, surfaced directly rather than left buried in the log
given it's an actionable parameter mismatch.

**Decision (asked, not assumed).** Update `Config.n_gates_random` to match the
paper, rather than leave it at 60 and document the gap.

**Fix.** `src/circopt_adder/config.py`: `n_gates_random` changed from `60` to `70`,
with a comment recording *why* (matches Riu et al. Table 1's `twoqubits`-objective
circuit generation specifically, as distinct from their `total_gates`-objective
setup, which does use 60). This is the only place the literal value lived --
`train_agent_r.py`, `evaluate.py`, and `analyze_random_vs_structured.py` all read
`cfg.n_gates_random` dynamically, so the fix propagates without further code
changes. Updated `analyze_random_vs_structured.py`'s module docstring
(`~60 gates` -> `~70 gates`) since it's forward-looking code documentation, not a
historical record. Full test suite rerun (8/8 passed).

**Deliberately not touched: the gate-type recipe.** The CNOT-inclusion probability
this project's generator already uses (`1 - 0.17 - 0.24 - 0.25 = 0.34`) is close
enough to the paper's stated `1/3 ≈ 0.333` for `twoqubits` training that no change
was needed there -- only circuit size (gate count) was off.

**Not retroactively fixed: docs/random_vs_structured_composition_analysis.md.**
That document reports actual measurements from a specific already-run analysis (at
`n_gates_random=60`); left as an accurate historical record rather than silently
edited to claim numbers that weren't actually measured at 70 gates. A re-run under
the new default would be a deliberate follow-up, not an automatic consequence of
this config change.

**Consequence for existing checkpoints.** `agent_R_random.pt`,
`agent_P_pure_random.pt`, `agent_A_adder.pt` (§12) were trained at the old
`n_gates_random=60` and are now stale relative to `Config`'s current default -- low
cost, since per §13 these were smoke-test/pipeline-validation checkpoints only (19
PPO updates), not a real result to begin with, and full training hasn't started yet.

### 19. Pre-training EDA on the actual training-circuit pipeline

**Trigger.** Requested explicitly: EDA for data understanding before committing to
full training, now that §18 settled the training config.

**Method.** New script `scripts/eda_training_data.py`. 200 simulated
`env.reset()`-equivalent episodes per regime (600 total), using each generator with
the exact parameters `train_agent_{r,p,a}.py` actually pass to it (not the wider
adder bit-width range used for the composition analysis). Recorded, per episode:
graph size after `circuit_to_graphlike` (spider count), feasible LC/pivot action
counts (hence total PPO action-space size at reset), extraction success, and
`headroom_units = reward_metric / max_compression` -- `max_compression` is a fixed
constant (10, `env.py`'s `_normalisation()`), not scaled per circuit, so this ratio
is how much reward-signal magnitude an episode can actually produce.

**Result.** Full writeup: `docs/training_data_eda.md`. Headline findings:
- **0/600 extraction failures** -- the pipeline is reliable across all three
  generators as currently configured; no blocker found before full training.
- **`cuccaro_adder` has exactly `n_lc=1` at every single reset** (std 0.00, all 200
  draws, bit-widths 2/3/5), a structural invariant of the Cuccaro construction after
  Clifford-simplification -- and independently corroborates Riu et al. (2025)
  Section 4.1's finding that LC is not impactful for two-qubit-gate optimisation
  (they found this empirically on random circuits; here it's true by construction
  for Agent A's entire training distribution).
- **`pure_random`'s action-space size (`n_actions`: 1-109) and reward headroom
  (`headroom_units`: 0.1-15.9, a 159x spread) are far more variable than
  `structured_random`'s or `cuccaro_adder`'s** -- a direct, mechanical consequence of
  pairing an unconstrained circuit-size generator with a *fixed* reward-normalisation
  constant. Flagged as a plausible, checkable source of extra training noise
  specific to Agent P, to consider before over-interpreting any R-vs-P-vs-A
  performance gap as purely about the training-data *distribution* rather than
  partly about reward-signal consistency.

**Artifacts.** `results/logs/training_data_eda.csv` (600 rows), `results/figures/
eda_action_space.png`, `eda_reward_headroom.png`, `docs/training_data_eda.md`.

**Status:** no blockers found; full training runs still not yet started.

### 20. Cross-checked the circuit generator against the paper's actual released code, found and fixed a pre-processing bug

**Trigger.** Direct question: "is my circuit generator the same as the paper?" --
answering it properly meant checking the actual source, not just the prose.

**Method.** Fetched `rl-zx/gym-zx/gym_zx/envs/zx_env.py` from
`qilimanjaro-tech/Circopt-RL-ZXCalc` (the repo the paper cites for its own code,
`github.com/qilimanjaro-tech/Circopt-RL-ZXCalc`) directly via the GitHub API/raw
content -- `generators.py`'s own comment claimed to match this file, so verified
against it rather than trusting the comment. Read `reset()` line by line.

**Confirmed matches:** gate-type probabilities (`p_t=0.17, p_s=0.24, p_hsh=0.25`)
are byte-for-byte identical to the paper's actual `zx.generate.cliffordT(...)` call.
Qubit count (5) and gate count (70, per §18's fix) also match Table 1/Section 3.3.2's
`twoqubits`-objective setup.

**Found: a real, significant discrepancy in pre-processing.** The paper's actual
`reset()` applies, to the freshly-generated random circuit, *before the agent acts*:
`basic_optimise` (light peephole gate cleanup) then `zx.simplify.teleport_reduce`
(T-count-only reduction -- Duncan et al. Section 2.4.2, cited by the paper itself,
proves this "does not change... the number or location of the two qubit gates").
This project's `make_random_circuit_generator` and
`make_pure_random_circuit_generator` instead called `zx.simplify.full_reduce` --
which applies the *same* local-complementation and pivoting rules the RL agent is
being trained to discover -- directly on the freshly-generated circuit, before the
environment ever saw it. Net effect: training circuits were being substantially
pre-solved before each episode even started, likely starving the agent of the very
structure it needs to learn from. This is not a hypothetical concern -- §19's EDA
(run before this fix) had already surfaced the symptom without the cause being
identified yet: `structured_random`'s `n_lc` mean was an implausibly low 2.58.

**Decision (asked, not assumed).** Fix both `make_random_circuit_generator` (must
match the paper) and `make_pure_random_circuit_generator` (not paper-derived, but
same underlying environment-design principle applies -- an agent shouldn't be handed
pre-solved circuits either way) -- chosen over fixing only Agent R's generator or
leaving both as a documented deviation.

**Fix.** New shared helper `_light_preprocess(g)` in `generators.py`: `Circuit.
from_graph(g)` -> `zx.basic_optimization(c.to_basic_gates())` -> back to a graph ->
`zx.simplify.teleport_reduce` -> `zx.simplify.to_graph_like` (structural
normalisation only, required for extraction, not a further reduction) ->
`zx.extract_circuit`. Both generators now call this instead of `full_reduce`.
`make_adder_generator`/`ripple_carry_adder` needed no change -- it builds gates
directly and never called `full_reduce`.

**Verified.** Full test suite reran clean (8/8). Re-ran `scripts/eda_training_data.py`
(§19) after the fix -- full comparison and updated numbers in
`docs/training_data_eda.md`. Headline: `structured_random`'s `n_lc` mean rose from
2.58 to **9.59** (range 2-19, up from 0-7); `n_actions` mean rose from 20.4 to 28.0.
`pure_random`'s `headroom_units` spread narrowed from 159x (0.1-15.9) to a much
more reasonable 0.0-6.3, now broadly comparable to the other two regimes rather than
an outlier by an order of magnitude. `cuccaro_adder` numbers are byte-identical
before/after, as expected (its generator was never affected).

**Also worth noting for Methodology:** the fixed generators no longer artificially
inflate `initial_2q` the way the old `full_reduce`-first version did -- `reward_metric`
(2-qubit gate count at reset) fell for both affected regimes (structured_random
~29->23, pure_random ~49->24 mean), consistent with §16/17's independent finding
(corroborated by the paper's own Figure 2b) that `full_reduce` tends to *increase*
2-qubit gate count rather than reduce it. The old generator's upfront `full_reduce`
call was quietly inflating what got recorded as each episode's untouched starting
point.

**Status:** this was caught before any full training run, at the cost of the three
existing smoke-test checkpoints (§12) being further superseded -- already flagged
as pipeline-validation-only, so no real loss. Full training still not started;
config and generators are now believed to genuinely match the paper for Agent R.

### 21. Gate-box -> ZX-diagram -> graph-like visualization, custom renderer replaced with Qiskit

**Goal.** Show one representative circuit from each training regime
(`cuccaro_adder`, `structured_random`, `pure_random`) through the full pipeline
`env.py`'s `reset()` performs: textbook gate-box circuit -> ZX-diagram
(`circuit.to_graph()`) -> graph-like ZX-diagram (`circuit_to_graphlike()`). pyzx has
no gate-box circuit renderer of its own (it's a ZX-calculus library, not a general
circuit-drawing one; `zx.draw()` on a `Circuit` already converts to its ZX-diagram),
so this needed new drawing code, in `scripts/visualize_graph_transform.py`.

**First pass: a small custom matplotlib gate-box drawer.** Covered the gate
vocabulary these circuits use (H, S, T/T-dagger and other `ZPhase`, CNOT, CZ, SWAP,
Toffoli, CCZ) by hand, reading each gate's `.target`/`.control`/`.ctrl1`/`.ctrl2`
attributes directly. Two bugs caught and fixed by inspecting zoomed crops of the
actual output rather than trusting it at a glance:
- Gate boxes rendered as squished vertical slivers, not squares -- the axes had no
  fixed aspect ratio, so a "square" box drawn in data units came out stretched by
  whatever aspect ratio the figure happened to have (very elongated for a
  many-gates/few-qubits circuit like the adder). Fixed with `ax.set_aspect("equal")`
  plus a figsize computed at one consistent inches-per-data-unit scale on both axes.
- The 3-column composite (gate-box / before / after) wasted huge whitespace because
  all three columns shared one column width, and gate-box diagrams are drawn far
  wider per-gate than ZX-diagrams. Fixed by giving each column its own width
  (`assemble_grid`, replacing the earlier `build_panel`).

**Second pass: replaced the custom drawer with Qiskit.** Asked directly ("can't you
install qiskit to make it easier") rather than continuing to harden the hand-rolled
version. Installed `qiskit` + `pylatexenc` (the latter is `qiskit`'s own mpl-drawer
dependency, not pulled in automatically). Pipeline: `circuit.to_qasm()` (pyzx) ->
regex relabel of generic `rz(theta)` gates back to named QASM gates where the phase
matches a standard one-letter gate (`t`/`tdg`/`s`/`sdg`/`z` -- pyzx normalises
phases to `[0, 2*pi)` when emitting QASM, e.g. T-dagger comes out as `rz(1.75*pi)`,
not `rz(-0.25*pi)`, which the first version of this substitution got wrong and had
to be corrected) -> `qiskit.qasm2.loads()` -> `QuantumCircuit.draw("mpl", fold=-1)`.
Kept `qiskit`/`pylatexenc` as an optional extra (`pyproject.toml`'s new
`[project.optional-dependencies] viz`, plus a commented-out note in
`requirements.txt`) rather than a core dependency, since nothing else in this
project needs it -- consistent with the project's existing minimal-dependency
discipline (see e.g. the choice not to add `pandoc`/`python-docx` for the earlier
LaTeX-template conversion, §14).

**Result.** Publication-quality gate-box diagrams (proper H/T/T†/S/S†/Z boxes,
correct CNOT/Toffoli control-dot notation) replacing the custom renderer, at the
cost of one new optional dependency. Verified correct via zoomed crops on all three
circuit types, including the adder's Toffoli decomposition (H, CNOT, T†, CNOT, T,
CNOT, T†, CNOT, T, T, H sequence, matching `generators._add_toffoli` exactly) and
confirming un-named phases (3π/4, 5π/4) correctly remain as generic `Rz` boxes
rather than being forced into a named label that doesn't exist.

**Artifacts.** `scripts/visualize_graph_transform.py` (rewritten), `pyproject.toml`
(`viz` extra), `requirements.txt` (commented note), `results/figures/
graph_transform_before_after.png`.

### 22. Concrete single-circuit demonstration: full_reduce vs. the RL's actual pre-processing

**Goal.** Make §16/17/20's finding (full_reduce can increase 2-qubit gate count;
this project's training pipeline deliberately avoids it) concrete on one circuit,
rather than only as aggregate statistics.

**Method.** New script `scripts/demo_full_reduce_adder.py`. Took
`ripple_carry_adder(2)` (5 qubits, smallest/most legible) through three states:
as-generated; after `zx.simplify.full_reduce`; after `generators._light_preprocess`
(the RL's actual pre-processing: `basic_optimization` + `teleport_reduce`) applied
to the same starting circuit. Rendered gate-box diagrams for all three via
`visualize_graph_transform.render_gate_box` (§21).

**Bug caught and fixed.** `full_reduce`'s output on this circuit contains a `SWAP`
gate that Qiskit's `qasm2` parser rejects (`'swap' is not defined in this scope`),
even though `swap` is standard OpenQASM 2.0 -- apparently a gap in Qiskit's built-in
`qelib1.inc` handling rather than anything wrong with pyzx's output. Fixed in
`render_gate_box` itself (so both this script and §21's benefit) by calling
`circuit.to_basic_gates()` before QASM export, which decomposes `SWAP` into 3 CNOTs
and sidesteps the parser gap entirely.

**Result.** On this one circuit: `full_reduce` → gates 68→100 (+47.1%), **2-qubit
32→40 (+25.0%, increase)**, T 28→16 (−42.9%). RL pre-processing → gates
68→94 (+38.2%), **2-qubit 32→22 (−31.2%, decrease)**, T 28→20 (−28.6%). Opposite
sign on the exact metric (`reward_mode="twoqubits"`) the agent is trained on, on the
same circuit -- as clean a single-example confirmation of the §20 fix's rationale as
this project has produced. Note explicitly caveated in the script: `ripple_carry_adder`
circuits are never actually run through `_light_preprocess` in real training (only
`structured_random`/`pure_random` are); this comparison is illustrative, not a
description of Agent A's actual training data.

**Artifacts.** `scripts/demo_full_reduce_adder.py`, `results/figures/
full_reduce_adder_demo.png`.

### 23. Incremental checkpointing and logging for full training runs

**Trigger.** Answering "how will results be recorded" for the not-yet-started full
training runs surfaced a real risk worth fixing before launching them: `ppo.py`'s
`train()` buffered every update's log row in memory and saved the checkpoint only
once, after the entire loop finished. A full run is `total_timesteps=2,000,000 /
n_steps=512 ≈ 3,906 updates`, likely 8-16 hours unattended on this CPU-only
machine -- a crash, OOM, laptop sleep, or accidental interrupt at update 3,900 of
3,906 would have lost the training log *and* the trained policy entirely, with
nothing to show for the run or resume from.

**Fix.** `src/circopt_adder/ppo.py`'s `train()`: the log CSV header is written up
front and each update's row is appended immediately (`pd.DataFrame([row]).to_csv(...,
mode="a")`) instead of being buffered and written once at the end. The checkpoint is
now saved every `cfg.checkpoint_interval` updates, plus unconditionally on the final
update (so behaviour for short/smoke-test runs, where `n_updates` is well under the
interval, is unchanged -- one save, at the end, same as before). New `Config` field:
`checkpoint_interval: int = 100`.

**Verified.** A tiny dry run (`total_timesteps=2048` → 4 updates,
`checkpoint_interval=2`) confirmed: 4 rows appended to the CSV matching the 4
returned in-memory, correct columns, and a valid, non-trivial checkpoint file on
disk. Full test suite still passes (8/8). Scope was deliberately kept to
*durability* (nothing is lost on interruption) rather than full resume-from-checkpoint
training continuation (reloading optimizer state, RNG state, etc.) -- that would be a
separate, larger feature, not requested here.

**Status:** full training runs still not started; this was the last identified gap
before doing so.

---

### 24. Pilot run, full-training launch, and Agent A's collapsed-policy investigation

**Goal.** Run the pilot (200k timesteps, Agent R) to confirm the training pipeline
learns something real before committing to the full 2,000,000-timestep budget across
all three agents, then launch and monitor the full runs.

**Pilot result.** Reached update 338/390 (~87%) before its background process was
lost with no error trace (see "process reliability" below) -- but by then the trend
was already conclusive: reward improved from a noisy -0.44 to -0.69 in the first ~95
updates to a stable, repeatedly-new-best band of -0.04 to -0.09 from update ~180
onward (best -0.042 at update 325), entropy fell from ~2.7 to a low of 0.166 around
update 275, and the extraction failure rate stayed at 0.0% for the entire run. Judged
sufficient go/no-go evidence; proceeded to full training rather than re-running the
pilot to completion.

**Full training launch.** All three agents (`train_agent_r/p/a.py`, default
`total_timesteps=2,000,000`, ~3,906 updates each) launched in parallel as detached
background processes (`nohup ... & disown`), not tied to the interactive session, on
the 10-core CPU-only machine (no CUDA; MPS exists but wasn't used since the training
loop is per-transition, not batched -- see ppo.py's own docstring on this -- so GPU
acceleration would mostly add transfer overhead, not help). Progress tracked via a
low-frequency (20-min) polling script reading the per-update CSV logs directly,
rather than reacting to individual PPO updates as the pilot's monitoring did --
appropriate for one ~13-hour run, not three in parallel.

**Process reliability, general.** Both the pilot process and (twice) Agent A's
process were separately lost mid-run with no traceback, no OOM evidence in `vm_stat`/
`swapusage`, and no macOS crash report -- while Agent R and Agent P ran continuously
for 2.5+ hours without incident in the same parallel batch. Root cause not
conclusively identified; not pursued further for R/P since they're unaffected, but
noted as an open reliability gap (`ppo.py`'s `train()` still has no
resume-from-checkpoint path, so any future process loss discards all progress since
the last save).

**Agent A: collapsed-policy investigation (bug found and fixed; underlying problem
not solved).** Around update 150 (~4% of the full run), Agent A's reward flatlined at
exactly 0.0 with entropy collapsing to ~0.0008 (from a healthy ~1.2 and improving,
matching Agent R's trajectory shape up to that point) -- while Agent R and P showed
no such pattern at the same stage. Four things were tried, in order, each verified
empirically rather than assumed:

1. **Exception-handling bug (real, fixed, but not the cause).** `env.step()` had no
   `try/except` around applying an LC/pivot action (only around extraction) --
   reproduced a concrete `KeyError` crash from `apply_pivot` on a stale vertex
   reference. Fixed by wrapping the action-application + extraction in `env.py` in a
   `try/except`, converting any failure into the existing extraction-failure penalty
   (-1.0, terminated) instead of an uncaught process-killing exception. Verified with
   a 30-episode random-policy stress test (no crashes). *However*: re-running Agent A
   with the fix reproduced the exact same collapse, bit-for-bit identical to the
   original run's logged values (same fixed seed) -- proving the crash this fix
   guards against was never actually triggered on this trajectory, so it wasn't the
   cause. Kept anyway as a legitimate defensive fix.
2. **Baseline-computation fix (tested numerically, does nothing).** Hypothesis: since
   raw ripple-carry adders' `two_qubit_gate_count` is never beaten by either
   deterministic baseline (`clifford_simp`, `teleport_reduce`) computed in `reset()`
   (confirmed directly: baseline == initial metric for bits=2/3/5), immediate STOP is
   a free, zero-variance action with no incentive to explore past it. Tried adding
   `basic_optimization` before graph conversion (matching `_light_preprocess`'s
   order) to strengthen the baselines -- numerically confirmed this produces *zero*
   change (baseline still == initial for all three bit-widths tested), so not
   implemented in `env.py`. The earlier `docs/project_log.md` entry 22's "-31.2%"
   full_reduce-vs-light_preprocess figure, re-examined here, turns out to compare
   native (undecomposed) Toffoli gate counts against a fully CNOT-decomposed circuit
   -- not an apples-to-apples comparison, so it wasn't real evidence of exploitable
   structure either.
3. **Gadget-fusion (Riu et al. Eq. 8) implemented and verified, but inert on these
   circuits.** `zx_utils.py`'s scope note had flagged pivot-boundary, pivot-gadget,
   and gadget-fusion as missing non-Clifford rules. Implemented gadget-fusion
   (`find_feasible_gadget_fusions`, `apply_gadget_fusion`): two phase gadgets acting
   on the same target set combine their leaf phases additively and the second
   axel/leaf are removed. Verified correct via hand-built test graphs and
   `zx.compare_tensors` (including an explicit T + T* -> phase-0 cancellation check,
   tensors equal up to global phase in all cases) before wiring into `env.py`'s
   action space (uses previously-reserved-but-unused node/edge feature slots --
   `actor_node_feat_dim`'s "5 action-type flags" and `edge_feat_dim`'s 6th channel --
   so no model dimension changes needed). *However*: raw ripple-carry adders' 24
   non-Clifford spiders are all in-line (degree 2-6), not gadget-leaf form (degree 1),
   so gadget-fusion never has candidates to act on. Even after gadgetizing with
   pyzx's own `simplify.gadgetize()`, zero fusable pairs were found on any tested
   bit-width -- these T/T* gates simply don't share target sets in a freshly-
   decomposed Toffoli chain. Kept as a real, verified capability for future data
   (and for R/P, whose Clifford+T circuits may have this structure) but doesn't
   affect Agent A's current collapse.
4. **Switched Agent A to `_light_preprocess`'d adders (matching R/P), tested
   empirically.** Static check first: even on the preprocessed adder, no single
   LC/pivot move improves on the (now lower) initial metric either -- but this
   doesn't rule out a multi-step improving *sequence*, which is exactly the kind of
   thing one-step lookahead can't detect and RL is supposed to find. Implemented
   (`make_adder_generator` now runs `_light_preprocess` on the generated adder,
   matching R/P), stress-tested, relaunched. Result: the collapse recurred anyway
   (reward 0.0, entropy -> ~0.0005 by update 186), and the process died silently
   again at roughly the same point. Kept the preprocessing change (adders now match
   R/P's pipeline) since it's not harmful and is more consistent, but it did not fix
   the collapse.

**Decision.** After four investigated angles without a fix, and given the compounding
unexplained process-death issue, paused Agent A rather than continuing to burn
compute on repeated collapsed runs. Agent R and Agent P continue their full runs
unaffected (separate processes, already running 2.5+ hours with no issues at time of
decision). Revisit Agent A once R/P finish, or once there's time to investigate the
process-death issue directly (e.g. running attached/foreground rather than
backgrounded, to catch whatever kills it in real time) -- both the reward-collapse
and the process-reliability question are still open.

**Result, for methodology purposes.** Whatever the eventual fix turns out to be, the
current finding stands on its own as informative: under the Clifford-only-plus-
gadget-fusion action space and the environment's fixed-baseline reward, raw and
`_light_preprocess`'d ripple-carry adders alike offer essentially no single-step
improving move and the deterministic baselines never beat the starting circuit --
unlike Agent R's Clifford+T circuits, which do have exploitable Clifford-only
structure (R converged cleanly to a stable -0.04 to -0.09 reward band). This is a
substantive, structural difference between the training regimes, not a training bug,
and worth reporting as such regardless of how Agent A is eventually resolved.

**Artifacts:** `src/circopt_adder/env.py` (exception handling around action
application; gadget-fusion wired into the action space), `src/circopt_adder/
zx_utils.py` (`find_feasible_gadget_fusions`, `apply_gadget_fusion`,
`_gadget_axel_and_targets`), `src/circopt_adder/generators.py`
(`make_adder_generator` now applies `_light_preprocess`), `results/logs/
agent_R_random_train_log.csv`, `agent_P_pure_random_train_log.csv` (in progress),
`agent_A_adder_train_log.csv` (two collapsed attempts, paused).

---

### 25. Agent A collapse: what the trained checkpoint actually does, and a scoped
    bits=2 retry with a search-space justification

**Trained-checkpoint demo.** `scripts/demo_agent_a_trained.py` loads Agent A's only
saved checkpoint (update 99 of the second, `_light_preprocess`'d-adders run -- the
last save before that run's own collapse) and applies it greedily (argmax, not
sampled) to a fresh `ripple_carry_adder(2)` episode, printing the full step-by-step
trajectory rather than just before/after numbers. Result: the policy took exactly one
action (a local complementation) that made the two-qubit count *worse* (22 -> 23,
penalized -0.1), correctly didn't keep it as the best-seen circuit, and then stopped
-- ending with the same 2-qubit/T count it started with. (The printed "gates" column
in that script's table drops from 94 to 54 independent of any agent action -- verified
directly by re-running `circuit_to_graphlike` + `extract_and_cleanup` with zero
actions taken and getting the same 54; this is a re-extraction artifact of
`basic_optimization` finding a smaller gate decomposition for the same 2-qubit/T
structure, not a result to report.) Figure: `results/figures/
agent_a_trained_adder_demo.png`.

**Root-cause refinement via exhaustive multi-step search.** Extended the earlier
one-step-lookahead finding (project log entry 24, point 4: no single move improves
the metric at any tested bit-width) to two-step lookahead, exhaustively, on the
`_light_preprocess`'d adder at each training bit-width:

| bits | initial (2q) | step-1 moves | 2-step combos | best 2-step result |
|---|---|---|---|---|
| 2 | 22 | 13 | 44 | 20 (-9.1%) |
| 3 | 33 | 21 | 135 | 33 (+0.0%, nothing found) |
| 5 | 55 | 37 | 285 | 53 (-3.6%) |

> **Correction (entry 33):** the "20" (bits=2) and "53" (bits=5) results in this
> table could not be reproduced. An independent re-run of this same exhaustive
> 2-step search, after the multi-step chaining bug (entry 26/27) was actually
> fixed, found 13/21/37 step-1 moves for bits=2/3/5 respectively -- matching this
> table exactly -- but 124/368/1240 total 2-step combinations, not 44/135/285,
> and **no improving 2-step sequence for any of the three bit-widths** (best
> result = initial value in every case). The bits=3 "nothing found" row is the
> only one of the three that reproduces. Root cause not identified -- this
> table's search was run ad hoc and not saved as a script, so there is nothing
> to diff against. See entry 33 for the full re-verification, including a
> cross-check under the gflow condition (also found nothing) and the
> corresponding fix to the dissertation text, which had stated the bits=2
> "22->20" result as established fact.

This is decisive evidence that the earlier "no exploitable structure" conclusion
(entry 24) was too strong: improving sequences *do* exist (bits=2, bits=5), they're
just one reward-neutral-or-negative step away from the payoff -- exactly the kind of
local reward-landscape valley that makes STOP (a free, always-available, zero-risk
action) attractive enough for PPO's constant entropy bonus to lose the fight against,
especially as the search space balloons with bit-width (13->21->37 first-step moves).
bits=2's 44 total two-step combinations is small enough that random exploration
should plausibly find the -9.1% sequence early, before the collapse dynamic sets in.

**Decision.** Scoped retry, not a permanent redefinition of Agent A:
`scripts/train_agent_a_bits2.py`, `adder_min_bits=adder_max_bits=2`,
`ent_coef` raised from 0.01 to 0.05 (a complementary safety net -- even in this
smaller search space, a run of early bad-luck rollouts could still push the policy
toward STOP before it stumbles onto the improving sequence). Distinct run_name
(`agent_A_bits2`) so the two collapsed full-range runs' logs/checkpoints are
preserved as evidence, not overwritten. Smoke-tested (7 updates, clean) before
launching the full 2,000,000-timestep run alongside R and P.

**Crash diagnostics added to `ppo.py`'s `train()`,** after two silent,
traceback-less process deaths (the pilot run, and Agent A twice) with no OOM
evidence in `vm_stat`/`swapusage` and no macOS crash report found:
`faulthandler.enable()` (dumps a Python traceback on a fatal signal -- SIGSEGV/
SIGABRT/SIGBUS/SIGFPE, e.g. a C-level crash inside numpy/torch/pyzx -- which a plain
`try/except` cannot catch), `faulthandler.register(signal.SIGTERM, ...)` (dumps on an
external kill request specifically), a `try/except Exception` around the update loop
writing any genuine Python exception's traceback to `{run_name}_crash.txt`, and a
heartbeat file (`{run_name}_heartbeat.txt`) written at the start and end of every
single update -- the one thing that survives even an uncatchable SIGKILL, since it at
least pins down exactly which update the process was on when it died. None of this
changes training behaviour; it's diagnostic-only, and benefits R/P too on any future
relaunch (already-running R/P processes are unaffected since they loaded the old
`ppo.py` into memory before this edit).

**Status:** `agent_A_bits2` running alongside R and P; whether it escapes the
collapse (or dies again, now hopefully with a traceable cause) is the open question.

**Outcome (same session, ~40 min later).** Collapsed again -- faster this time.
Reward and entropy declined smoothly and monotonically together from update ~17
(reward -0.19, entropy ~1.9) to update ~65 (reward 0.0, entropy 0.032), roughly half
the ~150-update timescale of the two full-range collapses, despite the smaller
search space and the 5x entropy bonus both being intended to slow this down, not
speed it up. Notably the smooth trajectory shape does *not* match a policy that
found and is exploiting the known-good 22->20 two-step sequence (which should yield
a net *positive* episode reward via the terminal bonus, not exactly 0) -- it matches
the same always-STOP equilibrium as before. `value_loss` collapses toward zero in
the same rows as reward and entropy, consistent with the value function converging
to predict ~0 everywhere, which would flatten the advantage signal on the rare
rollouts that do sample the good sequence and remove the gradient pressure needed to
reinforce it -- a plausible mechanism, not confirmed.

This is the fifth distinct intervention (exception handling, baseline computation,
gadget-fusion, light_preprocess switch, bits=2 + higher ent_coef) to fail against the
same collapse. Paused rather than attempting a sixth without further discussion --
see next entry.

---

## How to keep this log going

Append a dated entry each time a new piece of work is done, using the same
Goal / Method / Decisions+rationale / Result / Artifacts shape used above -- that
shape maps directly onto dissertation Methodology (Goal, Method, Decisions) and
Results (Result, with the exact numbers already pulled out rather than needing to be
re-derived from raw CSVs later).

---

### 26. Root-caused the collapse against the paper's own text, and implemented (a
    scoped version of) the missing piece

**Trigger.** User re-shared the full Riu et al. (2025) PDF (previously only
partially cross-referenced -- see entries 17-18) while asking how this project's
evaluation results compared to the paper's. Full-text reading of Section 3.3.2
surfaced something entries 24-25's investigation had missed entirely.

**The explanation.** Riu et al.'s terminal reward bonus compares the agent's
result against `cflow-zx` -- "the most competitive ZX-calculus based optimizer for
two-qubit gate reduction" -- not the weaker `clifford_simp`/`teleport_reduce`
baselines this project's `env.py` used (already flagged as a substitution in
`docs/deviations_from_paper.md`, but without the causal story). Their own text:
"We find that, without this reward, controlling the balance between exploration
and exploitation with only the entropy parameter is extremely hard, and the agent
tends to rapidly converge to a policy of not applying actions to the circuit."
That is Agent A's collapse, stated in advance, with the mechanism named
explicitly. `clifford_simp`/`teleport_reduce` essentially never beat the starting
circuit (confirmed empirically across entries 24-25), so the "beat the baseline"
bonus was trivially satisfied by doing nothing -- no amount of entropy tuning
fixes a reward that's structurally satisfied by inaction.

**Quantitative comparison against Table 2** (matched by circuit name, best of
Agent R/P's `final_2q`): paper's `rl-zx + cflow-zx` mean reduction = 24.6%, this
project's agents = 2.4%, with many circuits (barenco-tof, tof, gf2-mult family) at
exactly 0%. Caveat: not apples-to-apples -- their number is from a hybrid
gflow-RL-plus-cflow-zx-postprocessing workflow with 1000 stochastic attempts per
circuit, not a pure RL policy evaluated with 10 samples. But directionally
consistent with the terminal-bonus explanation: weak baseline -> weak training
signal -> policies that barely move past "do nothing" -> both Agent A's collapse
and R/P's comparatively modest real-world transfer.

**Implementation.** `cflow-zx` itself isn't published in enough detail to
reimplement from Riu et al. alone; it's their citation to Holker (2024) "Causal
flow preserving optimisation of quantum circuits in the zx-calculus"
(arXiv:2312.02793), which in turn relies on Mhalla & Perdrix (2008) "Finding
Optimal Flows Efficiently" (arXiv:0709.2670) for the actual causal-flow-finding
algorithm. Both fetched in full (`WebFetch` returns unparsed binary for arXiv
PDFs; `pypdf`, installed via `pip3 install pypdf` since `brew install poppler`
failed on a blocked CDN, extracted clean text from the saved PDF instead) and
read in full rather than reimplementing from the summary alone.

- `src/circopt_adder/causal_flow.py` (new): `causal_flow()` implements
  Mhalla & Perdrix Algorithm 1 exactly (only the flow-existence/successor-function
  part; the labelling/partial-order it also computes isn't needed here). Verified
  against two hand-worked graphs with known ground truth (a 3-vertex path -- flow
  exists, N2Q=0; a 4-cycle -- no causal flow) before trusting it on anything real.
  `n2q_via_causal_flow()` implements Holker's Lemma 17 (`N2Q = |E| - |V| + |I|`,
  computable directly from the graph with no circuit extraction) -- cross-checked
  against real `zx.extract_circuit` on 15 fresh random circuits: 15/15 exact
  matches once compared against *raw* extraction (matching against
  `extract_and_cleanup`'s output, which also runs `basic_optimization`
  afterwards, gave 8/15 -- that post-hoc gate-level cleanup can change the count
  Lemma 17 doesn't and isn't a bug in the formula).
- `src/circopt_adder/baselines.py`: `baseline_cflow_opt()`, a greedy optimizer
  using this scoring function plus the project's existing `apply_local_complementation`/
  `apply_pivot`. **Scoped down from the paper's actual algorithm partway through
  implementation**, after two rounds of real bugs surfaced when chaining multiple
  moves together: (1) reapplying a move to the live graph after having already
  scored it on a copy sometimes produced a *worse* result than the copy predicted
  (fixed by never reapplying -- advance to the already-scored copy directly);
  (2) that fix immediately exposed a deeper issue -- pyzx's own `extract_circuit`
  sometimes failed outright (`"No extractable vertex found"`) on a graph the code
  claimed had a valid causal flow, after several chained moves. A candidate cause
  was identified (when two correctors compete for the same vertex in one round of
  `causal_flow()`, the current implementation discards both from future rounds
  instead of just the winner) but not confirmed as *the* explanation before time
  was called on this investigation. **User's explicit decision: pause the
  multi-step version, ship single-step only.** `baseline_cflow_opt()` now scores
  every feasible LC/pivot move once and applies at most the single best one --
  the same code path already verified safe as part of the (abandoned) chaining
  loop's per-candidate scoring, so no new correctness risk. Verified across 40
  random-circuit trials: 0 regressions, 2 strict improvements, rest either no
  move helped or a found improvement was fully absorbed by the trailing
  `basic_optimization` cleanup (never *worse* than the unmodified input, which is
  the only guarantee actually needed for its use as a baseline).
- `src/circopt_adder/env.py`: wired in as a third baseline candidate in
  `reset()`/`step()`'s terminal-bonus computation (`self.cflow_gates`, alongside
  the existing `korb_gates`/`pyzx_gates`), gated to `cfg.reward_mode == "twoqubits"`
  (Lemma 17 only computes the two-qubit count, not T-count or total gates).
  Smoke-tested over 30 full episodes (random + adder generators, 0 crashes):
  helped in 2/15 random-circuit episodes, 0/15 adder episodes.

**Known limitation, stated plainly.** The single-step restriction means this
does *not* fix Agent A's collapse -- entry 25 already established that adders
need a *2-step* sequence to find any improvement at all (22->20 two-qubit gates,
found by exhaustive 2-step search), and a single-step optimizer structurally
cannot find that. It also doesn't replicate the paper's actual `cflow-zx`
(neighbour-unfusion is entirely unimplemented, which their own Table 1 shows
contributes roughly a third of `flow-opt`'s total gain, 14.27%->19.79%). What
this genuinely is: a real, verified, safe, non-trivial baseline -- strictly
better than doing nothing, unlike the two baselines it sits alongside for these
training regimes -- available for R/P's terminal bonus and any future,
better-resourced attempt at the full chained version.

**Status:** merged, tested, not yet training-tested (no new full run launched
against it this session). The multi-step chaining bug remains open; the
candidate cause (corrector-collision handling in `causal_flow()`) is written
down here for whoever picks this back up.

**Artifacts:** `src/circopt_adder/causal_flow.py` (new),
`src/circopt_adder/baselines.py` (`baseline_cflow_opt`), `src/circopt_adder/env.py`
(`self.cflow_gates`, updated terminal-bonus `baseline = min(...)`).

---

### 27. Checked the paper's own released code for a reference `cflow-zx` implementation

**Trigger.** User pointed at the paper's training repo,
`https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc`, as a possible way to
resolve entry 26's open multi-step chaining bug (`extract_circuit` raising
`"No extractable vertex found"` after several chained LC/pivot moves, despite
`causal_flow()` reporting a valid flow throughout).

**What the training repo actually contains.** Fetched
`rl-zx/gym-zx/gym_zx/envs/zx_env.py` (1168 lines) in full via the GitHub REST
API (no `gh` CLI available; used `curl` against `api.github.com` and
`raw.githubusercontent.com` instead).

> **Correction (entry 32):** the claim originally written here -- that this
> environment does not compute `cflow-zx` live, and instead reads a
> precomputed `circuit_data` dict -- was wrong, caught during a user fact-check
> of a different but related question. `new_gates_cflow = circuit_data[self.gate_type]`
> reads a *local* variable computed two lines above it, straight from a live
> `zx.flow_2Q_simp(g)` call inside `step()`'s terminal block (preceded by
> `teleport_reduce` and `to_graph_like` on the same graph) -- i.e. the
> environment *does* call the real algorithm live, at the end of every
> episode, to compute the terminal-bonus target. The
> `# from pyzx.gflow import gflow` commented-out import is real but is just
> unrelated dead code from an earlier iteration, not evidence of caching --
> conflating the two was the actual error. See entry 32 for the full
> re-verification. The rest of this entry (the fork investigation, the
> corrector-collision finding, the `g.replace()` lead) is unaffected by this
> correction.

The actual `cflow-zx` algorithm implementation itself isn't in this
training repo, regardless -- it lives in the author's own pyzx fork, linked
in the paper's footnote 1: `github.com/calumholker/pyzx` (confirmed via its
GitHub description, which cites Holker's own arXiv ID).

**What the fork actually contains, and what it settles.** Fetched three files
from `calumholker/pyzx` directly (`pyzx/flow.py`, `pyzx/heuristics.py`,
`pyzx/simplify.py`) and read the real implementation:

- `flow.py`'s `cflow()` is essentially the same Mhalla-Perdrix core loop as this
  project's `causal_flow()` -- including the exact "corrector collision"
  behaviour entry 26 flagged as a *candidate* bug (two correctors racing for the
  same external neighbour in one round; the later one silently overwrites the
  successor assignment, and both get consumed from future rounds regardless of
  which one "won"). Holker's reference implementation does the same thing. **This
  rules out that candidate cause** -- it's how Mhalla & Perdrix's Algorithm 1
  actually works, not a bug introduced here.
- The real `cflow-zx`/`flow-opt` greedy loop lives in `simplify.py`, as
  `flow_2Q_simp()` calling a generic driver `selective_simp()`. Its structure:
  score every candidate match once up front, then loop -- each iteration takes
  the highest-scoring remaining match, applies it to a **clone** of the graph
  (`check_g = g.clone()`), checks a `condition(check_g, match)` (for cflow mode:
  `cflow(check_g) is not None`), and only on success calls `g.replace(check_g)`
  (an in-place structural replace of `g`'s own contents, not a Python reference
  reassignment) before incrementally updating the match-score dict for the
  affected neighbourhood and continuing. `extract_circuit` is never called
  mid-loop -- only once, at the very end, same as this project's design.
  Neighbour-unfusion (the piece already known to be unimplemented here, entry 26)
  is handled via a `max_lc_unfusions`/`max_p_unfusions`-bounded search baked into
  the match-generation step, not as a separate move type.

**Reading on the open bug.** The `g.replace()`-based in-place commit (vs. this
project's plain-reference `g = best_graph` in the abandoned chaining version) is
the one structural difference worth flagging for whoever resumes this: it's
possible pyzx's `extract_circuit` relies on some internal graph-state
bookkeeping (row/qubit metadata, vertex ordering, or similar) that `.replace()`
preserves correctly across repeated `.clone()`/mutate/replace cycles but that a
sequence of plain `.copy()` calls does not. Not confirmed -- this is a
concrete, testable lead for a future attempt, not a fix applied now.

**Decision.** No code changed this entry. The single-step `baseline_cflow_opt`
from entry 26 stands as-is, per the user's earlier explicit "pause here, use
single-step version only" -- this was a research check on a pointer the user
gave, not a re-opening of that paused work. Logged so the `g.replace()` lead
isn't lost if the multi-step version gets picked back up later.

**Artifacts:** none (research only); reference files saved to this session's
scratchpad (`holker_flow.py`, `holker_simplify.py`, `holker_heuristics.py`,
`holker_optimize.py`), not part of the repo.

---

### 28. Drafted the Results and Analysis dissertation chapters

**Trigger.** User decided against retraining R/P against the new (single-step)
cflow baseline for now -- the smoke-tested effect size (2/15 random episodes
helped, 0/15 adders) didn't justify another multi-hour run against dissertation
time -- and asked to move on to writing up instead.

**What was written.** `Writing/latex/chapters/chapter04_results.tex` and
`chapter05_analysis.tex`, replacing their placeholder content, using this
project's actual completed-training data (`results/logs/*_train_log.csv`) and
the best-of-10 evaluation run (`results/logs/evaluation_results.csv`, 47
circuits x 5 methods) already sitting in the repo from earlier entries. All
numbers quoted in both chapters were recomputed directly from the CSVs in this
session rather than carried over from earlier ad hoc `python3 -c` output, so
they're independently reproducible. Five figures copied into a new
`Writing/latex/figures/` directory (kept separate from the trailing-space
`Dissertation ` path for cleaner `\includegraphics` references):
`agent_r_p_training_curves.png`, `agent_a_training_curves.png`,
`agent_a_trained_adder_demo.png`, `comparison_barplot.png`, `adder_scaling.png`.
Two new `references.bib` entries (`holker2024causal`, `mhalla2008finding`),
needed to cite the `cflow-zx`/causal-flow work directly in the Analysis
chapter's Agent A discussion.

**Results chapter:** reports training outcomes (R/P completed cleanly, ~16h
wall-clock each; A collapsed under all 3 configurations, in every case within
2-5% of Agent R's training budget), the main 2Q-reduction table by
method x evaluation family, the held-out adder detail table, and the
recomputed comparison against Riu et al.'s Table 2 (2.14% vs their 24.6%, on
the identical 33-circuit benchmark set) -- stated as fact, not yet interpreted.

**Analysis chapter:** ties together findings that were previously scattered
across entries 16-17 (deterministic baselines increase 2Q count -- corroborated
by Riu et al. Section 2.4.1/Fig 2b, not project-specific) and 26 (Agent A's
collapse root-caused to the paper's own stated weak-baseline failure mode) into
a single argument: the real-world-benchmark gap to Riu et al.'s reported 24.6%
and Agent A's outright collapse are the same underlying cause at different
severities, not two separate problems. Also flags, as a new finding surfaced
while writing this chapter rather than something already known: the held-out
adder set's exact numerical agreement across all three agents (including
Agent A) is better explained by 9-sample stochastic search saturating a small,
repetitive action space than by genuine learned generalization -- a caution
against reading that flat 31.3% line as a stronger result than it is.

**Status:** both chapters are prose-complete against currently available data.
Not done: Methodology chapter's Environment/Reward Function/Network
Architecture/PPO Hyperparameters sections (only Training Data is written);
Discussion/Conclusion chapters (still placeholders). No LaTeX compiler
available in this environment to verify the build; brace/bracket/environment
balance checked manually and citation keys checked against `references.bib`.

**Artifacts:** `Writing/latex/chapters/chapter04_results.tex`,
`Writing/latex/chapters/chapter05_analysis.tex`, `Writing/latex/figures/*.png`
(new directory), `Writing/latex/references.bib` (two new entries).

---

### 29. Rewrote the Literature Review and expanded the Methodology chapter against actual supervisor feedback

**Trigger.** User asked to bring in the two Word first-draft chapters
(`Writing/Literature Review Chapter (first draft).docx`,
`Writing/Methodology Chapter (first draft).docx`) plus their supervisor
feedback docs (`Writing/Literature Review Feedback for Thiri.docx`,
`Writing/Thiri - Feedback for Methodology Chapter - AK.docx`), improve them
against that feedback, and add them into the LaTeX document. `python-docx`
was installed (`pip3 install python-docx`) and a small extraction script
written (checks body text, `word/comments.xml`, and tracked-change
insertions/deletions) since neither `pandoc` nor `python-docx` was already
available -- all four docs turned out to have their feedback as plain body
paragraphs, not Word comments or tracked changes.

**What the feedback actually said, in short.** Lit review: reads like a
technical intro, not a critical review -- cut basic gate/spider exposition,
add comparison/evaluation of approaches, sharpen the gap analysis, fix a
literal duplicated paragraph and a few typos. Methodology: reads like more
background rather than an experimental specification -- state the RQ/hypothesis/
outcome metric explicitly, define the R-vs-A comparison design, specify the RL
environment (features/actions/masking/reward/termination) and the network/PPO
settings precisely enough to reproduce, describe the evaluation protocol
including equivalence verification, and add reproducibility/limitations.

**What was actually built vs. what the draft proposed.** The old methodology
draft (Section 3.6) describes training on ripple-carry \emph{and Draper QFT}
adders against random circuits -- Draper QFT was never implemented; the actual
codebase uses Cuccaro adders plus a second random-circuit arm (Agent P) as
the control instead. Rewrote from the actual `Config`/`env.py`/`model.py`/
`evaluation.py` rather than transcribing the old proposal, and said so
explicitly in a new "Comparison design" section rather than silently
papering over the divergence.

**Literature Review** (`chapter02_literature_review.tex`, full rewrite):
compressed the gate/spider background to two short sections, then three
comparative sections -- optimisation approaches (gate-local vs. ZX-graph-rewriting
vs. learned, with Quarl/AlphaTensor/transformer-based methods as
non-RL learned alternatives), the ZX-calculus's own limitations (extraction
is \#P-hard per de Beaudrap/Kissinger/van de Wetering 2022, and `full_reduce`'s
own multi-metric non-monotonicity, forward-linked to this project's Chapter 5
finding), and a critical review of RL-for-ZX specifically (Riu et al.'s own
"majority not all" outperformance claim taken at face value, Zen et al.'s
AlphaTensor-Quantum reusability report used as field-wide evidence that
reproducibility is a live concern not unique to this project, Huang & Ontañón
on action-masking sensitivity, an interpretability point). Gap analysis
sharpened with Philipps (2025)'s direct negative generalisation result
(EP-trained policies underperforming on IQP/Clifford+T+CCZ) as the concrete
evidence the training-distribution question is real, and the risk of the
opposite failure (over-fitting to one narrow structured family) stated
up front rather than only after results are in. Five new `references.bib`
entries added (`zurek2003decoherence`, `debeaudrap2022circuit`,
`vandewetering2020zxcalculus`, `philipps2025gnns`, `brody2022attentive`);
several already-present-but-unused bib entries (`li2024quarl`,
`ruiz2025quantum`, `zen2026reusability`, `staudacher2023reducing`,
`charton2023teaching`, `huang2020closer`, `bukov2026reinforcement`,
`wu2020comprehensive`) turned out to cover exactly the comparative points
the feedback wanted and are now actually cited.

**Methodology** (`chapter03_methodology.tex`, expanded): kept the existing
Training Data section (entry from an earlier session) untouched, and added
Introduction (RQ/hypothesis/primary metric), Comparison design, RL Environment
(observation encoding, action types and why masking is structural rather than
post-hoc -- citing Huang & Ontañón's finding that masking strategy matters --
exact reward equation, termination rule), Network Architecture (GATv2 choice
justified via Brody et al.'s fix to GAT's static-ranking limitation), PPO
Training Settings (hyperparameter table; explicitly flagged that
`cfg.n_envs=8` is carried over from the paper's Table 1 but not functionally
used, since this project's loop is single-environment), Evaluation Protocol
(best-of-10 rationale restated from `evaluation.py`'s own docstring), and
Reproducibility \& Limitations. The equivalence-verification point from
feedback surfaced a real, previously-unstated gap: grepped the whole
codebase for `compare_tensors`/equivalence checks and found none in the
actual evaluation pipeline (only ad hoc use during development, verifying
individual new rules like gadget fusion) -- stated as an explicit limitation
rather than silently left unmentioned, with a note that it's flagged for
future work.

**Bugs hit and fixed along the way.** A reward-equation `\frac{m_{\text{best}}...}`
used `\text{}`, which needs `amsmath` -- the document only had `amssymb`
loaded, causing a fatal `Undefined control sequence` error on compile. Fixed
by adding `\usepackage{amsmath}` to `main.tex` (a generically useful, safe
addition for a document that will keep needing equations, not a narrow patch).
Also found and fixed one stray American-spelling "optimization" in prose
(pre-existing, from the earlier Training Data section) left inconsistent with
the rest of the document's British spelling -- left the three legitimate
`\texttt{basic\_optimization}` code-identifier occurrences alone, since that's
the actual `pyzx` function name, not a spelling choice.

**Not done.** The feedback also flagged a workflow diagram in the old draft
that reverses two pipeline steps -- no diagram exists in the LaTeX version at
all yet (text-only), so there's nothing to fix per se, but a real
circuit-to-graph-to-extraction pipeline figure would still strengthen
Section~2.4/3.5-equivalent material if produced later. Full document
recompiled clean after both fixes (41 pages, 0 undefined refs/citations,
only cosmetic overfull-hbox warnings on a couple of long `\texttt{}` tokens).

**Artifacts:** `Writing/latex/chapters/chapter02_literature_review.tex` (full
rewrite), `Writing/latex/chapters/chapter03_methodology.tex` (expanded),
`Writing/latex/main.tex` (`amsmath` added), `Writing/latex/references.bib`
(five new entries).

---

### 30. Agent A cflow-baseline smoke test: a clean, stronger negative confirmation

**Trigger.** User asked "can we smoke train using that" after a discussion of
what would most improve results, in reference to the single-step
`baseline_cflow_opt` (entry 26) which was already wired into `env.py`'s
terminal bonus but never actually training-tested. Ran
`scripts/train_agent_a_bits2_cflow_smoke.py` (new): the exact same bits=2-only,
5x-entropy-coefficient configuration as the earlier `agent_A_bits2` run
(entry 24-25), same seed, under a new run_name so the earlier evidence isn't
overwritten, for 60,000 timesteps (~117 updates) -- long enough to run well
past the ~35,000-timestep point where the earlier run had already collapsed.

**Result: still collapsed, at the identical point.** Entropy crossed below
0.01 at timestep 36,352 in both runs. Directly diffing the two runs' logged
`mean_reward` and `entropy` columns confirms they are **bit-for-bit identical
across all 75 overlapping updates** (max abs diff = 0.0). This is a stronger
result than it first looks: it isn't just "the cflow baseline didn't help
much" -- it means `baseline_cflow_opt` found zero improving single-step moves
across every single episode of an entire ~60,000-timestep run, so the
environment's reward signal was completely unchanged by adding it, and
training reproduced the prior run's trajectory exactly. This directly
confirms (with a full run's worth of evidence, not just entry 26's 15-episode
smoke test) that a single-step optimiser cannot help bit-width-2 adders at
all -- consistent with, and now much more strongly evidenced than, entry 25's
finding that the only known improving move there is a genuine 2-step
sequence.

**A background process-tracking pitfall worth remembering.** The first launch
attempt used `nohup ... & disown` inside a `run_in_background` Bash call,
which double-backgrounds the process: the harness reports the wrapper shell
as "completed" almost instantly (since it forked and returned), while the
actual training process keeps running fully detached from the harness's
tracking, with no completion notification ever coming. Caught by checking
`ps` directly and seeing the process still alive after the "completed"
notification. Fixed by killing it and relaunching as a plain foreground
command under `run_in_background: true` with no shell backgrounding
operators -- the correct pattern for this harness to track and notify on
completion of a long-running local process.

**Not done / not changed.** This was a diagnostic run, not a retrain of the
actual reported Agent A checkpoint -- no change to `docs/project_log.md`
entries 24-26's standing conclusions, and Chapter 5's discussion of the
single-step baseline's limits (Section on the partial cflow fix) already
anticipated this outcome; this entry is the confirmed version of that
prediction, not a new one. Whether to fold this specific confirmation into
the dissertation text itself was left to the user to decide.

**Artifacts:** `scripts/train_agent_a_bits2_cflow_smoke.py` (new),
`results/logs/agent_A_bits2_cflow_smoke_train_log.csv`,
`results/logs/agent_A_bits2_cflow_smoke_stdout.log`.

---

### 31. Batched the PPO update (ppo.py), verified against the original loop before trusting it

**Trigger.** User asked to vectorise the training loop. `ppo.py`'s own docstring
had already flagged this as future work, splitting it into two separable
problems: batching the PPO *update* (tractable, tensor-side only) vs.
parallelising environment *rollout collection* (harder -- `pyzx` has no
batched/vectorised API of its own, so that would need real multiprocessing).
Given this touches the exact code that produced every checkpoint reported so
far, presented both options plus a "just discuss it" option rather than
picking one; user chose the scoped, lower-risk option: batch the update only.

**What changed.** `ppo_update`'s inner `for i in batch_idx: policy.evaluate_actions(...)`
loop (one GNN forward pass per transition) replaced with a single
`torch_geometric.data.Batch.from_data_list(...)` forward pass per minibatch
(`evaluate_actions_batched`, new). The one real subtlety: batching
concatenates every graph's action-node logits into one flat tensor, so a
plain softmax over that tensor would blend actions from different graphs
together. Fixed using `torch_geometric.utils.softmax`/`.utils.scatter`,
which do a *segmented* softmax/sum grouped by which graph each entry belongs
to (`action_batch`) -- this is what keeps each graph's action distribution
independent despite the shared forward pass. Converting each transition's
own *local* action index into the right *global* offset into the flat batch
uses an exclusive-prefix-sum over per-graph action-node counts.

**Verification, not just trust.** The original per-transition logic was kept
as a named reference function (`evaluate_actions_loop`, not inlined away)
specifically so the batched version could be tested against it, not just
assumed correct. New `tests/test_ppo_batching.py`: collects a real rollout
from `pure_random` circuits deliberately (varying episode/graph size on
purpose, so the per-graph offset logic is actually exercised, not just
fixed-size graphs that would pass trivially even with an indexing bug), then
checks (a) `evaluate_actions_batched`'s log-probs/entropies/values match
`evaluate_actions_loop`'s within `1e-4` tolerance, and (b) -- specifically
because a batching bug could break gradient *routing* through the shared GNN
trunk without showing up in forward values at all -- that backpropagating a
simple scalar loss through each path produces matching parameter gradients
too, within `1e-3`. Both pass. Full existing suite (`tests/test_zx_utils.py`)
still passes unmodified -- 10/10 total.

**Measured speedup, not assumed.** Timed both paths on a real 512-transition
rollout (`cfg.n_steps`, one full minibatch): forward+backward through
`evaluate_actions_loop` took 0.893s, `evaluate_actions_batched` took 0.186s
-- **4.8x** on the update computation itself. Rollout collection (still
per-step, untouched by this change) took 2.51s for the same 512 steps.
Combining across a full update's 8 PPO epochs: old estimated total
$\approx 2.51 + 8\times0.893 = 9.65$s, new $\approx 2.51 + 8\times0.186 =
4.0$s -- **roughly 2.4x** end-to-end per update, bounded below the update-only
4.8x because rollout collection (the `pyzx`-bound half of every update) isn't
touched by this change, exactly as flagged going in. A 4-update end-to-end
smoke run through the real `train()` function (not just the isolated
functions) confirmed the full pipeline -- checkpointing, CSV logging,
heartbeat file -- still works unmodified, at ~3.3-3.7s/update, consistent
with the estimate. If this ratio holds across a full run, a 2M-timestep
training run would drop from ~16-17h to roughly ~7h wall-clock -- not
confirmed with an actual full run yet, since none was re-launched this
session.

**Not done.** Rollout-collection parallelism (the larger, `pyzx`-bound half
of the cost) remains unimplemented, per the user's explicit scope choice.
R/P/A's already-reported checkpoints are untouched by this change -- it only
affects the *speed* of any future training run, not any existing result.

**Artifacts:** `src/circopt_adder/ppo.py` (`evaluate_actions_batched`,
`evaluate_actions_loop`, `ppo_update` rewritten to use the batched path;
module docstring updated), `tests/test_ppo_batching.py` (new).

---

### 32. Fact-checked how the paper's own environment applies basic_optimization around the agent

**Trigger.** User asked to fact-check a specific claim: that Riu et al.'s
pipeline applies `full_reduce` or `basic_optimization`, then the trained
agent, then extracts the circuit and applies `basic_optimization` again.
Re-fetched `agent_test.py` and `zx_env.py` fresh from
`qilimanjaro-tech/Circopt-RL-ZXCalc` (main branch) rather than relying on
entry 27's summary of the same file, and read `step()`/`reset()` in full.

**Verdict: correct.** Traced exactly:
- `reset()`: the freshly generated circuit is passed through
  `self.basic_optimise()` (their `zx.basic_optimization` wrapper) *before*
  the agent ever sees it -- this becomes the circuit the episode's
  `initial_stats` are measured against.
- `step()`: after the agent applies its chosen action to the graph, the
  circuit is extracted (`zx.extract_circuit(graph, up_to_perm=True)`) and
  `self.basic_optimise()` is applied *again* -- and this happens on **every
  single step**, not only at the episode's end. That cleaned-up gate count is
  what the per-step reward is computed from.

**A real correction surfaced along the way.** Re-reading `step()`'s terminal
block (reached at episode end) to answer this precisely, found that entry
27's claim -- that this environment reads a precomputed `circuit_data` value
for the cflow-zx baseline rather than computing it live -- was wrong.
`new_gates_cflow = circuit_data[self.gate_type]` reads a *local* variable
assigned two lines above it, from a live call chain in the same block:
`zx.teleport_reduce(g)` -> `zx.to_graph_like(g)` -> `zx.flow_2Q_simp(g)`
(Holker's actual cflow-zx algorithm) -> `zx.extract_simple` ->
`self.basic_optimise()`. So the environment *does* call the real algorithm
live, once per episode, specifically to compute the terminal-bonus target --
the earlier "reads a cache" conclusion conflated a genuine but unrelated
piece of dead code (`# from pyzx.gflow import gflow`, a leftover commented
import from an earlier iteration) with evidence of caching that doesn't
exist. Corrected in place at entry 27 rather than only here, since that's
where a future reader would encounter the wrong claim first. This claim
never made it into the actual dissertation LaTeX text (checked: no match
anywhere under `Writing/latex/`) -- it was confined to entry 27 and this
session's own prior chat responses, so no chapter file needed correcting,
only the log.

**Not affected by this correction:** entry 27's fork investigation
(`calumholker/pyzx`'s `cflow()`/`flow_2Q_simp()` implementations), the
corrector-collision finding, and the `g.replace()` lead for the unresolved
multi-step chaining bug all stand as originally written -- none of that
relied on the wrong caching claim.

**Artifacts:** none (fact-check and correction only); reference files saved
to this session's scratchpad (`agent_test.py`, `zx_env.py`), not part of the
repo.

---

### 33. Implemented gflow-zx, fixed the multi-step cflow-zx chaining bug via the
    `g.replace()` lead, and found (then corrected) a real error in entry 25

**Trigger.** User asked whether cflow and gflow were both implemented (only
cflow was), then whether gflow-zx could be added, then whether the four
baselines' comparison graph from Riu et al.'s Figure 2 could be reproduced.
That reproduction worked and directly showed the single-move restriction's
cost (cflow-zx/gflow-zx track `basic-opt` almost exactly rather than clearly
beating it, unlike the paper's own iterated versions). With five days left
before submission and the dissertation otherwise content-complete, user asked
to explore further, starting with whether gflow-zx (a strictly more
permissive flow condition than cflow) finds anything on Agent A's adders that
cflow-zx's single-move version could not, then to attempt the `g.replace()`
fix identified in entry 27.

**gflow-zx implementation** (`baselines.baseline_gflow_opt`,
`zx_utils.replace_in_place`). Unlike causal flow, `pyzx` 0.10.5 already ships
a working gflow-finding algorithm (`pyzx.gflow.gflow`, Perdrix & Mhalla via
GF(2) linear algebra) -- no need to reimplement it from the paper, unlike
`causal_flow.py`. gflow has no closed-form gate-count formula, though
(Riu et al.: "unclear how to predict the resulting amount of two-qubit gates
without performing circuit extraction" for gflow specifically), so each
candidate move is scored via real extraction instead of a graph-only
formula -- slower per candidate (up to ~1.3s on an 80-2q-gate circuit) but
entirely tractable; a 32-point sweep to 320 gates ran in ~90s.

**Single-move gflow-zx vs. cflow-zx on Agent A's adders (all 7 relevant
bit-widths, deterministic, no resampling needed since ripple-carry adders
are exact per bit-width).** Neither found an improving move at bits
2/3/4/5/7/8; both found the *same* one-move improvement at bits=9
(99->98, ~1%), previously untested since entries 24-25 only covered the
trained widths. **gflow-zx found nothing cflow-zx could not** -- rules out
"wrong flow condition" as the bottleneck, strengthening the case that the
single-move restriction itself was the more promising thing to fix next.

**The `g.replace()` fix.** Mainline `pyzx` has no `.replace()` (confirmed:
`grep -rn "def replace" pyzx/` finds nothing on `BaseGraph`/`GraphS`), so
`zx_utils.replace_in_place(g, source)` reimplements its effect directly --
overwrites `g`'s own internal attribute dicts (`graph`, `ty`, `_phase`,
`_qindex`, `_rindex`, `_vdata`, `_edata`, `phase_index`, `phase_mult`, plus
scalar attrs) from a scored clone, so `g`'s Python identity survives every
committed move. Combined with `.clone()` (already in mainline pyzx, unlike
`.replace()` -- and unlike `.copy()`, which silently renumbers vertex IDs on
every call, the most likely actual bug in the original abandoned attempt).
New `baselines.baseline_cflow_multistep()` reimplements Holker's greedy
chained-move algorithm using `check_g = g.clone()` / apply / re-check flow /
`replace_in_place(g, check_g)` on success, exactly mirroring the reference
implementation's commit pattern (entry 27). **Verified: 20/20 chained-move
trials on random circuits (up to 10 moves each) completed with zero
extraction failures**, where the old `g.copy()` + reference-reassignment
version reliably failed. The candidate cause identified in entry 27 was
correct.

**Then: could not reproduce entry 25's own "22->20" result, or its "53"
result at bits=5.** Re-ran the same exhaustive 2-step search entry 25
describes, now via the fixed multi-step machinery, on all three originally-
tested bit-widths (2, 3, 5), covering LC, pivot, *and* gadget-fusion moves
(the full implemented action space), under both the causal-flow and gflow
conditions. Step-1 move counts matched entry 25's table exactly (13/21/37).
Total 2-step combinations did not (124/368/1240 vs. the logged 44/135/285),
and **no improving 2-step sequence was found for any of the three
bit-widths** -- bits=3's "nothing found" row is the only one of the three
that reproduces; the two rows that claimed an improvement (bits=2, bits=5)
both do not. This pattern (the "nothing found" result reproduces, both
"found an improvement" results don't) suggests entry 25's original ad hoc
search likely had its own bug, rather than this session's broader search
missing something -- but it wasn't run as a saved script, so there is
nothing to diff against and the root cause is not identified. Correction
appended in place at entry 25 rather than silently edited.

**Dissertation text corrected to match.** Chapter 5
(`subsec:analysis-cflow-fix`) previously stated the bits=2 "22->20" result
as an established fact explaining why the single-move restriction blocked
Agent A specifically. Rewritten to: (a) describe the `g.replace()` fix as
completed and verified, not a future lead; (b) state the correction
directly, rather than silently dropping the old claim; (c) conclude that the
single-move restriction is *not*, by itself, a sufficient explanation for
cflow-zx's failure to help Agent A, since even the working multi-step
version finds nothing further at depth two -- a narrower, more honest
finding than the one it replaces. Chapter 6's Limitations and Future
Research sections updated correspondingly (the `g.replace()` lead is no
longer future work; the open future-work item is now retraining Agent A
with the completed multi-step optimiser wired into training itself, not
just used as a post-hoc evaluation tool, which remains untested). Chapter 3
(Methodology) gained a new section, `sec:methodology-baselines`, defining
all four deterministic baselines (`basic_optimization`, `full_reduce`,
`cflow-zx`, `gflow-zx`) together with a comparison figure
(`figures/baseline_comparison.png`, generated by
`scripts/plot_baseline_comparison.py`) sweeping all four across random
10-qubit Clifford+T circuits, so later chapters can reference one shared
definition instead of re-deriving it piecemeal.

**Decision.** `baseline_cflow_opt` (single-move, live in `env.py`'s terminal
bonus) is untouched by any of this -- `baseline_cflow_multistep` and
`baseline_gflow_opt` are new, separate functions, not wired into training.
Retraining Agent A with either is explicitly flagged as future work
(Chapter 6), not attempted here, given the remaining time before submission.

**Artifacts:** `src/circopt_adder/baselines.py` (`baseline_gflow_opt`,
`baseline_cflow_multistep`), `src/circopt_adder/zx_utils.py`
(`replace_in_place`), `scripts/compare_cflow_gflow_adders.py`,
`scripts/reproduce_riu_figure2.py`, `scripts/plot_baseline_comparison.py`,
`results/figures/reproduction_riu_figure2.png`,
`results/figures/baseline_comparison.png` (also copied to
`Writing/latex/figures/`). Dissertation chapters edited:
`chapter03_methodology.tex`, `chapter05_analysis.tex`,
`chapter06_discussion.tex`.

---

### 34. Scoped cflow-zx/gflow-zx back out of the dissertation narrative: literature
    citation only, not a project-implemented baseline

**Trigger.** After checking Riu et al.'s own text directly (Section 3.3.2:
their terminal reward against `cflow-zx` is stated as necessary to prevent
exactly Agent A's collapse, not an optional refinement) and confirming via
this log that the multi-step cflow-zx/gflow-zx work (entry 33) was never
part of any actually-reported training run, user reconsidered: this
dissertation is inspired by Riu et al.'s approach, not a full
reimplementation of it, and asked to remove cflow-zx/gflow-zx from the
dissertation's experiment description and revert to `full_reduce` and
`basic_optimization` as the only baselines discussed, keeping only the
literature fact (Riu et al. use `cflow-zx`, and state why) as a citation.

**What changed (dissertation text only, `Writing/latex/`).** Removed
project-implementation claims and the entire "Deterministic baseline
methods" section and its 4-way comparison figure from Chapter 3; removed
Chapter 5's "verified fix" subsection (the multi-step implementation, the
gflow-zx addition, and the entry-25 correction narrative) in full, since
none of that is needed to support the chapter's actual central claim (weak
terminal baseline -> collapse), which rests on Riu et al.'s own stated
design rationale plus this project's own empirical baseline-weakness
result, not on a project-built cflow-zx; updated the Reward subsection's
terminal-bonus formula to drop the `$m_{\text{cflow}}$` term, which is now
also a *more* accurate description of what actually trained the reported
R/P/A checkpoints (entry 33 already established `cflow-zx` was added to
`env.py` only after those runs completed); adjusted Chapter 6's
Recommendations, Contributions, Limitations, and Future Research
accordingly (the "strongest test not attempted" limitation is essentially
back to its pre-entry-33 form, without the specific 22->20/g.replace()
detail); fixed Chapter 7's Objective 4 summary and Chapter 1's
Contributions section, which both referenced the now-removed subsection;
fixed two mentions in Chapter 2's worked-example walkthrough that
referenced the project-written cflow-zx approximation.

**What did *not* change.** The actual Python code from entry 33
(`baseline_gflow_opt`, `baseline_cflow_multistep`,
`baseline_gflow_multistep`, `zx_utils.replace_in_place`) is untouched and
still importable -- it simply isn't referenced by the dissertation
narrative any more. `env.py`'s live `baseline_cflow_opt` call (gated to
`reward_mode == "twoqubits"`) is also untouched; as established in entry
33, it was never active during the reported R/P/A training runs regardless.
No retraining, no evaluation-pipeline changes -- `full_reduce` and
`basic_optimization` were already the only baselines used in the actual
evaluation protocol (Chapter 4), so this is a documentation/narrative
simplification, not an experimental one.

**Result.** Chapters 1-7 word count dropped from 14,894 to 13,595 (chapters
1-7 only). Recompiled clean: 43 pages, 0 undefined refs/citations.

**Artifacts:** `Writing/latex/chapters/chapter01_introduction.tex`,
`chapter02_literature_review.tex`, `chapter03_methodology.tex`,
`chapter05_analysis.tex`, `chapter06_discussion.tex`,
`chapter07_conclusion.tex` (all edited, no new files). No changes under
`src/` or `scripts/`.

---

### 35. Dropped the direct results comparison against Riu et al.'s reported
    numbers; success criterion is now "beats the deterministic baselines"

**Trigger.** Continuing the "inspired by, not reimplementing" reframing
from entry 34, user asked to remove any section directly comparing this
project's trained-agent results against \citet{riu2025reinforcement}'s own
reported figures (the $24.6\%$ vs.\ $2.14\%$ real-world-benchmark gap),
and to reframe success purely in terms of beating
`full_reduce`/`basic_optimization`/`teleport_reduce`, which the project's
own agents already do consistently across every evaluation family.

**What changed.** Removed Chapter 4's dedicated "Comparison against Riu et
al. (2025)" section in full (the $24.6\%$ figure, the $2.14\%$
recomputation, the sample-budget/hybrid-workflow caveats); its chapter
summary now states plainly that every trained agent outperformed both
deterministic baselines on every evaluation family. Chapter 5's central
section, previously titled "The gap to Riu et al.'s results, and Agent A's
collapse, share a cause," is retitled "Agent A's collapse, and Agent R/P's
modest margins, share a cause" -- the causal argument itself is unchanged
(weak terminal baseline, grounded in \citeauthor{riu2025reinforcement}'s
own Section 3.3.2 warning, explains both Agent A's total collapse and why
R/P plateau at small margins), it just no longer needs the paper's
reported number as the thing being explained; R/P's own modest margin over
the deterministic baselines is sufficient motivation on its own. Chapters
1, 6, and 7 had smaller matching edits (Objective 3's stated evaluation
target, a contribution-to-knowledge point, and the Objective 3
retrospective in the Conclusion) to remove the same paper-comparison
framing.

**What did not change.** The underlying evaluation data is untouched --
Table~\ref{tab:results-2q-by-family} in Chapter 4 still reports the same
numbers for every method on every family; only the *comparison target*
was removed, not any result. The root-cause mechanism in Chapter 5 (weak
terminal baseline -> trivially-satisfied "beat the baseline" bonus ->
collapse) is unchanged and still grounded directly in
\citeauthor{riu2025reinforcement}'s own stated design rationale, which
remains cited as literature evidence throughout.

**Result.** Chapters 1-7 word count: 13,649 (from 13,595 after entry 34 --
Chapter 5 grew slightly reframing its argument, Chapter 4 shrank by
removing the comparison section). Recompiled clean: 43 pages, 0 undefined
refs/citations.

**Artifacts:** `Writing/latex/chapters/chapter01_introduction.tex`,
`chapter04_results.tex`, `chapter05_analysis.tex`,
`chapter06_discussion.tex`, `chapter07_conclusion.tex` (all edited, no new
files).

---

### 36. Added `teleport_reduce`/`clifford_simp` as measured baselines; found a
    real re-extraction artefact and an eval-circuit reproducibility gap

**Trigger.** Continuing the baseline-methods thread, user asked to add
`teleport_reduce` and `basic_opt+teleport_reduce` as evaluated baselines
alongside the existing `full_reduce`/`basic_opt+full_reduce`, then asked a
broader question (are all four baseline functions plus `clifford_simp`
actually used anywhere, and why specifically `teleport_reduce` +
`clifford_simp` for the terminal bonus), then requested a dedicated
100-circuit-per-family sweep of all six methods, then asked to add a
Methodology section using that sweep's results table plus a comparison
figure.

**New baseline functions** (`baselines.py`): `baseline_basic_optimization`,
`baseline_clifford_simp`, `baseline_teleport_reduce`,
`baseline_basic_plus_teleport_reduce`. First version of the latter two had
a real bug -- missing `zx.simplify.to_graph_like(g)` before
`zx.extract_circuit(g)` -- causing `"Input graph is not graph-like"` on
every one of the 47 real-world benchmark circuits (47/47 failures) when
first run through `evaluate.py`. Fixed by matching `_light_preprocess`'s
own pattern (which already includes this step) exactly; re-ran clean, 0
failures.

**A confirmed re-extraction artefact, not a project bug.** Investigating
why `teleport_reduce` showed positive-looking 2Q reduction (9.8%/7.7%/25.0%
by family) despite its own provable 2Q-neutrality, built a zero-
simplification control (`_roundtrip_only`: circuit -> graph ->
`to_graph_like` -> extract, no rewrite rule applied at all). It produced
**identical** results to `teleport_reduce`, circuit for circuit, across
all 300 swept circuits (100 per family) and on the four specific held-out
adder bit-widths from Table~\ref{tab:results-adder-detail} (4/7/8/9-bit,
all exactly 25.0%). `teleport_reduce`'s own contribution is confirmed
zero; the apparent reduction is entirely `pyzx`'s extraction step
synthesising a different gate decomposition of the same structure.

**Consequence for the already-reported adder result.** Of the reported
31.25% two-qubit reduction on held-out adders (identical across all three
agents, already flagged in Chapter 5 as likely a search-budget artefact),
**25 of the 31.25 percentage points (80%) is this same re-extraction
artefact, not real optimisation by any method** -- traced to
`evaluate.py`'s `build_eval_circuits()` using raw `ripple_carry_adder(n_bits)`
directly for the adder eval set, unlike the random/pure-random eval sets
which reuse the training generators (and therefore already went through
`_light_preprocess`, which absorbs this same artefact before evaluation
starts). Only 6.25 percentage points -- consistent across all four bit
widths -- reflects anything any method actually did. Not yet corrected in
the dissertation text; flagged to the user, decision pending on how (or
whether, given time remaining) to fold this into Chapter 5's existing
adder-generalisation discussion.

**A separate reproducibility gap, also found while re-running
`evaluate.py`.** The random/pure-random eval circuit generators draw from
`pyzx`'s own unseeded global RNG (`zx.generate.cliffordT` has no seed
hook), not the local seeded `random.Random` the generator functions
accept -- confirmed directly: re-running `evaluate.py` twice produced
different structured-random/pure-random circuits and materially different
results each time (e.g. Agent R's structured-random figure: $2.3\%$
originally reported vs.\ $-4.2\%$ on this re-run), while the
deterministic adder and real-world-benchmark circuits stayed stable.
Table~\ref{tab:results-2q-by-family}'s structured-random/pure-random
columns were therefore never exactly reproducible from a fresh run, on
top of the already-documented small-sample-size ($n=5$) caveat. Not fixed;
flagged to the user, decision also pending.

**100-circuit sweep** (`scripts/sweep_baselines_100.py`, new): all six
baselines applied to 100 unpreprocessed circuits per family (structured-
random, pure-random, adders with bit-width uniform in $[2,10]$), with the
round-trip control included for interpretability. Results (mean \% 2Q
reduction) -- structured-random / pure-random / adders:
`basic_opt` +2.9/+3.6/+31.3, `full_reduce` -40.0/-91.3/-56.7,
`teleport_reduce` +9.8/+7.7/+25.0, `basic_opt+full_reduce`
-25.8/-80.8/-36.9, `basic_opt+teleport_reduce` +3.7/+4.5/+31.3,
`clifford_simp` -40.9/-93.9/-25.2. Two clean clusters: {basic_opt,
teleport_reduce, basic_opt+teleport_reduce} track the diagonal and stay
non-negative everywhere; {full_reduce, basic_opt+full_reduce,
clifford_simp} diverge sharply worse as circuit size grows, on every
family. This is the first direct measurement of `clifford_simp`'s own
weakness -- previously only asserted by analogy to `full_reduce`, never
measured.

**New Methodology section** (`sec:methodology-baselines`, back in
`chapter03_methodology.tex` after being removed at entry 34, this time
without cflow/gflow): defines all six methods, presents the sweep table
and a new comparison figure
(`scripts/plot_baseline_comparison.py`, `figures/baseline_comparison.png`,
regenerated for six methods on raw circuits, replacing the earlier
cflow/gflow version), states the re-extraction-artefact finding directly
as a methodological caveat, and states plainly which methods serve which
of the three roles (preprocessing, terminal-bonus reward, evaluation) --
directly answering the user's "where is each one actually used" question
in the dissertation text itself, not just in conversation.

**Result.** Chapters 1-7 word count: 14,315. Recompiled clean: 45 pages, 0
undefined refs/citations.

**Artifacts:** `src/circopt_adder/baselines.py` (four new functions),
`scripts/sweep_baselines_100.py` (new), `scripts/plot_baseline_comparison.py`
(rewritten for six methods), `results/figures/baseline_comparison.png`
(also copied to `Writing/latex/figures/`), `results/logs/evaluation_results.csv`
(regenerated, now includes `teleport_reduce`/`basic_opt+teleport_reduce`
rows). `Writing/latex/chapters/chapter03_methodology.tex` edited (new
section). Two open findings not yet acted on: the adder re-extraction
confound and the eval-circuit RNG reproducibility gap, both above.

---

### 37. Fixed the baseline-comparison figure's red-region shading bug

**Trigger.** User noticed the "unsuccessful compression" red region in
`figures/baseline_comparison.png` only covered about half the visible
plot area.

**Root cause.** `plot_baseline_comparison.py` bounded the red
`fill_between`'s upper edge at `hi` (the max \emph{initial}-gate-count
value on the x-axis, $\approx$86), not the actual maximum y-value in the
plotted data. `full_reduce`/`clifford_simp`'s own output points run up to
$\approx$210, well above 86, so those points -- exactly the ones the red
shading exists to flag -- were sitting on an unshaded white background.

**Fix.** Compute `y_top = max(hi, max(all_y))` and bound the red fill's
top edge there instead, plus set `ax.set_ylim(top=y_top*1.02)` so the
axes and shading agree. Regenerated the figure, copied into
`Writing/latex/figures/`, recompiled clean (0 undefined refs/citations).
Same bug pattern exists in two now-unused exploratory scripts
(`reproduce_riu_figure2.py`, `plot_baseline_comparison_multistep.py`) --
left as-is since neither is embedded in the dissertation.

**Artifacts:** `scripts/plot_baseline_comparison.py` (fixed),
`results/figures/baseline_comparison.png` (regenerated, copied to
`Writing/latex/figures/`).

---

### 38. Experiment audit, step 1: dropped `clifford_simp` from the baseline
    characterisation entirely

**Trigger.** User opened a broader "redesign/audit the experiment" pass,
starting with the baselines, asking to drop `clifford_simp` completely.

**Checked first, since this differs from the cflow-zx/gflow-zx removal
(entry 34).** `clifford_simp` is not like cflow-zx: it's used live in
`env.py`'s terminal-bonus reward (`zx.simplify.clifford_simp(g_korb)`,
inline, not via the `baselines.py` wrapper added this session), and *was*
part of what actually trained the reported Agent R/P/A checkpoints from
the start -- unlike cflow-zx, which was added only after those runs
completed. Dropping it from the dissertation's characterisation work
(this session's new sweep/table/figure) is safe and doesn't touch
training history; dropping it from the Reward subsection's description of
the actual formula would not be historically accurate, so that mention
was kept.

**What changed.** Removed `clifford_simp` from
`scripts/sweep_baselines_100.py` and `scripts/plot_baseline_comparison.py`
(both re-run; five methods now, not six), removed the now-unused
`baseline_clifford_simp` function from `baselines.py` entirely (confirmed
unreferenced anywhere else first), and updated
`chapter03_methodology.tex`'s `sec:methodology-baselines`: five methods
defined/characterised (down from six), table and figure both regenerated
with fresh sweep numbers, the "two clusters" paragraph now describes only
`full_reduce`/`basic_opt+full_reduce` as the divergent cluster. The
"which method serves which role" paragraph keeps `clifford_simp`
mentioned as part of the actual live terminal-bonus reward (historically
accurate), with an explanatory sentence citing the earlier (now removed
from the table) measurement showing it behaves like `full_reduce`
(substantially worsening two-qubit count, 25-94\% depending on family)
rather than like `teleport_reduce` -- which sharpens the existing point
that neither of the reward's two comparison targets ever meaningfully
beats doing nothing.

**Result.** Chapters 1-7 word count: 14,352. Recompiled clean: 45 pages, 0
undefined refs/citations.

**Artifacts:** `src/circopt_adder/baselines.py` (function removed),
`scripts/sweep_baselines_100.py`, `scripts/plot_baseline_comparison.py`
(both edited, re-run), `results/figures/baseline_comparison.png`
(regenerated, five methods, copied to `Writing/latex/figures/`).
`Writing/latex/chapters/chapter03_methodology.tex` edited. Audit
continuing per user's stated intent -- this is step 1, not the full scope.

---

### 39. Made `baseline_comparison.png` a two-panel figure (total gates +
    two-qubit gates), mirroring Riu et al.'s Figure 2 layout

**Trigger.** User asked to add a companion panel for total gate count
alongside the existing two-qubit-gate-count figure.

**What changed.** Rewrote `scripts/plot_baseline_comparison.py`: now
computes both `gate_count` and `two_qubit_gate_count` per circuit/method
and renders two side-by-side panels, (a) total gate count, (b) two-qubit
gate count, each against its own initial count, both fixed against the
earlier red-shading bug (entry 37). Same five methods, same random
10-qubit sweep. Figure regenerated and copied to `Writing/latex/figures/`;
`chapter03_methodology.tex` updated to reference both panels, widened to
full text width, and gained a short new paragraph on the finding this
surfaced: **`teleport_reduce` is the worst performer on total gate count**
specifically (it targets T-count only and provably can't touch two-qubit
gates, so it has no mechanism to reduce overall gate count either), while
`basic_opt` performs best there -- the reverse of the two-qubit-count
picture, illustrating that "best baseline" is metric-dependent, not a
fixed property of a method.

**Result.** Chapters 1-7 word count: 14,459. Recompiled clean: 45 pages, 0
undefined refs/citations. Visually verified: both panels fit within text
width at full resolution, legend below, no overflow.

**Artifacts:** `scripts/plot_baseline_comparison.py` (rewritten),
`results/figures/baseline_comparison.png` (regenerated, two-panel, copied
to `Writing/latex/figures/`). `Writing/latex/chapters/chapter03_methodology.tex`
edited.

---

### 40. Found and fixed a real bug in `baseline_basic_plus_teleport_reduce`
    while answering "can we drop the combo baselines"

**Trigger.** User asked whether `basic_opt+full_reduce` and
`basic_opt+teleport_reduce` differ meaningfully from `full_reduce`/
`teleport_reduce` alone, or could be dropped as redundant.

**Checked circuit-by-circuit, not just aggregate means** (15 structured-random
circuits, direct comparison). `full_reduce` vs. `basic_opt+full_reduce`:
genuinely different on 14/15 circuits -- keep both, confirmed not redundant.
`basic_opt` vs. `basic_opt+teleport_reduce`: **identical two-qubit count on
all 15/15 circuits**, and total gate count roughly 1.5-2x inflated for the
`+teleport_reduce` version (e.g. 44 vs.\ 76 gates) despite that.

**Root cause: a real bug, not a property of the methods.**
`baseline_basic_plus_full_reduce` runs `basic_optimization` both before
*and after* extraction (matching the dissertation's own stated
description). `baseline_basic_plus_teleport_reduce` only ran it *before*
-- the post-extraction cleanup pass was missing entirely, so the inflated
gate count from `pyzx`'s re-extraction (entry 36's re-extraction artefact)
was never cleaned up the way the other combo method's is. Fixed by adding
the missing `zx.basic_optimization(c2)` call, matching the other
function's pattern exactly.

**Re-verified after the fix.** Total gate counts dropped back in line with
`basic_opt` alone (e.g. the 76-gate case above became 43). Two-qubit
counts, previously identical on 15/15 circuits, now differ on roughly
40\% of circuits -- expected, since the post-extraction `basic_optimization`
pass is not provably two-qubit-neutral the way `teleport_reduce` itself
is. In aggregate over the full 100-circuit sweep, though, the differences
run in both directions and largely cancel: `basic_opt+teleport_reduce`'s
mean is within a few points of `basic_opt` alone in every family, and
identical on adders (31.3\% both). Re-ran `sweep_baselines_100.py` and
`plot_baseline_comparison.py` with the fix; both dissertation table
(Table~3.4) and figure updated with corrected numbers.

**Answer given to the user's actual question, now evidence-based.** Keep
`full_reduce` and `basic_opt+full_reduce` both (genuinely different).
`basic_opt+teleport_reduce` is statistically near-redundant with
`basic_opt` alone in aggregate -- but retained anyway, since it is
literally the project's own preprocessing pipeline
(`_light_preprocess`), and characterising the actual pipeline in its own
right is more defensible than assuming it behaves like `basic_opt` alone.
Documented this reasoning directly in `sec:methodology-baselines` rather
than only in this log.

**Result.** Chapters 1-7 word count: 14,641 -- getting closer to the
15,000 nominal target (still comfortably inside the 10\% leniency band,
but worth watching). Recompiled clean: 45 pages, 0 undefined
refs/citations.

**Artifacts:** `src/circopt_adder/baselines.py` (bug fixed in
`baseline_basic_plus_teleport_reduce`), `scripts/sweep_baselines_100.py`
and `scripts/plot_baseline_comparison.py` re-run,
`results/figures/baseline_comparison.png` (regenerated, copied to
`Writing/latex/figures/`). `Writing/latex/chapters/chapter03_methodology.tex`
edited (table numbers corrected, new paragraph added).

---

### 41. Added a third panel (T-count) to the baseline-comparison figure

**Trigger.** User asked for a T-count panel alongside the existing total-
gate-count/two-qubit-gate-count panels, "to get the full picture."

**What changed.** `scripts/plot_baseline_comparison.py` now computes
`t_count` per circuit/method alongside the existing two metrics and
renders three side-by-side panels instead of two: (a) total gate count,
(b) two-qubit gate count, (c) T-count. Figure regenerated and copied to
`Writing/latex/figures/`; `chapter03_methodology.tex` updated (both the
lead-in sentence and the figure caption now mention all three panels),
with a new closing sentence on what panel (c) shows: **every method,
`full_reduce` and `basic_opt+full_reduce` included, lands in the
successful-compression region on T-count, with no divergence at all** --
the concrete, quantified version of the metric-tradeoff already
introduced in Chapter 2 (Section~2.4/`sec:lit-zx`): `full_reduce` isn't
simply a bad method, it is specifically bad at two-qubit gate count while
being consistently competent at the non-Clifford-phase reduction it
actually targets.

**Result.** Chapters 1-7 word count: 14,721. Combined with front matter
(abstract + declaration + pro-forma + title page, roughly 770 words),
total is now past the nominal 15,000-word target (~15,490) though still
comfortably inside the 10\% leniency band (16,500). Flagged to the user;
worth watching on any further additions. Recompiled clean: 45 pages, 0
undefined refs/citations, figure visually verified to fit at full text
width with all three panels legible.

**Artifacts:** `scripts/plot_baseline_comparison.py` (three-panel
rewrite), `results/figures/baseline_comparison.png` (regenerated, copied
to `Writing/latex/figures/`). `Writing/latex/chapters/chapter03_methodology.tex`
edited.

---

### 42. Styling: axes fill to the border, gridlines added; confirmed circuit
    counts for the table vs. the figure

**Trigger.** User asked how many circuits the figure/table are based on,
and to remove the default matplotlib margin (fill each panel to its
border) and add gridlines.

**Circuit counts, confirmed by reading the script, not memory.**
Figure~\ref{fig:baseline-comparison}: 32 circuits (`DEPTHS =
range(10, 321, 10)`), one structured-random circuit per depth, each run
through all five methods. Table~\ref{tab:baseline-sweep-100}
(`sweep_baselines_100.py`): 300 circuits (100 per family x 3 families).
These are two different, independently-generated samples -- worth being
clear the two aren't the same 100(or 32) circuits reused.

**Styling change.** `_plot_panel()` in `plot_baseline_comparison.py` now
sets `ax.set_xlim`/`set_ylim` explicitly to the data's own bounds (`lo`,
`hi`, and `y_bottom = min(lo, min(all_y))` in case any method's output
goes below the panel's x-range minimum) instead of leaving matplotlib's
default 5\% auto-margin, plus `ax.margins(0)`, plus white gridlines at
`zorder=1.5` (above the red/green shading at 0/1, below the data lines
and markers at their default zorder 2) so they read clearly against both
shaded regions without competing with the plotted data.

**Result.** Regenerated, copied to `Writing/latex/figures/`. Recompiled
clean: 45 pages, 0 undefined refs/citations.

**Artifacts:** `scripts/plot_baseline_comparison.py` (styling only, no
data/logic change), `results/figures/baseline_comparison.png`
(regenerated, copied to `Writing/latex/figures/`).

---

### 43. Denser depth sweep for the baseline-comparison figure (step 10 -> 2)

**Trigger.** User confirmed 10 qubits is fine, asked for a finer depth
step (10 -> 2) to smooth the figure's trend lines.

**What changed.** `plot_baseline_comparison.py`'s `DEPTHS` changed from
`range(10, 321, 10)` (32 circuits) to `range(10, 321, 2)` (156 circuits) --
5x more points. Ran in 13s (well within tolerance, no background run
needed). Regenerated, copied to `Writing/latex/figures/`. No dissertation
text referenced the exact circuit count (32), so nothing else needed
updating.

**Result.** Recompiled clean: 45 pages, 0 undefined refs/citations.

**Artifacts:** `scripts/plot_baseline_comparison.py` (`DEPTHS` step
changed), `results/figures/baseline_comparison.png` (regenerated, 156
circuits, copied to `Writing/latex/figures/`).

---

### 44. Depth step dialled back: 2 -> 5 (too dense at 2)

**Trigger.** User felt the step-2 sweep (156 circuits, entry 43) was too
dense/cluttered, asked to reduce to step 5.

**What changed.** `DEPTHS` changed to `range(10, 321, 5)` (63 circuits).
Regenerated, copied to `Writing/latex/figures/`. Recompiled clean: 45
pages, 0 undefined refs/citations.

**Artifacts:** `scripts/plot_baseline_comparison.py` (`DEPTHS` step = 5),
`results/figures/baseline_comparison.png` (regenerated, 63 circuits,
copied to `Writing/latex/figures/`).

---

### 45. Dropped `basic_opt+full_reduce` and `basic_opt+teleport_reduce`
    from the sweep/figure characterisation (kept in Chapter 4's evaluation)

**Trigger.** User asked to remove both composite baselines from ongoing
work (the sweep table and comparison figure).

**Scope decision, stated explicitly rather than assumed.** This request
followed several messages entirely about `sweep_baselines_100.py` and
`plot_baseline_comparison.py` specifically, so treated as scoped to that
characterisation work and `sec:methodology-baselines`'s prose -- not
Chapter 4's actual evaluation pipeline (`evaluate.py`/Table 4.1), which
still reports all four methods (`full_reduce`, `basic_opt+full_reduce`,
`teleport_reduce`, `basic_opt+teleport_reduce`) alongside the trained
agents, unchanged. If the user wants the composites dropped from Chapter
4's real results too, that's a separate, bigger decision not made here.

**What changed.** Both scripts' `METHODS`/imports trimmed to three:
`basic_opt`, `full_reduce`, `teleport_reduce` (the `roundtrip_only`
control kept in the sweep script). Both re-run; table and figure
regenerated with fresh 3-method numbers. `chapter03_methodology.tex`
rewritten: intro paragraph now frames the section as characterising
"three methods, each a fundamentally different rewriting strategy,"
explicitly notes the two composites are used in Chapter 4's evaluation
without needing separate characterisation here (their behaviour follows
from the three atomic methods); table shrunk to three rows; the "two
clusters" paragraph and metric-tradeoff discussion rewritten for three
methods; the paragraph specifically arguing for/against dropping the
composites (added in entry 40) removed entirely, since it no longer
applies once they're out of this section. The teleport_reduce
re-extraction-artefact caveat and the "which method serves which role"
paragraph (which still accurately lists all four Chapter-4-evaluated
methods) were left untouched.

**Result.** Chapters 1-7 word count: 14,552 (down from 14,721 -- net
simplification). Recompiled clean: 45 pages, 0 undefined refs/citations.

**Artifacts:** `scripts/sweep_baselines_100.py`,
`scripts/plot_baseline_comparison.py` (both trimmed, re-run),
`results/figures/baseline_comparison.png` (regenerated, 3 methods, copied
to `Writing/latex/figures/`). `Writing/latex/chapters/chapter03_methodology.tex`
edited.

---

## 2026-08-28 -- Full six-action-type implementation and Agent A all-rules smoke test

### 46. Implemented identity removal, pivot-boundary, and pivot-gadget; wired
    all six of Riu et al.'s action types into `env.py`; ran a smoke test

**Trigger.** User asked to retrain Agent A "with more zx rules" as a
30-60 minute smoke test (a sixth intervention attempt against the collapse
documented in entries 24-26). Initially scoped down to identity removal
alone (lowest-risk, since gadget fusion had already been tried and failed
as intervention #3); user then asked to include all six action types from
Riu et al. after being shown the tradeoff explicitly via AskUserQuestion
(pivot-boundary/pivot-gadget assumed at that point to need a new XZ/YZ-plane
spider representation, a multi-day undertaking), and chose "attempt all
six" anyway.

**Method.**
- `find_feasible_identity`/`apply_identity_removal` (`zx_utils.py`): reuses
  pyzx's own `check_remove_id`/`unsafe_remove_id`. Found and fixed a real
  bug: `unsafe_remove_id` can compose two Hadamard edges into a plain
  (non-graph-like) edge -- correct ZX algebra, but breaking
  `extract_circuit`'s graph-like requirement -- confirmed via a failing
  extraction on the first spot-checked circuit. Fixed by calling
  `zx.simplify.to_graph_like(g)` unconditionally inside
  `apply_identity_removal` itself.
- Before implementing pivot-boundary/pivot-gadget from scratch, checked
  whether pyzx already had them -- it does:
  `pyzx.rewrite_rules.pivot_rule.check_pivot_boundary`/`unsafe_pivot_boundary`
  and `check_pivot_gadget`/`unsafe_pivot_gadget`. These handle a
  non-Clifford spider by *gadgetizing* it (moving its phase onto a fresh
  leaf spider hung off a new ancilla) before pivoting, so the pivot step
  itself still only ever acts on a Pauli pair -- no new XZ/YZ-plane
  representation needed after all, contrary to the original time estimate.
- `find_feasible_pivot_boundary`/`find_feasible_pivot_gadget` added: both
  `check_*` functions are role-asymmetric in `(v, w)`, so both orderings of
  each Hadamard-connected pair are checked.
- Verified before wiring into training: unitary-equivalence-preserving
  (`zx.compare_tensors`) on spot-checked random circuits; graph-like
  preserved with no extra cleanup call needed (unlike identity removal);
  non-empty candidate counts on every adder bit-width used in this project
  (2, 3, 4, 5, 7, 8, 9), including 4 boundary + 35 gadget candidates on the
  raw 2-bit adder alone.
- Wired all six action types (LC, PIV, PIVB, PIVG, GF, ID) plus STOP into
  `env.py`: action-index construction, `step()` dispatch, and the
  observation encoding. Node action-type flags grew from 5 slots (4 used:
  LC/PIV/STOP/GF) to 7 (all used); `actor_node_feat_dim` 16->18;
  `edge_feat_dim` 6->9. `config.py` updated to match. Ran a 30-episode
  env+model smoke loop (random policy) with no exceptions; confirmed 5 of
  6 non-STOP kinds fired (GF didn't fire in that particular sample --
  structurally rarer -- but was already independently verified working in
  entry 24).
- Updated three stale spots in `chapter03_methodology.tex` that claimed
  only LC/pivot/STOP were implemented (already inaccurate before this
  entry, since GF had been live since entry 24): the observation-encoding
  paragraph, the actions-and-invalid-action-handling paragraph, and the
  "Action-space scope" limitation paragraph in the reproducibility-challenges
  section (reframed to describe the staged rollout honestly, deferring any
  claim about its effect on Agent A to the results chapters). Recompiled
  clean: 45 pages, 0 undefined refs/citations.
- Launched `train_agent_a_bits2_all_rules_smoke.py`: same bits=2-only,
  5x-entropy-coefficient configuration as `agent_A_bits2` and
  `agent_A_bits2_cflow_smoke` (entries 24, 30), 60,000 timesteps / 117
  updates, new run_name `agent_A_bits2_all_rules_smoke` so neither prior
  run's logs/checkpoint were overwritten.

**Result.** Ran to completion (117/117 updates, ~9 minutes wall time on
CPU). Entropy never collapsed: 3.930 at update 0, 3.956 at update 116,
effectively flat the entire run (min 3.862, max 3.974) -- unlike both
prior collapse runs, which had already reached near-zero entropy by the
equivalent point (update ~65-75). This rules out the specific always-STOP
collapse mode this and the two prior interventions were testing for.

However, `mean_reward` also never improved: flat around -0.87 to -0.90 the
whole run (range -1.11 to -0.73), never once positive or even near zero
across all 117 updates. So this is not the clean success it might look
like at first glance -- the policy stayed high-entropy/exploratory
throughout rather than collapsing, but showed no measurable learning
either, a different failure mode rather than a fix. The most likely
explanation, checked directly: the action space grew substantially. The
2-bit adder now presents 55 candidate actions at reset (1 STOP + 1 LC + 12
PIV + 4 PIVB + 35 PIVG + 2 ID), dominated by pivot-gadget candidates,
versus a much smaller space with just LC/PIV/GF previously. A ~50-action
categorical policy needs more than 60,000 timesteps (117 updates) to show
learning signal above noise, especially with PPO's per-update batch size
unchanged -- this smoke-test budget was calibrated for a much smaller
action space in entries 24/30, not this one.

**Open question, not resolved here.** Whether a longer training budget
(matching the full ~15-17 hour runs used for R/P/A's actual training)
would let the policy learn something useful in this expanded space, or
whether the sheer volume of low-value PIVG candidates on this circuit
family needs pruning/deprioritising first, is untested. Not run here,
consistent with the smoke-test framing (30-60 min budget) this
intervention was scoped to.

**Artifacts:** `src/circopt_adder/zx_utils.py` (`find_feasible_identity`,
`apply_identity_removal`, `find_feasible_pivot_boundary`,
`apply_pivot_boundary`, `find_feasible_pivot_gadget`,
`apply_pivot_gadget` added; module docstring updated),
`src/circopt_adder/env.py` (all six action types wired into
action-index construction, `step()`, and `_build_observation`;
`_ACTION_TYPE_OFFSET` mapping added), `src/circopt_adder/config.py`
(`actor_node_feat_dim` 16->18, `edge_feat_dim` 6->9),
`scripts/train_agent_a_bits2_all_rules_smoke.py` (new),
`results/logs/agent_A_bits2_all_rules_smoke_train_log.csv`,
`results/checkpoints/agent_A_bits2_all_rules_smoke.pt`.
`Writing/latex/chapters/chapter03_methodology.tex` edited (three
locations).

---

### 47. Agent R/P out-of-distribution generalisation sweep (qubit count and
    gate count, smaller and larger than training)

**Trigger.** User asked to test the trained Agent R and Agent P checkpoints
on structured_random/pure_random circuits with qubit and gate counts both
smaller and larger than each agent's training distribution, to see how they
generalise.

**Blocker found before this could run at all.** `env.py`/`config.py` have
been edited in place three times since R/P trained (gadget fusion in entry
24; identity removal, pivot-boundary, pivot-gadget in entry 46), each time
widening the observation dims with no versioned snapshot in between (these
files were never committed at each stage -- git history only has one very
old pre-training commit). Confirmed directly against the checkpoints'
own tensors (`input_proj.weight` shape `[32,16]`, `convs.*.lin_edge.weight`
shape `[32,6]` for both) that R/P were trained with `actor_node_feat_dim=16`,
`edge_feat_dim=6` -- the current env.py now builds 18/9-dim observations,
which would either crash on load or route new action types (GF/ID/PIVB/PIVG)
through weight columns that were always fed zero during R/P's actual
training and therefore never received gradient (an `nn.Linear` column whose
input is identically zero across every training example gets no gradient
signal and never leaves random initialisation). Built `legacy_env_rp.py`
(`LegacyRPEnv`, subclasses `ZXOptEnv`, overrides only `_build_observation`)
to reproduce exactly the STOP/LC/PIV-only, 16/6-dim action space R/P
actually learned under. Verified: both checkpoints load into it with no
shape errors, and a live rollout completes normally.

**Method.** Two 1-D sweeps, holding one size axis at Agent R's training
value while varying the other: qubit sweep (2,3,5,8,12,15,20,30 qubits,
gates fixed=70), gate sweep (10,30,70,100,150,200,300,500 gates, qubits
fixed=5). Both structured_random (Agent R's tuned recipe) and pure_random
(uniform recipe) generated at every point, n=3 circuits/cell, each run
through `_light_preprocess` exactly as training circuits are. Both agents
evaluated best-of-10 (`LegacyRPEnv`); `teleport_reduce` and
`basic_opt+teleport_reduce` included as deterministic reference points.

**Result.** Effect sizes are small overall and both agents track
`basic_opt+teleport_reduce` closely across most of the range -- neither
consistently beats it by a wide margin, nor falls far short of it. Two
genuine patterns, not just noise:
- **Qubit sweep**: reduction is concentrated entirely at small/dense sizes
  (2-3 qubits, i.e. 70 gates packed onto very few qubits) -- up to 20% on
  pure_random @ 2 qubits, 13% on structured_random @ 2 qubits -- and flattens
  to ~0% for both agents *and* both baselines from 12 qubits onward. At 70
  gates spread across 12+ qubits the circuits are sparse enough that
  `_light_preprocess` (which every circuit already passes through once)
  leaves little to no further LC/PIV-feasible structure for anything to
  find -- confirmed this is a real structural floor, not an agent-specific
  failure, since the two deterministic baselines hit exactly the same 0%
  at the same sizes.
- **Gate sweep**: agent_R keeps finding improvement out to the largest
  tested sizes (300, 500 gates, well past both agents' training ranges) on
  both circuit families, gently outperforming agent_P and
  `basic_opt+teleport_reduce` at those two points specifically, rather than
  degrading with distance from training the way a naive overfitting story
  would predict.

**Caveat, stated plainly.** n=3 circuits/cell is small; several rows show
agent_R and agent_P moving in lockstep with `basic_opt+teleport_reduce`
almost exactly (e.g. the shared dip at 100 gates on both families), which
given all four methods including the two independent deterministic
baselines move together, reads as circuit-instance noise from the small
sample, not a shared method effect. This sweep is exploratory, not
sized for a dissertation-grade generalisation claim.

**Artifacts:** `src/circopt_adder/legacy_env_rp.py` (new),
`scripts/evaluate_rp_generalization.py` (new),
`results/logs/rp_generalization_eval.csv`,
`results/figures/rp_generalization.png`.

---

### 48. Chapter 4 re-anchored to the current evaluation snapshot; total-gate-count
    table added (Objective 3 alignment)

**Trigger.** User revised Chapter 1's objectives; new Objective 3 promises
evaluation of "total gate count and two-qubit gate count", but Chapter 4
reported only two-qubit results. User confirmed adding total gate count
"to get the full picture".

**Snapshot inconsistency found and resolved.** The on-disk
`evaluation_results.csv` was regenerated mid-project (entry 36's rerun,
2026-08-28) with freshly drawn random-circuit families (pyzx's generators
are unseeded -- the documented RNG-reproducibility gap), so its
structured-random/pure-random numbers no longer matched Table 4.1, which
had been frozen from an earlier run. Checked per-circuit: the adder rows
in the current CSV are *identical* to Table 4.2 (deterministic circuits,
deterministic methods), and the old Table 4.1 adder mean (-58.1%) was
actually inconsistent with its own Table 4.2's per-circuit data (which
gives -60.5%) -- i.e. the old 4.1 was computed from a third, even earlier
state. Decision: make the current CSV the single authoritative snapshot.
Updated Table 4.1's numbers, the dependent prose ranges in ch4 (31--128%
-> 19--152%; agent range -1.8..5.3 -> -4.2..13.7; 31.3% -> 31.2%), the two
quoted numbers in ch5 (Sections 5.2/5.5), and copied the matching
regenerated figures (comparison_barplot, adder_scaling, both Aug 28) over
the stale Aug 25 copies in Writing/latex/figures.

**What was added.** New Chapter 4 section "Total gate count reduction"
(`sec:results-gate-reduction`, Table `tab:results-gates-by-family`): mean
% total-gate reduction, same circuits/protocol/selection as Table 4.1,
with prose noting (a) best-of-10 selection is still on 2Q count (total
gates reported for the selected circuit, not separately optimised), (b)
the baselines are far more competitive on total gates for random families
(basic_opt+full_reduce 21-42%, slightly ahead of the agents on
structured-random) but still increase total gates on adders/real-world,
(c) the agents are the only methods positive on every family on this
metric. Also reconciled Chapter 7's "Revisiting the objectives" to the
user's renumbered objective list (old root-cause Objective 4 folded into
new Objective 4's analysis narrative; "replicating" -> "inspired by").

**Result.** Compiles clean, 52 pages, 0 undefined refs. Tables 4.1/4.2
are now mutually consistent for the first time.

**Artifacts:** `Writing/latex/chapters/chapter04_results.tex` (table
numbers updated + new section), `chapter05_analysis.tex` (two numbers),
`chapter07_conclusion.tex` (objectives rewritten),
`chapter01_introduction.tex` (typo fix, objective-chapter mapping),
`Writing/latex/figures/comparison_barplot.png`, `adder_scaling.png`
(refreshed to current snapshot).

---

### 49. Chapter 4 rebased onto the three atomic baselines (basic_opt,
    full_reduce, teleport_reduce); evaluate_final.py run

**Trigger.** User: "in the results, comparing the agents with the
baseline, compare them with only basic optimise, full reduce and teleport
reduce" -- completing the propagation of the entry-35/38 framing decision
into Chapter 4, which still used full_reduce / basic_opt+full_reduce.

**Method.** The existing evaluation CSV had no standalone basic_opt rows
and its random circuits are unrecoverable (unseeded generators), so a
fresh run was required: `scripts/evaluate_final.py`, same 47-circuit
evaluation set and best-of-10 protocol as evaluate.py, methods =
basic_opt / full_reduce / teleport_reduce + the three trained agents.
Agents run through LegacyRPEnv (all three checkpoints predate the
entry-46 action-space expansion; 16/6-dim weights confirmed). Output:
`results/logs/evaluation_results_final.csv`, now the authoritative
Chapter 4 snapshot; comparison_barplot/adder_scaling regenerated from it.

**Result -- the cleanest statement yet of the session's central finding.**
2Q mean reductions: basic_opt 4.7/2.4/31.2/2.0 across the four families;
agents 4.7-11.5/2.4-11.7/31.2/2.1 (Agent P equals basic_opt exactly on
both random families; all agents equal basic_opt exactly, per-circuit, on
the held-out adders: 44/77/88/99); full_reduce -34.3 to -96.4 everywhere;
teleport_reduce exactly 0.0 on the (already-preprocessed) random families
-- idempotency confirmed in the official evaluation itself. Total gates:
basic_opt and agents ~39-45% on random families, 22.6% adders, ~11.8%
real-world; full_reduce/teleport_reduce negative on adders/real-world.

**Text updated.** ch4: intro (baseline set + legacy action-space note),
Table 4.1, its prose (rewritten around the three baselines' divergent
behaviour), total-gate table + prose, adder-detail table (now includes
basic_opt column, identical to agents), summary (headline: agents'
results identical or nearly identical to basic_opt on both metrics and
all families). ch5: intro findings, section retitled "Why full_reduce
makes circuits worse", quoted ranges, sec 5.3 opening ("no better than
matching basic_opt"), sec 5.5 closing (basic_opt reaches the same adder
counts deterministically -- strengthens the search-coverage argument).
ch7 Objective 3 paragraph aligned. Compiles clean: 52 pages, 0 undefined.

**Artifacts:** `scripts/evaluate_final.py` (new),
`results/logs/evaluation_results_final.csv`,
`results/logs/comparison_barplot.png`, `adder_scaling.png` (both
regenerated + copied to Writing/latex/figures), chapters 04/05/07 edited.

---

### 50. Self-play terminal-bonus baseline: implementation, 200k partial
    success, 600k scale-up stopped early with a negative result

**Trigger.** After entries 46-49 established that every external terminal
baseline collapses to a tie with the light-preprocessed starting circuit
(so "do nothing" is reward-optimal), the user asked to try a self-play
alternative, then to scale it up, then to stop the scale-up and take
stock ("can you stop the training and give me the results for now").

**Implementation.** `src/circopt_adder/self_play_env.py` (`SelfPlayEnv`):
persistent per-circuit best-ever metric folded into the terminal-bonus
`min()` as a ratchet -- first visit identical to the old design; once any
episode beats a circuit's past best, that becomes the bar, making
inaction penalised (about -0.1 per gate short, at max_compression=10) and
only further improvement rewarded. Keyed by exact gate sequence;
meaningful only for Agent A's three fixed, repeating training circuits
(R/P generate fresh circuits per episode -- would degrade to a no-op).
Ratchet verified by forced-improvement unit check before training.

**Run 1 (agent_A_adder_selfplay_200k, 200,000 timesteps, {2,3,5}-bit mix,
all-six-action env, ent_coef=0.05).** Training reward never exceeded
-0.79; entropy never collapsed (final ~4.3 band then variable). But the
checkpoint produced the project's first genuine beats of basic_opt:
2-bit adder 22->21 2Q, 3-bit 33->32, verified reproducible at ~1/40
stochastic samples; nothing on bit-widths 4-9 (0/40 each).

**Run 2 (agent_A_adder_selfplay_600k, tripled budget; stopped at user
request at update 722/1171 = ~370,000 timesteps, checkpoint from update
~719).** Reward climbed from -1.2 to a hard plateau at exactly ~-0.10
from update ~576 onward while entropy fell from ~4.4 to ~0.9-1.1. That
-0.10 plateau is mechanistically legible: it is the signature of the
ratchet sitting one gate (1/max_compression = 0.1) below what the policy
reliably reproduces -- i.e. the ratchet found and held the ~1-gate
improvements, but PPO never consolidated them into the policy.
Evaluation of the stopped checkpoint: **0/40 samples beat preprocessing
on every bit-width (2,3,4,5,7,8,9)** -- the scale-up not only failed to
extend the wins to larger circuits, the mid-training checkpoint cannot
reproduce the 2-/3-bit wins the 200k checkpoint could occasionally hit.

**Honest reading.** The self-play ratchet does its half of the job
(discovering and holding a harder target); the missing half is
consolidation -- turning rarely-found better trajectories into reliably
reproduced policy behaviour. That is precisely what self-imitation
learning adds (an imitation loss on best-so-far trajectories) and this
design lacks. More timesteps alone made the policy narrower (entropy
~1.0) without making it better, which is consistent with PPO's on-policy
gradient washing out a ~1/40-frequency behaviour. n=40 caveats apply to
the per-checkpoint hit rates, but the direction (no extension to larger
widths; no improvement in reproduction rate) is clear.

**Artifacts:** `src/circopt_adder/self_play_env.py`,
`scripts/train_agent_a_selfplay.py`,
`scripts/train_agent_a_selfplay_600k.py`,
`results/logs/agent_A_adder_selfplay_200k_train_log.csv` + checkpoint,
`results/logs/agent_A_adder_selfplay_600k_train_log.csv` + checkpoint
(724 updates logged; run intentionally stopped early).

---

### 51. Expert-iteration experiment: search-distillation partially succeeds,
    and produces the project's first genuine learned transfer

**Trigger.** Following entry 50's diagnosis (search discovers, PPO fails
to consolidate) and the random-init control (an untrained policy's
best-of-10 matches a trained agent's on random circuits), the user chose
the expert-iteration design ("i'll choose option 3").

**Method** (`scripts/expert_iteration_agent_a.py`, current six-action env,
{2,3,5}-bit light-preprocessed adders): (1) 1,000 uniform-random rollouts
per circuit, keeping winning trajectory prefixes cut at each episode's
best point with a STOP label appended there; (2) behaviour-clone a fresh
policy (200 epochs, cross-entropy on action nodes); (3) 200 policy-guided
rollouts per circuit; better finds trigger a merged round-2 distillation;
(4) evaluate greedy + 40-sample hit rates on training circuits and on the
held-out 4-/7-bit adders search never touched.

**Results.**
- Random search: 2-bit 22->21 (19 distinct winners), 3-bit 33->32 (2),
  5-bit 55->54 (2) -- the 5-bit improvement is the first time anything in
  this project beat basic_opt on that circuit.
- Guided search after one distillation round: 2-bit -> **20** and 5-bit ->
  **53** (both new project bests, found with 5x fewer episodes than random
  search), 3-bit hit rate up ~35x (2/1000 -> 14/200). The expert-iteration
  loop (search -> distill -> better search) demonstrably works.
- Final policy, training circuits: greedy reproduction FAILED everywhere
  (22/33/55); sampled hit rates 3/40, 3/40, 5/40 (vs ~1/40 for PPO
  checkpoints and 2-19/1000 for uniform random); the round-2 bests (20,
  53) were not reproduced within 40 samples.
- Final policy, held-out circuits (no training data from them): 4-bit
  found **42** vs basic_opt's 44 in 7/40 samples; 7-bit found 76 vs 77 in
  1/40. Uniform-random control on the same circuits: **0/40 on both**.
  This is the first genuine learned cross-circuit transfer observed in the
  entire project -- a distilled search bias that generalises to unseen
  bit-widths, where every prior method (all PPO checkpoints, basic_opt,
  40-sample random search) found nothing.
- Distillation accuracy plateaued low (36/203, 40/245): partly inherent
  label conflict (multiple winning trajectories diverge from the same
  state), partly plausible capacity limits of the 32-channel single-logit
  policy head.

**Verdict.** Deterministic consolidation (the original success criterion)
failed -- the network cannot greedily replay taught trajectories, locating
part of the problem in the network/representation rather than the RL
signal alone. But as a *search prior* the distilled policy is strictly and
substantially better than uniform random and than every RL-trained
checkpoint: higher hit rates on trained circuits, new bests when used to
guide search, and -- uniquely -- transfer to held-out circuits confirmed
against a 0/40 random control. The accurate framing: expert iteration
turned the policy from a failed final optimizer into a working search
prior, mirroring how the field actually deploys these systems (Riu et
al.'s 1,000-tries protocol; AlphaZero's policy-guided search).

**Artifacts:** `scripts/expert_iteration_agent_a.py`,
`results/checkpoints/agent_A_expert_iteration.pt`, full phase log in the
session scratchpad (expert_iteration.log).

---

### 52. Expert-iteration on the R/P families: distillation loses to the
    untrained-prior control -- completing a structured-vs-random dichotomy

**Trigger.** User asked whether entry 51's recipe would also improve
Agents R and P, then chose to run it ("run it").

**Method** (`scripts/expert_iteration_agents_rp.py`): per family
(structured_random = Agent R's recipe, pure_random = Agent P's), 150
fresh circuits x 15 uniform-random rollouts, best winning trajectory per
circuit kept (prefix to best point + STOP label); behaviour-clone a fresh
policy (60 epochs); evaluate best-of-10 on 30 further fresh circuits
against (a) the untrained random-init control -- the honest bar, per the
entry-51 controls -- and (b) a single basic_opt pass. No guided round.

**Results (mean 2Q reduction over 30 fresh circuits; head-to-head):**
- structured_random: basic_opt -1.0%, distilled 8.2%, untrained 11.6%;
  untrained better on 15 circuits vs distilled's 5 (10 ties).
- pure_random: basic_opt -0.7%, distilled 2.7%, untrained 3.3%;
  untrained better on 9 vs 3 (18 ties).
Search supply confirmed abundant (130/150 and 64/150 circuits improved
during collection), so the failure is not data scarcity.

**Verdict.** No -- the recipe does NOT transfer to the random families:
the distilled policies are strictly worse than the untrained control on
both. Distillation narrows the policy's action distribution, and on
heterogeneous random circuits the diversity it gives up is worth more
than the bias it gains; on the structured adder family (entry 51) the
opposite held (7/40 vs 0/40 on a held-out circuit). Together the two
experiments form a clean dichotomy that answers the dissertation's own
research question at a deeper level than the original ablation design:
training-circuit structure does not merely modulate how well a learned
optimiser performs -- it determines whether learning helps at all.
Structured, self-similar families (rare, recurring improvement motifs):
a distilled prior transfers and beats blind search. Unstructured random
families (frequent, heterogeneous improvements): blind search diversity
is the better prior, and learning subtracts value.

**Artifacts:** `scripts/expert_iteration_agents_rp.py`,
`results/checkpoints/agent_expert_iteration_structured_random.pt`,
`agent_expert_iteration_pure_random.pt`, phase log
expert_iteration_rp.log (session scratchpad).

---

### 53. Dissertation restructured around the two-phase narrative
    ("inspired by Riu -> diagnosed failure -> search distillation")

**Trigger.** User: "i want to go to this direction, first my method was
inspired by riu but it failed so this was done like that."

**What changed.** The entries 46-52 work is now in the dissertation as an
explicit second experimental phase, framed as a designed response to the
first phase's diagnosed failure rather than an appendix of extra
experiments. New sections: ch3 `sec:methodology-phase2` (self-play
ratchet, the two controls, expert-iteration pipeline; cites
silver2017mastering, which re-enters the printed bibliography); ch4
`sec:results-phase2` (controlled-comparison table
`tab:results-phase2` + three observational findings, including the 7-bit
held-out 76-vs-77 within the standard protocol); ch5
`sec:analysis-phase2` (discovery-vs-consolidation split; the
structured-vs-random dichotomy as the deeper answer to the RQ).
Connective edits: ch1 contributions + structure paragraph; ch6 practice
recommendation confirmed by the untrained control, fourth knowledge
finding, phase-2 scale caveat in limitations, future-work item 4 replaced
(the "unexplained Agent A strength" question is now RESOLVED by the
untrained-network control -- replaced with the consolidation-gap /
self-imitation direction); ch7 Objective 4/5 paragraphs and Summary.

**Status.** Compiles clean (54 pages, 0 undefined refs). Word count
(chapters 1-7, approximate counter): ~15,650 -- now ABOVE the 15,000
target by ~650 words; trim decision pending with the user.

---

### 54. ATQ-style scaling figure; best-of-60 capture beats basic_opt on
    EVERY adder bit-width 2-9, four of them for the first time

**Trigger.** User shared AlphaTensor-Quantum's adder scaling figure
(Toffoli count vs size) and asked for the analogous figure for Agent A.

**Method.** `scripts/plot_agent_a_scaling_atq_style.py`: two-qubit gate
count vs bit-width 2-9; series = raw Cuccaro (16n), basic_optimization
(11n), and the expert-iteration distilled policy's best-of-60 (1 greedy +
59 sampled, light-preprocessed start -- the standard protocol at a 60-
sample budget). Panel (a) full comparison; panel (b) zoom on the two
near-overlapping optimised lines with trained {2,3,5} vs held-out widths
marked -- deliberately mirroring ATQ's Fig (adders panel) visual grammar.

**Result (new empirical bests in bold):** agent vs basic_opt per width:
21/22, 32/33, 43/44, 54/55, **65/66 (6-bit, first time this width was
ever tested)**, **75/77 (7-bit -- two below basic_opt, better than the
previous best 76)**, **87/88 (8-bit, first improvement ever)**,
**98/99 (9-bit, first improvement ever)**. I.e. at a 60-sample budget the
distilled prior beats basic_opt on every width tested, including all
held-out widths; the earlier 0/40 results on 8/9-bit were a budget
effect, not an absence of reachable improvements. Raw numbers saved to
results/figures/agent_a_scaling_atq_style.json alongside the PNG.

**Honest scale note.** The figure's visual grammar matches ATQ's, but the
magnitudes do not: ATQ's optimised lines sit at roughly half the
before-line (their metric: Toffoli count, vs Gidney's 4n construction);
ours sit ~1-2 gates (~1-2%) below basic_opt on a different metric
(two-qubit count). Any dissertation use must keep the axes/metric
distinction explicit (see entry: AlphaTensor comparison discussion).

**Artifacts:** `scripts/plot_agent_a_scaling_atq_style.py`,
`results/figures/agent_a_scaling_atq_style.png` + `.json`.

---

### 55. AlphaTensor-Quantum positioning added to Chapter 6

**Trigger.** User asked whether scaling training could find a
Gidney-equivalent adder as AlphaTensor-Quantum did; after the assessment
(representational threshold, not a training one), user: "add this to the
dissertation".

**What changed.** New ch6 section `sec:disc-atq` ("Positioning against
AlphaTensor-Quantum"): the 4n-vs-10n T-count gap on adders is a search-
space boundary, not a compute gap -- Gidney's halving needs ancilla +
measurement-based uncomputation, present in ATQ's gadgetisation action
space and absent by construction from this project's unitary
rewrite-and-extract pipeline; the reachable unitary frontier is ~8n
(full_reduce measured at 7-8.7 T/bit at severe 2Q cost). Framed as a
restatement of the central finding at framework level: what a learned
optimiser can achieve is fixed first by what its action space admits.
Future-work list extended from four to five directions (ancilla +
measurement gadgets + T-count objective as the route to testing whether
search-distillation can rediscover Gidney-style constructions). ch6 intro
updated to list the new section. gidney2018halving enters the printed
bibliography (now 36 entries). Compiles clean: 54 pages, 0 undefined.
Word count remains over target; trim plan still pending (memory:
word-count-trim-plan).

**Artifacts:** `Writing/latex/chapters/chapter06_discussion.tex`.

---

### 56. Distilled agent on real-world adder/arithmetic benchmarks:
    3 of 8 beaten, 5 ties, none worse

**Trigger.** User asked whether real-world circuits use many adders and
whether the agent can optimise them; the benchmark suite already contains
eleven adder/modular-arithmetic circuits, so the question was answered
empirically.

**Method.** Expert-iteration distilled checkpoint, best-of-30 (1 greedy +
29 sampled), standard env/protocol, vs basic_optimization, on eight
tractable adder-family benchmark circuits (the two ~900-gate giants
skipped for time).

**Results (2Q count: initial / basic_opt / agent, hits below basic_opt):**
- vbe_adder_3 (VBE): 70/58/58, tie (0/30)
- rc_adder_6 (ripple-carry): 93/81/81, tie (0/30)
- mod5_4: 28/27/27, tie (0/30)
- mod_mult_55 (modular multiplier): 48/48/**45**, BEATS (3/30)
- mod_red_21: 105/105/105, tie (0/30)
- QFTAdd8 (QFT adder): 184/184/**183**, BEATS (30/30 -- every sample)
- Adder8: 243/243/243, tie (0/30)
- qcla_com_7 (carry-lookahead): 186/174/**172**, BEATS (5/30)

**Reading.** 3/8 real-world circuits improved beyond basic_opt, never
worse. The pattern defeats the mid-run hypothesis that the learned bias is
Cuccaro-construction-specific: the closest structural cousin (rc_adder_6)
only tied, while the wins came on structurally different designs
(modular multiplier, QFT adder, carry-lookahead). Also notable: the wins
cluster on circuits where basic_opt itself found little or nothing.

**Attribution caveat (unresolved).** No untrained-network control was run
on these circuits, so per the project's own standard (entries 51-52) the
wins cannot yet be attributed to the distilled prior rather than to
generic best-of-30 search -- QFTAdd8's 30/30 hit rate in particular
suggests an easy move any sampler may find. The tool-level claim ("the
method improves 3/8 real circuits over basic_opt in minutes of CPU, never
hurting") stands regardless; the learning-level claim needs the control.

**Artifacts:** inline run, log at scratchpad/realworld_adders.log; no
repo script (rerun recipe recorded here).

---

### 57. Untrained control on the real-world adder circuits: generic search
    beats basic_opt on 8/8, and beats the distilled agent 5-2-1

**Trigger.** User: "do the untrained control run" (attribution check for
entry 56's real-world wins).

**Method.** Identical protocol to entry 56 (same 8 benchmark circuits,
best-of-30, same seed), with a randomly initialised, untrained
ActorCriticGNN in place of the distilled checkpoint.

**Results (2Q: basic_opt / distilled agent / untrained control):**
- vbe_adder_3: 58 / 58 / **56** -- control wins
- rc_adder_6: 81 / 81 / **80** -- control wins
- mod5_4: 27 / 27 / **25** -- control wins
- mod_mult_55: 48 / **45** / 46 -- agent wins
- mod_red_21: 105 / 105 / **102** -- control wins
- QFTAdd8: 184 / 183 / 183 -- tie
- Adder8: 243 / 243 / **241** -- control wins
- qcla_com_7: 174 / **172** / 173 -- agent wins

**Two conclusions.** (1) Attribution: entry 56's real-world wins were
generic best-of-N search, not the distilled prior -- head-to-head the
untrained control beats the distilled agent 5-2 with 1 tie, and the
distilled policy's narrowed diversity costs it on off-family circuits,
extending entry 52's random-family finding to real-world circuits. The
learning-level transfer claim remains supported ONLY within the trained
Cuccaro family (entry 51/54's 0/40-vs-7/40 and all-widths results). (2) A
new tool-level finding: untrained-network best-of-30 search beats
basic_opt on ALL EIGHT real-world arithmetic benchmarks (by 1-3 2Q gates,
never worse) in minutes of CPU per circuit -- the strongest evidence yet
that cheap stochastic ZX search is a useful post-basic_opt optimisation
layer for real circuits, independent of any learning.

**Artifacts:** scratchpad/realworld_adders_control.log (protocol recorded
in entry 56).

---

### 58. Mixture-sampler test (lambda=0.5), stopped early at user request:
    per-step blending interpolates, it does not dominate

**Trigger.** Entry 57's dichotomy suggested blending the distilled policy
with uniform randomness might keep both arms' strengths; user approved
the test, then asked to stop and conclude after 7 of 16 planned rows.

**Results obtained (Cuccaro adders, best-of-30, preproc / distilled /
untrained / mixture):**
- 2-bit: 22 / 21 / 22 / 21 -- mixture matches distilled's win
- 3-bit: 33 / 33 / 32 / 32 -- mixture matches untrained's win
- 4-bit: 44 / 43 / 44 / 43 -- mixture matches distilled's win (held-out)
- 5-bit: 55 / 53 / 55 / 54 -- mixture beats preproc, trails distilled
- 6-bit: 66 / 65 / 66 / 66 -- mixture misses, distilled wins
- 7-bit: 77 / 76 / 77 / 77 -- mixture misses, distilled wins
- 8-bit: 88 / 86 / 88 / 87 -- mixture beats preproc, trails distilled
(Part B, real-world circuits, not reached.)

**Conclusion.** Per-step lambda=0.5 mixing does NOT dominate both arms;
it interpolates. It preserved or recovered wins on the small widths
(2/3/4-bit) but consistently trailed the pure distilled arm on the larger
ones (5-8 bit). The mechanism is arithmetic: a win requiring the prior to
be followed for k consecutive steps survives per-step mixing with
probability ~lambda^k, so exactly the rare long guided sequences that
distinguish the distilled arm on larger circuits are exponentially
suppressed. The evidently better design, not run here, is
episode-level portfolio sampling (allocate whole episodes to each pure
arm, e.g. 15 distilled + 15 random per circuit), which by construction
approximates the max of both arms at equal budget and needs no new
training -- recorded as the practical recommendation / future-work
refinement in place of per-step blending. Incidentally the distilled arm
itself reconfirmed at best-of-30: 86 on the 8-bit (a new best for that
width, below entry 54's 87).

**Artifacts:** `scripts/mixture_sampler_test.py`,
scratchpad/mixture_test.log (partial run).

---

### 59. Tier-1-max complete: chaining sets new bests nearly everywhere;
    every real-world circuit improved beyond basic_opt

**Method.** Entry 58 context; scripts/tier1_max_eval.py. Four arms
(distilled / per-step mixture / episode portfolio / untrained control),
iterated restarts (up to 3 rounds, continuing only on improvement),
N=60/round on Cuccaro 2-9, N=30/round on the 8 real-world circuits.

**Part A (Cuccaro, start -> best arm):** 2b 22->20 (mixture/portfolio/
untrained), 3b 33->30 (distilled), 4b 44->42 (portfolio), 5b 55->52
(mixture), 6b 66->63 (distilled), 7b 77->73 (distilled), 8b 88->84
(distilled), 9b 99->96 (distilled+portfolio). New all-time bests on six
of eight widths; distilled+chaining dominates from 6-bit up; untrained
scores only at 2-bit.

**Part B (real-world, basic_opt -> best arm):** vbe_adder_3 58->52
(distilled), rc_adder_6 81->77 (portfolio), mod5_4 27->18 (portfolio,
-33%), mod_mult_55 48->40 (untrained), mod_red_21 105->99
(distilled+mixture), QFTAdd8 184->182 (portfolio), Adder8 243->239
(mixture), qcla_com_7 174->169 (untrained). ALL EIGHT improved beyond
basic_opt by some chained arm; per-row winners rotate across all four
arms (high draw variance), so no single-arm attribution holds on
real-world circuits; the robust findings are (1) chaining is the
dominant lever, multiplying single-round gains several-fold, and (2) the
portfolio is the most consistent single arm.

**Also.** The overnight queue wrappers (distill sweep, trio v2) died
silently without launching; relaunched with PID-based handoff. Sweep now
running; trio v2 queued behind it.

**Artifacts:** scripts/tier1_max_eval.py, scratchpad/tier1_max.log.

---

### 60. Real-world evaluation set re-scoped to the adder/arithmetic subset;
    full suite moved to a new Appendix C

**Trigger.** User: "can you substitute them all with only circuits that
have adders in them" -- aligning the evaluation with the revised
Objective 3 ("real-world benchmark circuits that contain adder and
arithmetic structure").

**Safety check first.** Re-aggregated evaluation_results_final.csv over
the 11 arithmetic circuits (names matching add/qcla/mod) vs the other
22: agents 5.6-5.7% vs basic_opt 5.5% (arithmetic) and 0.3-0.4% vs 0.3%
(other) -- essentially all real-world improvement available to ANY method
lives in the arithmetic subset, and the substitution does not change the
agents' standing relative to any baseline, so the selection-bias concern
has a direct numerical answer. No re-running needed; per-circuit rows
already existed.

**What changed.** ch4: intro (primary set = 11 arithmetic circuits, full
suite in appendix), Table 4.1 caption (n=47 -> n=25) and real-world
column (agents 2.1 -> 5.6-5.7; basic_opt 2.0 -> 5.5; full_reduce -96.4 ->
-49.8; teleport 1.6 -> 4.2), total-gate table column, dependent prose
(34--96% -> 34--93%). ch3: evaluation-protocol paragraph (also fixed a
STALE baseline sentence still naming full_reduce/basic_opt+full_reduce
as the evaluated pair), generators-table cell. ch5: quoted range. New
appendices/appendixC.tex (label app:fullsuite): full-suite vs arithmetic
vs non-arithmetic aggregate table + explanation; included from main.tex
(appendices A/B remain reserved for the template's ethics documents).
comparison_barplot regenerated with the real-world category filtered to
the arithmetic subset. Compiles clean, 55 pages, 0 unresolved refs.

**Artifacts:** chapters 03/04/05 edited, appendices/appendixC.tex (new),
main.tex (include), figures/comparison_barplot.png (regenerated).

---

### 61. Distillation parameter sweep: capacity x label-dedup interaction
    CLOSES the consolidation gap (first greedy reproduction ever)

**Method.** scripts/distill_param_sweep.py (entry context 51/58): shared
1,000-rollout random search on the {2,3,5}-bit adders, then a 3x2 grid,
gnn_channels {32,64,128} x labels {all winners, one-per-circuit dedup},
300 BC epochs each; scored on fit, greedy reproduction, best-of-20.

**Results (per cell: 2b/3b/5b greedy | hits per 20):**
- 32/all:   fail/fail/fail | 0,1,0     - 32/dedup:  fail/fail/fail | 0,5,7
- 64/all:   fail/fail/fail | 0,1,0     - 64/dedup:  fail/fail/fail | 2,6,0
- 128/all:  fail/fail/fail | 0,0,1
- **128/dedup: 21/32/54 greedy -- reproduction on ALL THREE | 8,9,2;
  sampling exceeds targets (31 on 3b, 53 on 5b within 20 samples)**

**Verdict.** Neither factor suffices alone: capacity without clean labels
does nothing (128/all as weak as 32/all), and clean labels without
capacity lift hit rates but never greedy (32/64 dedup). Together they
close the consolidation gap completely -- the first policy in this
project that deterministically replays taught improving trajectories,
and the first whose GREEDY single rollout beats basic_opt. This converts
ch5's "points at the network and representation" from a plausible
reading into a tested mechanism: consolidation requires conflict-free
supervision AND sufficient capacity jointly. Caveats: one run per cell,
20-sample evals, three training circuits; the winning checkpoint is
saved (distill_sweep_ch128_dedup.pt).

**Artifacts:** scripts/distill_param_sweep.py, six checkpoints
results/checkpoints/distill_sweep_ch*_{all,dedup}.pt,
scratchpad/distill_sweep.log.

### 62. Agent A v3: the winning recipe end to end -- broad shallow
transfer, but per-circuit consolidation does not survive a shared
multi-width policy (2026-08-30)

**What ran.** scripts/train_agent_a_v3.py, the entries-51--61 synthesis
in one pipeline: chained random search (150 episodes/round, up to 3
rounds) on widths {2,3,5,6,7,8,9} with 4 and 10 fully held out;
dedup-by-construction labels (one best trajectory per width x round);
128-channel GATv2 distillation (300 epochs); one policy-guided chained
round (100 episodes/round); re-distillation; evaluation per width
(greedy, best-of-20, chained-greedy-20). Checkpoint:
results/checkpoints/agent_A_v3_128dedup.pt.

**Phase 1 (blind chained search).** Only the small widths paid off:
2b 22->20 (8 pairs), 5b 55->53 (15 pairs); 3/6/7/8/9-bit found nothing
at this budget (their single improvements are ~1-in-500+ draws). Dataset:
23 pairs. First distillation converged at only 9/23 -- the mixed
two-width dataset already resists the full reproduction the per-circuit
sweep achieved.

**Phase 3 (guided chained search) -- the strong result.** The thin
2b/5b prior transferred to five of seven widths and added 90 pairs:
3b 33->31 (a NEW all-time best, one below the long-standing 32, on a
width blind search missed entirely), 6b 66->63 and 9b 99->96 (both
matching tier-1-max bests), 7b 77->74, 2b/5b re-confirmed; only 8b
stayed empty at 100 episodes/round. Re-distillation on all 113 pairs
converged at 24/113.

**Phase 4 (evaluation).** A uniform shallow win: greedy improves the
preprocessed start by exactly one 2Q gate on 5/7 training widths
(5b 55->54, 6b 66->65, 7b 77->76, 8b 88->87, 9b 99->98) and on BOTH
held-outs (4b 44->43, 10b 110->109); 8b is notable as a width no search
phase ever cracked, so that gate is pure generalization. But the deep
chained targets are NOT reproduced: 2b/3b greedy flat, nothing reaches
20/31/53/63; only 7b chained-greedy recovers its search best (74).
Held-out 4b at 43 is one gate behind the earlier expert-iteration
policy's 42.

**Verdict.** The recipe's components split cleanly. Guided chained
search is genuinely powerful (new best on 3b; fills widths random can't
touch). But the entry-61 consolidation result does NOT scale from
per-circuit to shared multi-width training: with trajectories from many
widths in one dataset, label accuracy collapses (24/113) and greedy
replay of deep targets is lost, leaving a policy with broad one-gate
generalization instead of deep memorization. Consolidation and
generalization currently trade off. The v3.1 extension (250
episodes/round guided, 4 rounds) queued next tests whether more guided
data shifts this.

**Artifacts:** scripts/train_agent_a_v3.py,
results/checkpoints/agent_A_v3_128dedup.pt, scratchpad/agent_a_v3.log.

### 63. R/P v3 under the winning recipe: the structured-vs-random
dichotomy holds at 128 channels -- learning neither helps nor hurts on
random families (2026-08-30)

**What ran.** scripts/train_agents_rp_v3.py: the A-v3 recipe (search ->
dedup-by-construction labels -> 128-channel distillation, 150 epochs)
applied to the structured-random and pure-random families (150 training
circuits each), evaluated against the untrained-network control on 30
fresh circuits per family, best-of-15 per circuit.

**Results.** Structured-random: search improved 130/150 training
circuits (847 pairs); on fresh circuits distilled 10.3% vs untrained
9.8% mean 2Q reduction (head-to-head 8 wins / 10 losses / 12 ties);
basic_opt -1.0%. Pure-random: search improved 73/150 (505 pairs);
distilled 4.1% vs untrained 4.2% (6 / 6 / 18); basic_opt 0.5%.
Checkpoints agent_v3_128_{structured_random,pure_random}.pt.

**Verdict.** Statistical ties on both families. The entry-52 finding
softens but does not reverse: at 128 channels with clean labels the
distilled prior no longer LOSES to the untrained control (entry 52's
diversity-narrowing penalty is gone), but it still adds nothing --
confirming, under the project's best training recipe, that on families
without shared structure the value comes from generic search, not from
anything learned. This is the strongest version of the dichotomy yet:
same recipe, same budget, transfer on adders (entry 62) and a tie on
random families. Both agents still clearly beat basic_opt on these
families, as before, via best-of-N search.

**Artifacts:** scripts/train_agents_rp_v3.py,
results/checkpoints/agent_v3_128_structured_random.pt,
results/checkpoints/agent_v3_128_pure_random.pt, scratchpad/rp_v3.log.

### 64. Agent A v3.1 extension: more guided data closes the deployment
gap -- new all-time bests on three widths, best-ever held-out transfer
(2026-08-30)

**What ran.** scripts/train_agent_a_v3_ext.py: loaded the v3 checkpoint
as the search prior, ran a larger guided chained search (250
episodes/round, up to 4 rounds) on the seven training widths, distilled
a FRESH 128-channel policy on those 139 pairs only, and evaluated as
before. Checkpoint: results/checkpoints/agent_A_v3_1_128dedup.pt.

**Guided search.** Every width contributed for the first time,
including the previously impenetrable 8-bit: 2b 20, 3b 31, 5b 52 (one
below tier-1-max's format, deepest guided find yet), 6b 63, 7b 74,
8b 88->85 (first crack at any budget), 9b 96. Dataset 139 pairs.

**Evaluation -- the deployment gap closes.** Chained-greedy-20 now
matches or beats every previous all-time best on six of seven training
widths and sets three NEW all-time bests: 8b 83 (prev 84), 9b 94 (prev
96), 10b held-out 108 (prev 109); it reproduces 3b 31, 5b 52, 7b 73,
and 4b held-out 42. Sampling hit rates jumped (2b 8/20 at 20; 3b 8/20;
8b 7/20; 9b 9/20; held-out 4b 10/20 at 42 -- half of samples improve a
never-seen width, vs 7/40 for the entry-51 policy). Plain greedy
improves the start by one gate on 6/7 training widths and both
held-outs (2b flat).

**Verdict.** The v3 -> v3.1 comparison isolates dataset size as the
variable that closes the consolidation-generalization trade-off noted
in entry 62: same recipe, same architecture, 139 pairs covering all
widths instead of 113 skewed to two, and the deployed policy recovers
or exceeds its own search bests nearly everywhere while transferring
more strongly than any predecessor to held-out widths. v3.1 is the
strongest Agent A the project has produced on every deployment mode
measured. Caveats: single run, 20-sample evals, chained-greedy uses 20
rollouts/round; 7b's 73 and 4b's 42 tie rather than beat prior bests.

**Artifacts:** scripts/train_agent_a_v3_ext.py,
results/checkpoints/agent_A_v3_1_128dedup.pt, scratchpad/a_v3_ext.log.

### 65. Dissertation restructured around the final agents; Table 4.1/4.2
refreshed from a new same-protocol evaluation of the v3/v3.1 checkpoints
(2026-08-31)

**User decision.** The final agents (R v2 = agent_v3_128_structured_random,
P v2 = agent_v3_128_pure_random, A v2 = agent_A_v3_1_128dedup) are now
the main agents the dissertation is built on; the initial PPO agents
remain as the diagnosed failure that motivated them.

**New evaluation (scripts/evaluate_final_v2.py).** Same circuits and
best-of-10 protocol as evaluate_final.py; agents receive the
PREPROCESSED circuit (their documented pipeline; a first run feeding raw
circuits under-served them: A v2 4-bit gave 44 instead of 42), baselines
the raw one, reductions vs raw initial counts. Results
(evaluation_results_final_v2.csv): 2Q mean reduction, final agents
15.8/16.1/17.3% structured (basic_opt 6.8), 3.6-8.5% pure (basic_opt
0.0 this draw), 8.4-8.6% real-world arithmetic (basic_opt 6.7); adder
set: A v2 beats basic_opt on EVERY width (42/76/87/98 vs 44/77/88/99),
R/P v2 reproduce basic_opt exactly. Random-family draws are fresh
(pyzx generators unseeded), noted in the table caption; adder and
real-world sets are deterministic and comparable across runs.

**Chapter changes.** Ch4 restructured: trimmed "initial PPO agents"
section, phase-2 experiments moved up as "Building the final agents",
Tables 4.1/4.2 refreshed (final agents as main rows, initial agents as
one summary row, new baseline numbers from the same run), adder section
retabulated (A v2 column; honesty note that widths 7-9 are its training
widths, only 4-bit held out), new comparison_barplot_v2 figure, summary
rewritten. Ch1 objectives 2/3, contributions, ch3 phase-2 section
title, ch5 intro clarifier + section retitle, ch7 objectives 2/3, and
the ABSTRACT (which still overclaimed a cflow-zx reimplementation and
predated phase 2 entirely) all updated to the two-generation narrative.

**Artifacts:** scripts/evaluate_final_v2.py,
scripts/plot_comparison_barplot_v2.py,
results/logs/evaluation_results_final_v2.csv,
Writing/latex/figures/comparison_barplot_v2.png.

### 66. Adder generator carry-wiring bug found and fixed; Agent A v4
retrained on verified-correct Cuccaro adders (2026-08-31 -> 09-01)

**The bug.** Prompted by the user comparing our figures against Cuccaro
et al.'s paper, unitary simulation showed ripple_carry_adder was NOT a
correct adder: _maj/_uma had the CNOT controls and Toffoli target on the
carry wire instead of the a wire, so the ripple read its carry from a
wire it was never written to (12/16 correct basis states at n=2; carry
into bit 1 = a0 instead of MAJ(a0,b0)). Every previously trained agent
(PPO A, v3, v3.1) trained on this mis-wired family. Structurally the
families are twins: identical preprocessed counts (11n 2Q, 10n T) at
every width, so the optimisation landscape's headline statistics were
unaffected, and R/P (random families) plus all real-world evaluations
were never touched by the bug.

**The fix and its verification.** _maj/_uma rewritten to the paper's
exact wiring. Verified: exhaustive unitary simulation n=1..3 raw plus
n=4 THROUGH preprocessing (all basis states; A and ancilla restored;
ancilla=1 gives a+b+1 per the paper's Sec 4.2); functional agreement
with qiskit's CDKMRippleCarryAdder(kind='fixed') at n=2,3 (zero
mismatches); permanent regression test tests/test_adder_correctness.py.
Our construction = the paper's Section 4.1 modulo-2^n family (2n+1
qubits, no carry-out; sum onto register B).

**v4 retrain (scripts/train_agent_a_v4_correct.py).** The v3.1 recipe
end to end on the corrected family: chained random search (150/round,
<=3 rounds, widths {2,3,5,6,7,8,9}) gave only 18 pairs (widths 2,3,7);
128ch dedup distill; guided chained search (250/round, <=4 rounds)
filled EVERY width (31 pairs); fresh distill on guided pairs alone.
Checkpoint agent_A_v4_correct.pt.

**Results.** Deployment improves every width by exactly one 2Q gate:
train 21/32/54/65/76/87/98 (hits 6-14/20; chained-greedy reproduces
everywhere; plain greedy improves on 5b/7b/9b), held-out 4-bit 44->43
(greedy, 10/20) and 10-bit 110->109 (8/20). So the corrected-family
agent beats basic_opt on all nine widths including both held-outs.
Notably, NO deep multi-gate reductions exist on this family in any
phase: the correct adder sits at (apparently) one gate above its
reachable floor, whereas the mis-wired family's redundancy (its
back-to-back-cancelling top-bit structure plus broken carry) admitted
reductions to 83/94 etc. The old family's deep-target results were
partly mining the bug's redundancy -- an honest and important
correction for the write-up.

**Chapter alignment done alongside:** ch3 figures 3.3/3.4 regenerated
from the corrected generator (paper wire ordering; four-level figure
with Toffoli-decomposition mapping panel); Riu-style pipeline figure
3.5 (scripts/plot_pipeline_riu_style.py, unitary-verified example);
methodology phase-2 section corrected to the exact final recipe
(fresh-distill-on-guided-only, real budgets, R/P setup, initial-vs-
final width split noted); verification sentence added.

**Still to do:** re-evaluate the ch4 adder columns on corrected
circuits (baselines + final agents best-of-10), update tab
results-phase2 numbers to v4, and add the disclosure/limitations note.

### 67. Motif discovery, optimality certificates, and real-world win
decomposition (2026-09-02, deadline day)

**Motif.** Tracing final Agent A's winning episodes: on every adder
width (trained and held-out) the win is the same three-PIVG sequence,
gain landing only at step 3 -- one width-independent learned motif.

**Certificates (scripts/motif_experiments.py, E1).** Exhaustive
enumeration over all sequences of up to three pivot-gadget applications
certifies the agent's result OPTIMAL within the motif class: 2-bit 21
(56,715 sequences), 3-bit 32 (198,892). Enumerator initially invalid --
pyzx graph.copy() renumbers vertices, so parent-graph action tuples
mis-address copies; fixed by per-copy re-enumeration with index
correspondence, validated by reproducing the agent's recorded 2-bit
trajectory. 4-bit full + local 5/7/10 still running at write-up time.

**Real-world decomposition (E2, complete).** Agent A on the 11
arithmetic benchmarks (greedy + 10 samples): one genuine motif win
(mod_mult_55 48->47 via PIVG x2, greedy); two "wins" with EMPTY action
traces (QFTAdd8 184->183, adder_8 385->383) = the pipeline's
reset-time re-extraction round trip, not policy actions; remainder no
win this run (earlier family-table margins on other circuits came from
other agents or sampled luck). Honest attribution now written into ch5:
real-world margin = extraction round-trip + occasional motif wins +
sampled search.

**Chapters.** Ch5 final-agents section gains the motif/certificate/
decomposition paragraph; ch6 future work gains direction six (distil
the motif into an interpretable deterministic pivot-gadget peephole
pass). Dangling tab:results-phase2 ref (user removed the per-width
table as redundant with tab:adders-2q-all) repointed.

### 68. Motif localised: it acts once, on the top-bit wire (2026-09-01)

Prompted by the user's question (if the cell repeats, why not one gate
saved PER cell?): tracing the winning trajectories' target vertices
shows all three pivot-gadgets act on adjacent spiders of ONE wire at
every width -- qubit n, the top bit of register A, exactly the region
where the modulo-2^n construction omits the paper's carry-out and
leaves the top MAJ/UMA adjacent. Repetition of the bulk cell gives
recognisability (transfer to held-out widths); the site's uniqueness
gives exactly one gate per circuit, at every width, matching the
exhaustive certificates (2/3/4-bit full: 21/32/43 optimal in class;
5/7-bit local corroboration). Ch5 tracing paragraph updated from
"acts on the repeated cell" to the localised, observed mechanism.

## Entry 69 — Combined-agent experiments (overnight 2026-09-01/02, user: "can agent p, r and a be combined" / "do the training while I'm sleeping")

Two runs, both saved with checkpoints, logs, and eval CSVs.

**v1 (`train_agent_combined_v1.py`, pooling at the SEARCH stage):** the final
agents' own recipes pooled — A-v4 adder pipeline + R/P-v3 random collection,
one intermediate distill from the pooled blind pairs guiding the adder search.
Result: the pooled blind dataset was ~98% random-family pairs (adders
contributed ~18/1,159), the diluted intermediate policy found ZERO guided adder
wins (blind finds 3/7 widths at a smaller budget), so the final distill saw no
adder data. Eval: basic_opt floor on all 9 adder widths (motif gone);
random families track the untrained control; real-world mostly floor ties.
Checkpoint agent_combined_v1.pt; eval agent_combined_eval.csv.

**v2 (`train_agent_combined_v2_pooled_labels.py`, pooling at the DISTILLATION
stage):** adder trajectories collected with agent_A_v4_correct.pt guiding
chained search (found the one-below win on all 7 training widths; 29 clean
pairs), pooled with 1,140 random-family pairs, one fresh 150-epoch distill.
Result: motif still gone — basic_opt floor on all 9 adder widths. So label
dilution alone (29 vs 1,140 pairs, 2.5% share) destroys the motif even with
perfect adder labels; the failure is not only the broken search prior.
Checkpoint agent_combined_v2.pt; eval agent_combined_v2_eval.csv.

**Family means (best-of-10, % 2Q reduction):** v1: struct 8.0, pure −0.7,
adders 0.0/0.0, real 0.8; v2: struct 8.1, pure 2.9, adders 0.0/0.0, real 0.2.

**Incidental sampled-search bests (not learned behaviour):** adder_8 381 (v1)
and 382 (v2) vs the recorded 383; qcla_com_7 173 (v2) vs the 174 every prior
method tied at. Both confirm the ch5 framing: real-world margins are
search-dominated and protocol-bound.

**Conclusion:** the three final agents cannot be merged into one policy by
pooling training data under this recipe — structure-specific capability is
destroyed by dilution at either stage. Combining them as a deployment-time
portfolio (run all three, keep best) remains the only combination that works.
Exploratory only; NOT part of the dissertation's controlled ablation.
