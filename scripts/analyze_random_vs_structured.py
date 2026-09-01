"""Compare gate composition of random Clifford+T circuits against structured circuits.

Answers a simple question: how similar (in gate-type makeup) are the random circuits
used in Agent R's and Agent P's training regimes to the structured circuits used
elsewhere in this project (Agent A's adders) and in Riu et al.'s benchmark suite (held
out as validation only, never trained on -- see docs/random_vs_structured_composition_analysis.md)?

Four circuit families, 100 circuits each -- except riu_benchmarks, which is a fixed real-world
benchmark suite of 33 circuits (there's no "more" to generate; padding it to 100 by resampling
duplicates was considered and rejected as manufacturing fake data points):

  structured_random -- 100 circuits from make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random),
                     i.e. Agent R's actual training regime: fixed size (5 qubits, ~70 gates, matching
                     Riu et al.'s twoqubits-objective circuit generation, Table 1) and a fixed, tuned
                     gate-type recipe (T=17%, S=24%, HSH=25%, CNOT=34%) matching Riu et al.
                     Named "structured" because that recipe is itself a fixed structure being imposed
                     on the sampler, as opposed to pure_random below.
  pure_random     -- 100 circuits from make_pure_random_circuit_generator, i.e. Agent P's training
                     regime: no fixed size (n_qubits, n_gates both drawn per-circuit from a wide
                     range) and no fixed gate-type recipe (left as None, so pyzx.generate.cliffordT
                     falls back to a uniform 25/25/25/25 split over T/S/HSH/CNOT). Answers "is
                     structured_random's narrow cloud an artifact of Riu et al.'s specific sampling
                     recipe, or would any random Clifford+T circuit land there?"
  cuccaro_adder   -- 100 ripple-carry adders (Cuccaro et al. 2004), bit-width redrawn uniformly at
                     random from [2, 9] per circuit via generators.make_adder_generator -- mirrors
                     Agent A's training regime the same way structured_random mirrors Agent R's.
  riu_benchmarks  -- Riu, Nogue, Vilaplana, Garcia-Saez & Estarellas (2025)'s full benchmark
                     suite (benchmarks/fetch_paper_circuits.py): Toffoli cascades, QFTs,
                     multipliers, mod-arithmetic circuits, etc. -- n=33, see above. Validation-only:
                     none of Agent R/P/A train on this data.

Per circuit we count T gates, native Toffoli/CCZ gates, 2-qubit gates (CNOT/CZ/SWAP/CRZ),
and "other Clifford" (H, S, X, ...), using the same counting helpers evaluate.py relies on
(circopt_adder.zx_utils.t_count / two_qubit_gate_count) so the numbers here are consistent
with the rest of the project's reported metrics.

Usage:
    python scripts/analyze_random_vs_structured.py
"""

import certifi
import ssl
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # to import the sibling benchmarks/ package

# benchmarks/fetch_paper_circuits.py downloads over HTTPS; this machine's default
# Python install has no local CA bundle wired up, so point urllib at certifi's.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from benchmarks.fetch_paper_circuits import load_all as load_paper_benchmark_circuits
from circopt_adder.config import SEED, Config
from circopt_adder.generators import (
    make_adder_generator,
    make_pure_random_circuit_generator,
    make_random_circuit_generator,
)
from circopt_adder.zx_utils import t_count, two_qubit_gate_count

N_STRUCTURED_RANDOM = 100
N_PURE_RANDOM = 100
N_ADDER = 100
ADDER_MIN_BITS, ADDER_MAX_BITS = 2, 9  # spans the project's train + interpolation + extrapolation ranges

# dataviz reference palette, categorical slots 1-4 (blue/orange/aqua/yellow) -- this
# order is only validated for *adjacent* CVD separation (bar/box comparisons), not
# all-pairs, so the scatter plot below facets by family instead of overlaying all 4.
FAMILY_COLOR = {"structured_random": "#2a78d6", "pure_random": "#eda100",
                 "cuccaro_adder": "#eb6834", "riu_benchmarks": "#1baf7a"}
FAMILY_ORDER = ["structured_random", "pure_random", "cuccaro_adder", "riu_benchmarks"]
# display labels spell out each family's fixed recipe (or lack of one) directly on the
# axis, kept short enough to fit one column width without colliding with its neighbours;
# the full recipe (with n counts) is spelled out in the module docstring above.
FAMILY_LABEL = {
    "structured_random": "structured_random\n(T17/S24/H25)",
    "pure_random": "pure_random\n(uniform 25%)",
    "cuccaro_adder": "cuccaro_adder\n(n_bits~U[2,9])",
    "riu_benchmarks": "riu_benchmarks\n(fixed, n=33)",
}
OUT_DIR = REPO_ROOT / "results" / "figures"
LOG_DIR = REPO_ROOT / "results" / "logs"


def composition(circuit: "zx.Circuit", name: str, family: str) -> dict:
    gates = circuit.gates
    total = len(gates)
    t = t_count(circuit)
    toffoli = sum(1 for g in gates if g.name in ("Tof", "CCZ"))
    twoq = two_qubit_gate_count(circuit)  # CNOT/CZ/CX/SWAP/CRZ -- excludes Tof/CCZ, see gate attr check
    other_clifford = total - t - toffoli - twoq
    name_counts = Counter(g.name for g in gates)

    return {
        "circuit": name,
        "family": family,
        "n_qubits": circuit.qubits,
        "gate_count": total,
        "t_count": t,
        "toffoli_count": toffoli,
        "twoq_count": twoq,
        "other_clifford_count": other_clifford,
        "clifford_count": total - t - toffoli,
        "t_frac": t / total if total else 0.0,
        "toffoli_frac": toffoli / total if total else 0.0,
        "twoq_frac": twoq / total if total else 0.0,
        "other_clifford_frac": other_clifford / total if total else 0.0,
        "h_count": name_counts.get("HAD", 0) + name_counts.get("H", 0),
        "s_count": name_counts.get("S", 0),
    }


def build_dataset(cfg: Config) -> "tuple[pd.DataFrame, dict]":
    """Returns (metrics dataframe, {circuit_name: zx.Circuit}). The circuits dict is
    kept separate from the dataframe (rather than as a column) so the dataframe stays
    directly CSV-serializable; visualize_circuit_structures.py uses the circuits dict
    to draw actual circuit diagrams for the same population analyzed here."""
    rows = []
    circuits = {}

    structured_random_gen = make_random_circuit_generator(cfg.n_qubits, cfg.n_gates_random, seed=SEED)
    for i in range(N_STRUCTURED_RANDOM):
        c = structured_random_gen()
        name = f"structured_random_{i}"
        rows.append(composition(c, name, "structured_random"))
        circuits[name] = c

    pure_random_gen = make_pure_random_circuit_generator(
        cfg.pure_random_min_qubits, cfg.pure_random_max_qubits,
        cfg.pure_random_min_gates, cfg.pure_random_max_gates, seed=SEED,
    )
    for i in range(N_PURE_RANDOM):
        c = pure_random_gen()
        name = f"pure_random_{i}"
        rows.append(composition(c, name, "pure_random"))
        circuits[name] = c

    adder_gen = make_adder_generator(ADDER_MIN_BITS, ADDER_MAX_BITS, seed=SEED)
    for i in range(N_ADDER):
        c = adder_gen()
        name = f"cuccaro_adder_{i}_{c.qubits}q"
        rows.append(composition(c, name, "cuccaro_adder"))
        circuits[name] = c

    paper_circuits = load_paper_benchmark_circuits()
    print(f"Loaded {len(paper_circuits)} paper benchmark circuits")
    for name, c in paper_circuits.items():
        rows.append(composition(c, name, "riu_benchmarks"))
        circuits[name] = c

    return pd.DataFrame(rows), circuits


def plot_mean_composition(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar: mean gate-type makeup per family, as a fraction of gate count."""
    frac_cols = ["t_frac", "toffoli_frac", "twoq_frac", "other_clifford_frac"]
    labels = {"t_frac": "T", "toffoli_frac": "Toffoli/CCZ", "twoq_frac": "2-qubit (CNOT/CZ/SWAP)",
              "other_clifford_frac": "Other Clifford (H, S, X)"}
    means = df.groupby("family")[frac_cols].mean().reindex(FAMILY_ORDER)

    seg_colors = ["#eb6834", "#4a3aa7", "#2a78d6", "#c3c2b7"]  # T=orange, Toffoli=violet, 2q=blue, other=neutral
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = pd.Series(0.0, index=means.index)
    for col, color in zip(frac_cols, seg_colors):
        ax.bar(means.index, means[col], bottom=bottom, label=labels[col], color=color,
               edgecolor="#fcfcfb", linewidth=2)
        bottom += means[col]

    ax.set_xticks(range(len(FAMILY_ORDER)))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], fontsize=9.5)
    ax.set_ylabel("mean share of gate count")
    ax.set_ylim(0, 1.0)
    ax.set_title("Mean gate composition by circuit family")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "composition_stacked_mean.png", dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Two small multiples: spread of T-fraction and 2-qubit-fraction per family."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, col, title in zip(axes, ["t_frac", "twoq_frac"],
                               ["T-gate share of circuit", "2-qubit-gate share of circuit"]):
        sns.boxplot(data=df, x="family", y=col, order=FAMILY_ORDER, hue="family",
                    palette=FAMILY_COLOR, legend=False, ax=ax, width=0.5, fliersize=0)
        sns.stripplot(data=df, x="family", y=col, order=FAMILY_ORDER, color="#0b0b0b",
                      alpha=0.35, size=3, jitter=0.15, ax=ax)
        ax.set_xticks(range(len(FAMILY_ORDER)))
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("share of gate count")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "composition_distributions.png", dpi=150)
    plt.close(fig)


def plot_similarity_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """T-fraction vs 2-qubit-fraction per circuit, one small-multiple panel per family
    on shared axes -- the direct 'how similar are these populations' view: clouds that
    land in the same region across panels are similar, clouds in disjoint regions are not.
    Faceted (rather than one overlaid scatter) because 4 families of overlapping color
    would exceed the palette's all-pairs-safe slot count (3) at this point count."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = df[df["family"] == family]
        ax.scatter(sub["twoq_frac"], sub["t_frac"], s=28, alpha=0.6,
                   color=FAMILY_COLOR[family], edgecolor="#fcfcfb", linewidth=0.4)
        ax.set_title(FAMILY_LABEL[family], fontsize=9)
        ax.set_xlabel("2-qubit-gate share")
        sns.despine(ax=ax)
    axes[0].set_ylabel("T-gate share")
    fig.suptitle("Random vs structured circuits: gate-composition footprint")
    plt.tight_layout()
    plt.savefig(out_dir / "composition_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df, _circuits = build_dataset(cfg)
    df.to_csv(LOG_DIR / "circuit_composition.csv", index=False)

    summary = df.groupby("family")[["n_qubits", "gate_count", "t_frac", "toffoli_frac",
                                     "twoq_frac", "other_clifford_frac"]].agg(["mean", "std"])
    counts = ", ".join(f"{(df.family == fam).sum()} {fam}" for fam in FAMILY_ORDER)
    print(f"\n{len(df)} circuits analyzed ({counts})")
    print(summary.round(3).to_string())

    plot_mean_composition(df, OUT_DIR)
    plot_distributions(df, OUT_DIR)
    plot_similarity_scatter(df, OUT_DIR)
    print(f"\nSaved circuit_composition.csv to {LOG_DIR} and 3 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
