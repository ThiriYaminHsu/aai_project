"""Circuit generators injected into the Gym environment at construction time.

Migrated from Section 4 of the dissertation notebook, plus one addition (see below).
Interchangeable generator functions, all with signature `() -> zx.Circuit`. This is
the *only* thing that differs between Agent R, Agent A, and Agent P's training runs.
"""

import random
from fractions import Fraction
from typing import Callable, Optional, Tuple

import pyzx as zx

# ---------------------------------------------------------------------------
# Random Clifford+T generator (replicates Riu et al. Sec. 4 training regime)
# ---------------------------------------------------------------------------


def _light_preprocess(g: "zx.graph.base.BaseGraph") -> "zx.Circuit":
    """Pre-processing applied to a freshly-generated random circuit before the RL
    agent ever sees it. Verified against Riu et al.'s actual released reset()
    (rl-zx/gym-zx/gym_zx/envs/zx_env.py, qilimanjaro-tech/Circopt-RL-ZXCalc): light
    gate-level cleanup (basic_optimization) plus T-count-only reduction
    (teleport_reduce). teleport_reduce is explicitly chosen there (and here)
    because Duncan et al. Section 2.4.2 -- via Riu et al.'s citation of it --
    guarantees it "does not change the original structure of the circuit, and in
    particular, the number or location of the two qubit gates". This deliberately
    does NOT call full_reduce, which would apply the same local-complementation and
    pivoting rules the agent is being trained to discover, consuming the very
    structure the episode is supposed to give it to act on."""
    c = zx.Circuit.from_graph(g)
    c = zx.basic_optimization(c.to_basic_gates())
    g2 = c.to_graph()
    zx.simplify.teleport_reduce(g2)
    # extract_circuit requires graph-like form; to_graph_like is a structural
    # normalisation (spider fusion + Hadamard rules only, matching the paper's own
    # separate to_graph_like() step) -- not a further graph-theoretic reduction, so
    # this doesn't reintroduce the full_reduce problem this function exists to avoid.
    zx.simplify.to_graph_like(g2)
    return zx.extract_circuit(g2)


def make_random_circuit_generator(n_qubits: int, n_gates: int, seed: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    # `seed` is accepted for interface parity with make_adder_generator, but
    # zx.generate.cliffordT has no seed hook of its own -- it draws from
    # pyzx's global RNG state, so this generator is not independently
    # reproducible across instances the way the adder generator is.
    rng = random.Random(seed)

    def _generate() -> "zx.Circuit":
        rng.randint(0, 2**31 - 1)
        # zx.generate.cliffordT samples CNOT/CZ, Hadamard, S, and T gates
        # with these exact probabilities, matching Riu et al.'s released
        # code (gym-zx/envs/zx_env.py reset()), not the paper's prose summary
        g = zx.generate.cliffordT(n_qubits, n_gates, p_t=0.17, p_s=0.24, p_hsh=0.25)
        return _light_preprocess(g)

    return _generate


# ---------------------------------------------------------------------------
# Pure-random Clifford+T generator (Agent P's training regime)
# ---------------------------------------------------------------------------
# Not in Riu et al. -- added for this dissertation's random-vs-structured ablation.
# Unlike make_random_circuit_generator above, nothing is fixed: circuit size
# (n_qubits, n_gates) is redrawn every episode from the given ranges, and no
# gate-type probabilities are passed to pyzx, so p_t/p_s/p_hsh/p_cnot all default to
# None, which zx.generate.cliffordT turns into a uniform 25/25/25/25 split rather than
# Riu et al.'s tuned 17/24/25/34. See docs/random_vs_structured_composition_analysis.md
# for the composition analysis that motivated this as a third training regime.


def make_pure_random_circuit_generator(min_qubits: int, max_qubits: int, min_gates: int, max_gates: int,
                                        seed: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    rng = random.Random(seed)

    def _generate() -> "zx.Circuit":
        n_qubits = rng.randint(min_qubits, max_qubits)
        n_gates = rng.randint(min_gates, max_gates)
        g = zx.generate.cliffordT(n_qubits, n_gates)
        return _light_preprocess(g)

    return _generate


# ---------------------------------------------------------------------------
# Ripple-carry adder generator (Cuccaro et al. 2004 construction)
# ---------------------------------------------------------------------------
# Built directly with pyzx's Circuit/gate API rather than an external QASM
# file, so bit-width is a free parameter. This follows the standard
# Cuccaro ripple-carry adder wiring: n+1 qubits for the two n-bit operands
# (one register overwritten in place) plus 1 ancilla/carry qubit, using
# MAJ (Toffoli+CNOT) and UMA gate blocks.
#
# NOTE: Toffoli gates are compiled here via the explicit 7-T-gate Toffoli
# decomposition below -- this is deliberate: it matches the "naive
# compilation, no measurement-based uncomputation" baseline used throughout
# Riu et al. and AlphaTensor-Quantum, so any T-count reduction the agent
# finds is attributable to the ZX rewriting itself, not to a smarter Toffoli
# gadget chosen upstream.


def _tdg(c: "zx.Circuit", target: int) -> None:
    """T-dagger: pyzx has no 'T*' shortcut key, so build it directly as a
    ZPhase gate with phase -pi/4."""
    c.add_gate("ZPhase", target, phase=Fraction(-1, 4))


def _add_toffoli(c: "zx.Circuit", ctrl1: int, ctrl2: int, target: int) -> None:
    """Standard 7-T-gate Toffoli decomposition (Nielsen & Chuang Fig 4.9)."""
    c.add_gate("H", target)
    c.add_gate("CNOT", ctrl2, target)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, target)
    c.add_gate("T", target)
    c.add_gate("CNOT", ctrl2, target)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, target)
    c.add_gate("T", ctrl2)
    c.add_gate("T", target)
    c.add_gate("H", target)
    c.add_gate("CNOT", ctrl1, ctrl2)
    c.add_gate("T", ctrl1)
    _tdg(c, target)
    c.add_gate("CNOT", ctrl1, ctrl2)


def _maj(c: "zx.Circuit", a: int, b: int, cq: int) -> None:
    # Cuccaro et al. Figure 1, exact wiring: both CNOTs controlled by the
    # a wire, Toffoli(carry, b -> a) writes the next carry ONTO the a
    # wire, which is where the next bit's MAJ reads it (their Figure 3).
    # An earlier version of this function had the cq/a roles swapped,
    # which broke carry propagation (caught 2026-08-31 by unitary
    # simulation against the paper; see tests/test_adder_correctness.py).
    c.add_gate("CNOT", a, cq)
    c.add_gate("CNOT", a, b)
    _add_toffoli(c, cq, b, a)


def _uma(c: "zx.Circuit", a: int, b: int, cq: int) -> None:
    # Cuccaro et al. Figure 2a (2-CNOT UMA), exact inverse-then-add wiring.
    _add_toffoli(c, cq, b, a)
    c.add_gate("CNOT", a, cq)
    c.add_gate("CNOT", cq, b)


def ripple_carry_adder(n_bits: int) -> "zx.Circuit":
    """n-bit Cuccaro ripple-carry adder (addition modulo 2^n: the paper's
    Section 4.1 family; no carry-out qubit). Qubit layout:
    q[0]         = carry ancilla, starts and ends |0>
    q[1 .. n]    = register A (a_0 .. a_{n-1}), restored at the end
    q[n+1 .. 2n] = register B (b_0 .. b_{n-1}), overwritten with the sum
    Total qubits = 2n + 1. Verified by unitary simulation to compute
    B <- A + B (mod 2^n) with A and the ancilla restored."""
    n_qubits = 2 * n_bits + 1
    c = zx.Circuit(n_qubits)

    carry = 0
    a = list(range(1, n_bits + 1))
    b = list(range(n_bits + 1, 2 * n_bits + 1))

    for i in range(n_bits):
        cq = carry if i == 0 else a[i - 1]
        _maj(c, a[i], b[i], cq)
    for i in reversed(range(n_bits)):
        cq = carry if i == 0 else a[i - 1]
        _uma(c, a[i], b[i], cq)

    return c


def make_adder_generator(min_bits: int, max_bits: int, exclude_bits: Tuple[int, ...] = (),
                          seed: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    """Returns a generator sampling a random bit-width in [min_bits, max_bits]
    (excluding `exclude_bits`, e.g. a held-out interpolation width) on every
    call, and building a fresh ripple-carry adder at that width.

    Now runs the adder through _light_preprocess, same as Agent R/P, rather
    than using it "as-is": on raw ripple-carry adders, the environment's own
    deterministic baselines (clifford_simp, teleport_reduce) never beat the
    starting metric, which made STOP-immediately a free, always-optimal
    action and collapsed training to reward=0 with near-zero entropy after
    ~150 updates (confirmed reproducible with a fixed seed). Preprocessing
    doesn't remove the underlying "no single move improves on the light-
    preprocessed baseline" property either (also confirmed directly) -- but
    it does give the agent the same starting conditions and action-space
    richness as R/P, which is where R's actual gains came from a multi-step
    sequence, not any single greedy move. Whether that carries over to
    adders is an open empirical question this change is meant to test, not
    something either static check above settles on its own. See
    docs/project_log.md."""
    rng = random.Random(seed)
    valid_widths = [n for n in range(min_bits, max_bits + 1) if n not in exclude_bits]
    assert valid_widths, "No valid bit-widths left after exclusion."

    def _generate() -> "zx.Circuit":
        n_bits = rng.choice(valid_widths)
        return _light_preprocess(ripple_carry_adder(n_bits).to_graph())

    return _generate


def make_fixed_adder_generator(n_bits: int) -> Callable[[], "zx.Circuit"]:
    """Deterministic single-bit-width generator, used for evaluation."""
    def _generate() -> "zx.Circuit":
        return ripple_carry_adder(n_bits)
    return _generate


# ---------------------------------------------------------------------------
# Paper-benchmark generator -- EXPERIMENTAL, not part of the R/P/A design
# ---------------------------------------------------------------------------
# Every other agent in this dissertation is deliberately never trained on
# Riu et al.'s 33-circuit benchmark suite, so it remains a genuine held-out
# validation set (Section~\ref{sec:methodology-eval}). This generator exists
# only for a one-off exploratory test of what happens if an agent *is*
# trained directly on that fixed, finite set -- sampled with replacement each
# episode, so training sees heavy repetition rather than fresh circuits the
# way R/P/A do. Any checkpoint trained with this generator cannot be
# meaningfully evaluated against the paper benchmark suite afterwards, since
# it would no longer be held-out data for that checkpoint specifically.

def make_paper_benchmark_generator(seed: Optional[int] = None,
                                    max_gates: Optional[int] = None) -> Callable[[], "zx.Circuit"]:
    """Samples one of Riu et al.'s 33 benchmark circuits uniformly at random
    (with replacement) each episode, decomposed to Clifford+T basis and
    light-preprocessed exactly as R/P/A's own training circuits are.

    `max_gates` (native, pre-decomposition gate count) optionally excludes the
    largest circuits in the set -- sizes here range up to 1562 gates (e.g.
    QFT32_before), far beyond anything R/P/A ever train on (capped at 200),
    and this project's per-step feasible-action enumeration was never
    exercised at that scale: a single episode on the largest circuits was
    found to make rollout collection impractically slow (docs/project_log.md).
    """
    from benchmarks.fetch_paper_circuits import load_all

    circuits = list(load_all().values())
    if max_gates is not None:
        circuits = [c for c in circuits if len(c.gates) <= max_gates]
    rng = random.Random(seed)

    def _generate() -> "zx.Circuit":
        c = rng.choice(circuits)
        return _light_preprocess(c.to_basic_gates().to_graph())

    return _generate
