"""Deterministic pyzx baselines (no training required).

Migrated from Section 10 of the dissertation notebook. Runs PyZX's own
simplification routines on the same evaluation circuits, for the same comparison
table structure as Table 2 in Riu et al. and Table 1 in AlphaTensor-Quantum.
"""

from typing import Optional

import pyzx as zx

from .causal_flow import n2q_via_causal_flow
from .zx_utils import (
    apply_local_complementation,
    apply_pivot,
    circuit_to_graphlike,
    extract_and_cleanup,
    find_feasible_lc,
    find_feasible_pivots,
    replace_in_place,
    two_qubit_gate_count,
)


def baseline_basic_optimization(circuit: "zx.Circuit") -> "zx.Circuit":
    """basic_optimization alone, no graph-based step at all -- the gate-local
    cleanup pass on its own, isolated as its own reported baseline rather than
    only ever appearing as half of a combined method."""
    return zx.basic_optimization(circuit.to_basic_gates())


def baseline_full_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    g = circuit.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract_circuit(g).to_basic_gates()


def baseline_basic_plus_full_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    c = zx.basic_optimization(circuit.to_basic_gates())
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    c2 = zx.extract_circuit(g).to_basic_gates()
    return zx.basic_optimization(c2)


def baseline_teleport_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    """T-count-only simplification (Duncan et al. 2020, Section 2.4.2): provably
    "does not change the original structure of the circuit, and in particular, the
    number or location of the two-qubit gates" -- so final_2q should equal
    initial_2q exactly on every circuit, by construction rather than as an
    empirical finding. Included as a baseline in its own right (not just as the
    preprocessing half of basic_opt+teleport_reduce below) precisely to make that
    guarantee visible in the results table, alongside full_reduce's much less
    predictable effect on the same metric (Section~\ref{sec:analysis-baselines}).
    """
    g = circuit.to_graph()
    zx.simplify.teleport_reduce(g)
    zx.simplify.to_graph_like(g)
    return zx.extract_circuit(g).to_basic_gates()


def baseline_basic_plus_teleport_reduce(circuit: "zx.Circuit") -> "zx.Circuit":
    """basic_optimization + teleport_reduce -- exactly the preprocessing pipeline
    (_light_preprocess in generators.py) every training circuit passes through
    before the RL agent ever sees it. Evaluating it here as a baseline in its own
    right answers a natural question the preprocessing discussion otherwise leaves
    implicit: how much of any agent's apparent improvement is actually just this
    fixed pipeline, versus something the trained policy itself contributes on top
    of it."""
    c = zx.basic_optimization(circuit.to_basic_gates())
    g = c.to_graph()
    zx.simplify.teleport_reduce(g)
    zx.simplify.to_graph_like(g)
    c2 = zx.extract_circuit(g).to_basic_gates()
    return zx.basic_optimization(c2)


def baseline_cflow_opt(circuit: "zx.Circuit") -> Optional["zx.Circuit"]:
    """Single-step causal-flow-preserving two-qubit-gate optimizer, after Holker
    (2024) "Causal flow preserving optimisation of quantum circuits in the
    zx-calculus" (arXiv:2312.02793) -- a deliberately scoped reimplementation of
    just its scoring idea (Lemma 17: N2Q = |E| - |V| + |I| for a graph-like
    diagram admitting causal flow, computable straight from the graph, no
    extraction needed) applied to a single best LC/pivot move, not the full
    greedy loop over many chained moves.

    Every feasible LC/pivot move is scored this way; the one that reduces N2Q the
    most is applied, once, if any does. Restricted to one step deliberately: an
    earlier version chained moves iteratively (closer to the paper's actual
    Algorithm 1) and was verified correct move-by-move in isolation, but
    something about the accumulated graph state across several chained moves
    caused pyzx's own circuit extraction to fail outright on the result -- not
    yet root-caused. A single verified move sidesteps that entirely: it's scored
    and applied to the same graph object (never reapplied elsewhere), so there's
    no opportunity for the divergence seen in the multi-step version to arise.
    See docs/project_log.md for the full investigation and why this exists at
    all: Riu et al. (2025) Section 3.3.2 compares against `cflow-zx` for their
    terminal reward bonus, which this project's environment previously had no
    equivalent of at all.

    Returns None if the starting circuit doesn't admit a causal flow (e.g. it has
    unresolved T gates), or if no single move improves on it -- callers should
    fall back to a different baseline in either case, as for this module's other
    baselines when they fail.
    """
    g = circuit_to_graphlike(circuit)
    current_n2q = n2q_via_causal_flow(g)
    if current_n2q is None:
        return None

    best_graph = None
    best_n2q = current_n2q

    for v in find_feasible_lc(g):
        g2 = g.copy()
        try:
            apply_local_complementation(g2, v)
            candidate_n2q = n2q_via_causal_flow(g2)
        except Exception:
            # Graph surgery can raise on a target whose neighbourhood shifted
            # since find_feasible_lc() enumerated it (same issue guarded against
            # in env.py's step()) -- treat as infeasible, not fatal.
            continue
        if candidate_n2q is not None and candidate_n2q < best_n2q:
            best_n2q = candidate_n2q
            best_graph = g2

    for v1, v2 in find_feasible_pivots(g):
        g2 = g.copy()
        try:
            apply_pivot(g2, v1, v2)
            candidate_n2q = n2q_via_causal_flow(g2)
        except Exception:
            continue
        if candidate_n2q is not None and candidate_n2q < best_n2q:
            best_n2q = candidate_n2q
            best_graph = g2

    if best_graph is None:
        return None

    try:
        circuit_opt = zx.extract_circuit(best_graph.copy())
    except Exception:
        return None
    final = zx.basic_optimization(circuit_opt.to_basic_gates())
    if two_qubit_gate_count(final) > current_n2q:
        # basic_optimization is expected to be 2q-neutral, but this is exactly
        # the kind of post-hoc surprise the multi-step version hit -- verify
        # rather than trust, and refuse to hand back a result worse than the
        # unmodified starting circuit.
        return None
    return final


def baseline_gflow_opt(circuit: "zx.Circuit") -> Optional["zx.Circuit"]:
    """Single-step gflow-preserving two-qubit-gate optimizer, mirroring
    baseline_cflow_opt's scope but for the more general gflow condition
    (Riu et al.'s `gflow-zx`, Section 2.4.3).

    Unlike causal flow, gflow has no closed-form two-qubit-gate-count formula:
    Riu et al. state directly that "it is unclear how to predict the resulting
    amount of two-qubit gates without performing circuit extraction" for a
    gflow-preserving diagram. Every feasible LC/pivot move that preserves
    gflow is therefore scored by actually extracting the resulting circuit and
    counting two-qubit gates, not via a cheap graph-only formula -- more
    expensive per candidate than baseline_cflow_opt, but the gflow-finding
    algorithm itself (Perdrix & Mhalla, via GF(2) linear algebra) did not need
    to be reimplemented: pyzx 0.10.5 already ships it as `pyzx.gflow.gflow`,
    unlike causal flow, for which no pyzx equivalent existed
    (see causal_flow.py).

    Returns None if the starting circuit doesn't admit a gflow, or if no
    single gflow-preserving move improves on it.
    """
    from pyzx.gflow import gflow as compute_gflow

    g = circuit_to_graphlike(circuit)
    if compute_gflow(g) is None:
        return None

    base_circuit = extract_and_cleanup(g)
    if base_circuit is None:
        return None
    current_n2q = two_qubit_gate_count(base_circuit)

    best_graph = None
    best_n2q = current_n2q

    for v in find_feasible_lc(g):
        g2 = g.copy()
        try:
            apply_local_complementation(g2, v)
        except Exception:
            continue
        if compute_gflow(g2) is None:
            continue
        c2 = extract_and_cleanup(g2)
        if c2 is None:
            continue
        candidate_n2q = two_qubit_gate_count(c2)
        if candidate_n2q < best_n2q:
            best_n2q = candidate_n2q
            best_graph = g2

    for v1, v2 in find_feasible_pivots(g):
        g2 = g.copy()
        try:
            apply_pivot(g2, v1, v2)
        except Exception:
            continue
        if compute_gflow(g2) is None:
            continue
        c2 = extract_and_cleanup(g2)
        if c2 is None:
            continue
        candidate_n2q = two_qubit_gate_count(c2)
        if candidate_n2q < best_n2q:
            best_n2q = candidate_n2q
            best_graph = g2

    if best_graph is None:
        return None

    try:
        circuit_opt = zx.extract_circuit(best_graph.copy())
    except Exception:
        return None
    final = zx.basic_optimization(circuit_opt.to_basic_gates())
    if two_qubit_gate_count(final) > current_n2q:
        return None
    return final


def baseline_cflow_multistep(
    circuit: "zx.Circuit", max_moves: Optional[int] = None, verbose: bool = False
) -> Optional["zx.Circuit"]:
    """Multi-step, chained causal-flow-preserving optimizer -- an attempt at
    the version paused in entry 26 and investigated further in entry 27 of
    docs/project_log.md, retried using the g.clone()+g.replace() commit
    pattern identified by reading Holker (2024)'s own reference
    implementation, instead of this project's earlier g.copy()+plain-reference
    pattern (`g = best_graph`) that preceded the unresolved extraction failure.

    `.replace()` doesn't exist in mainline pyzx (this project's dependency,
    unlike Holker's own fork), so `zx_utils.replace_in_place` reimplements its
    effect directly on the GraphS backend: `g`'s own internal state is
    overwritten to match a scored clone's, so `g`'s Python identity survives
    every committed move, and `.clone()` (unlike `.copy()`) preserves vertex
    IDs exactly rather than renumbering them -- both properties Holker's
    algorithm relies on that this project's earlier attempt did not
    reproduce.

    Does NOT touch baseline_cflow_opt (the verified single-step version
    env.py's terminal bonus actually uses) -- this is a separate, standalone
    function for investigating whether the multi-step version now works at
    all, not a replacement wired into training.

    Recomputes the full candidate set fresh every round rather than
    incrementally updating scores the way Holker's actual algorithm does --
    a simplicity-over-speed choice, consistent with this project's general
    preference (e.g. the PPO batching work) for a slower but easier-to-verify
    implementation over a faster one within the time available.

    Returns None if the starting circuit doesn't admit a causal flow, or if
    no move improves it at all (zero moves applied). Otherwise returns the
    final circuit after all committed moves, extraction, and
    basic_optimization cleanup.
    """
    g = circuit_to_graphlike(circuit)
    current_n2q = n2q_via_causal_flow(g)
    if current_n2q is None:
        return None

    moves_applied = 0
    while max_moves is None or moves_applied < max_moves:
        best_check = None
        best_n2q = current_n2q
        best_desc = None

        for v in find_feasible_lc(g):
            check_g = g.clone()
            try:
                apply_local_complementation(check_g, v)
                candidate_n2q = n2q_via_causal_flow(check_g)
            except Exception:
                continue
            if candidate_n2q is not None and candidate_n2q < best_n2q:
                best_n2q = candidate_n2q
                best_check = check_g
                best_desc = ("lc", v)

        for v1, v2 in find_feasible_pivots(g):
            check_g = g.clone()
            try:
                apply_pivot(check_g, v1, v2)
                candidate_n2q = n2q_via_causal_flow(check_g)
            except Exception:
                continue
            if candidate_n2q is not None and candidate_n2q < best_n2q:
                best_n2q = candidate_n2q
                best_check = check_g
                best_desc = ("pivot", v1, v2)

        if best_check is None:
            break

        replace_in_place(g, best_check)
        current_n2q = best_n2q
        moves_applied += 1
        if verbose:
            print(f"  move {moves_applied}: {best_desc} -> n2q={current_n2q}")

    if moves_applied == 0:
        return None

    try:
        circuit_opt = zx.extract_circuit(g.copy())
    except Exception as e:
        if verbose:
            print(f"  extraction FAILED after {moves_applied} moves: {e}")
        return None

    final = zx.basic_optimization(circuit_opt.to_basic_gates())
    if two_qubit_gate_count(final) > current_n2q:
        return None
    return final


def baseline_gflow_multistep(
    circuit: "zx.Circuit", max_moves: Optional[int] = None, verbose: bool = False
) -> Optional["zx.Circuit"]:
    """Multi-step, chained gflow-preserving optimizer -- the gflow counterpart
    of baseline_cflow_multistep, added specifically so a fair multi-step vs.
    multi-step comparison could be made against cflow-zx (Chapter 3's
    baseline-comparison figure originally compared this project's
    *single-move* cflow-zx/gflow-zx against the paper's iterated originals,
    which is why they tracked basic-opt so closely there).

    Scoring differs from baseline_cflow_multistep in the same way
    baseline_gflow_opt's does from baseline_cflow_opt's: no closed-form
    gate-count formula exists for gflow, so every candidate move at every
    round is scored by real extraction, not a graph-only formula -- markedly
    more expensive per round than the cflow version, since each round both
    re-checks gflow existence (a linear-algebra computation over GF(2)) and
    extracts a full circuit for every feasible move.

    Same commit pattern as baseline_cflow_multistep (g.clone() + score +
    replace_in_place on success), same fresh-full-recompute-per-round
    simplicity trade-off, same non-interference with any function used live
    in env.py.
    """
    from pyzx.gflow import gflow as compute_gflow

    g = circuit_to_graphlike(circuit)
    if compute_gflow(g) is None:
        return None
    base_circuit = extract_and_cleanup(g)
    if base_circuit is None:
        return None
    current_n2q = two_qubit_gate_count(base_circuit)

    moves_applied = 0
    while max_moves is None or moves_applied < max_moves:
        best_check = None
        best_n2q = current_n2q
        best_desc = None

        for v in find_feasible_lc(g):
            check_g = g.clone()
            try:
                apply_local_complementation(check_g, v)
            except Exception:
                continue
            if compute_gflow(check_g) is None:
                continue
            c2 = extract_and_cleanup(check_g)
            if c2 is None:
                continue
            candidate_n2q = two_qubit_gate_count(c2)
            if candidate_n2q < best_n2q:
                best_n2q = candidate_n2q
                best_check = check_g
                best_desc = ("lc", v)

        for v1, v2 in find_feasible_pivots(g):
            check_g = g.clone()
            try:
                apply_pivot(check_g, v1, v2)
            except Exception:
                continue
            if compute_gflow(check_g) is None:
                continue
            c2 = extract_and_cleanup(check_g)
            if c2 is None:
                continue
            candidate_n2q = two_qubit_gate_count(c2)
            if candidate_n2q < best_n2q:
                best_n2q = candidate_n2q
                best_check = check_g
                best_desc = ("pivot", v1, v2)

        if best_check is None:
            break

        replace_in_place(g, best_check)
        current_n2q = best_n2q
        moves_applied += 1
        if verbose:
            print(f"  move {moves_applied}: {best_desc} -> n2q={current_n2q}")

    if moves_applied == 0:
        return None

    try:
        circuit_opt = zx.extract_circuit(g.copy())
    except Exception as e:
        if verbose:
            print(f"  extraction FAILED after {moves_applied} moves: {e}")
        return None

    final = zx.basic_optimization(circuit_opt.to_basic_gates())
    if two_qubit_gate_count(final) > current_n2q:
        return None
    return final
