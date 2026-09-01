"""ZX-diagram utilities: graph-like conversion, feasible-action enumeration, and rule application.

Migrated from Section 3 of the dissertation notebook. Direct graph-surgery
implementations of the two Clifford simplification rules used as the RL action
space: local complementation (interior spiders with phase +-pi/2) and pivoting
(interior spider pairs with phase 0 or pi, connected by a Hadamard edge), following
Duncan, Kissinger, Perdrix & van de Wetering (2020) and Eq. 2-3 of Riu et al. These
operate directly on a pyzx.Graph via its low-level edge/phase API rather than
pyzx.simplify's batch routines, because the RL setting needs to apply and score
*one* action at a time.

This module implements all six of Riu et al.'s action types: LC, plain pivot,
pivot-boundary, pivot-gadget (the two non-Clifford pivot variants, p2/p3),
gadget fusion, and identity removal. The non-Clifford pivot variants reuse
pyzx's own verified `check_pivot_boundary`/`unsafe_pivot_boundary` and
`check_pivot_gadget`/`unsafe_pivot_gadget` (pyzx.rewrite_rules.pivot_rule),
which handle a non-Clifford phase by "gadgetizing" it -- moving the phase onto
a fresh leaf spider hanging off an ancilla, so the actual pivot step still only
ever acts on a Pauli/Pauli pair -- rather than requiring a new XZ/YZ-plane
spider representation as the paper's own prose might suggest in isolation;
pyzx's existing phase-gadget encoding already covers that case.
"""

from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Tuple

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType, EdgeType

# ---------------------------------------------------------------------------
# Circuit <-> graph-like ZX-diagram conversion
# ---------------------------------------------------------------------------


def circuit_to_graphlike(circuit: "zx.Circuit") -> BaseGraph:
    """Convert a pyzx Circuit into its graph-like ZX-diagram form."""
    g = circuit.to_graph()
    zx.simplify.to_graph_like(g)
    return g


def gate_count(circuit: "zx.Circuit") -> int:
    return len(circuit.gates)


def t_count(circuit: "zx.Circuit") -> int:
    """Counts T-phase gates by phase value, not gate name, since circuits
    that have passed through zx.extract_circuit() get their non-Clifford
    phases synthesized as generic ZPhase gates rather than named T/T*."""
    count = 0
    for gate in circuit.gates:
        if gate.name in ("T", "T*"):
            count += 1
        elif gate.name == "ZPhase" and hasattr(gate, "phase"):
            phase_mod = gate.phase % 2          # normalize to [0, 2)
            if phase_mod in (Fraction(1, 4), Fraction(7, 4)):
                count += 1
    return count


def two_qubit_gate_count(circuit: "zx.Circuit") -> int:
    """Counts gates acting on two qubits (CNOT, CZ, SWAP, etc.), matching the two-qubit gate metric reported in Fig. 8 of Riu et al."""
    return sum(1 for gate in circuit.gates if hasattr(gate, "control"))


def extract_and_cleanup(g: BaseGraph) -> Optional["zx.Circuit"]:
    """Extract a circuit from a graph-like diagram and run cheap gate-level
    cleanup (mirrors the `basic_optimization` post-processing step used
    before reward computation in Riu et al., Sec 3.3.2). Returns None if
    extraction fails (can happen mid-episode on malformed graphs)."""
    try:
        g_copy = g.copy()
        c = zx.extract_circuit(g_copy)
        c = zx.basic_optimization(c.to_basic_gates())
        return c
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feasible-action enumeration
# ---------------------------------------------------------------------------
# Phase convention: pyzx stores phases as Fraction multiples of pi, i.e.
# g.phase(v) == Fraction(1, 2) means phase = pi/2.

PHASE_PI_2 = Fraction(1, 2)
PHASE_3PI_2 = Fraction(3, 2)
PHASE_0 = Fraction(0, 1)
PHASE_PI = Fraction(1, 1)


def _is_interior(g: BaseGraph, v: int) -> bool:
    """A spider is interior if none of its neighbours are boundary
    (input/output) vertices, matching the definition in Section 3.1.4 of
    Nogue's thesis."""
    return all(g.type(n) != VertexType.BOUNDARY for n in g.neighbors(v))


def _all_hadamard_edges(g: BaseGraph, v: int) -> bool:
    return all(g.edge_type(g.edge(v, n)) == EdgeType.HADAMARD for n in g.neighbors(v))


def find_feasible_lc(g: BaseGraph) -> List[int]:
    """Interior Z-spiders with phase +-pi/2, connected to all neighbours
    via Hadamard edges only -- eligible for local complementation."""
    candidates = []
    for v in g.vertices():
        if g.type(v) != VertexType.Z:
            continue
        if g.phase(v) not in (PHASE_PI_2, PHASE_3PI_2):
            continue
        if not _is_interior(g, v):
            continue
        if not _all_hadamard_edges(g, v):
            continue
        candidates.append(v)
    return candidates


def find_feasible_pivots(g: BaseGraph) -> List[Tuple[int, int]]:
    """Interior Z-spider pairs with phase in {0, pi}, connected to each
    other by a Hadamard edge -- eligible for pivoting."""
    candidates = []
    for v1, v2 in g.edges():
        if g.edge_type(g.edge(v1, v2)) != EdgeType.HADAMARD:
            continue
        if g.type(v1) != VertexType.Z or g.type(v2) != VertexType.Z:
            continue
        if g.phase(v1) not in (PHASE_0, PHASE_PI):
            continue
        if g.phase(v2) not in (PHASE_0, PHASE_PI):
            continue
        if not (_is_interior(g, v1) and _is_interior(g, v2)):
            continue
        candidates.append((v1, v2))
    return candidates


def _gadget_axel_and_targets(g: BaseGraph, v: int) -> Optional[Tuple[int, FrozenSet[int]]]:
    """If `v` is a phase-gadget leaf (a degree-1 Z-spider attached via a
    single Hadamard edge to a phaseless, non-boundary "axel" spider that in
    turn connects to the rest of the diagram), return (axel, target_set).
    Otherwise None. This is the standard ZX encoding of a parity-phase gate
    e^{i*pi*phase*Z...Z} (Duncan, Kissinger, Perdrix & van de Wetering 2020)."""
    if g.type(v) != VertexType.Z:
        return None
    neighbours = list(g.neighbors(v))
    if len(neighbours) != 1:
        return None
    axel = neighbours[0]
    if g.edge_type(g.edge(v, axel)) != EdgeType.HADAMARD:
        return None
    if g.type(axel) != VertexType.Z or g.phase(axel) != PHASE_0:
        return None
    targets = frozenset(g.neighbors(axel)) - {v}
    if not targets:
        return None
    return axel, targets


def find_feasible_gadget_fusions(g: BaseGraph) -> List[Tuple[int, int, int, int]]:
    """Riu et al. Eq. 8 (gf): phase gadgets acting on the same target set
    combine additively, e^{i*a*P} . e^{i*b*P} = e^{i*(a+b)*P} for the same
    Pauli string P -- exactly the redundancy a Toffoli decomposition creates
    (e.g. a T and T* gadget on the same qubits cancel to identity). Returns
    (axel1, leaf1, axel2, leaf2) tuples: fusing leaf2's phase into leaf1 and
    removing axel2/leaf2 is the one-action-at-a-time version of this rule."""
    by_target: Dict[FrozenSet[int], List[Tuple[int, int]]] = {}
    for v in g.vertices():
        match = _gadget_axel_and_targets(g, v)
        if match is None:
            continue
        axel, targets = match
        by_target.setdefault(targets, []).append((axel, v))

    candidates: List[Tuple[int, int, int, int]] = []
    for pairs in by_target.values():
        if len(pairs) < 2:
            continue
        axel1, leaf1 = pairs[0]
        for axel2, leaf2 in pairs[1:]:
            candidates.append((axel1, leaf1, axel2, leaf2))
    return candidates


def find_feasible_identity(g: BaseGraph) -> List[int]:
    """Riu et al.'s (id) action: a degree-2, phase-0 interior spider is a
    no-op that can be removed, directly fusing its two neighbours together
    (Duncan, Kissinger, Perdrix & van de Wetering 2020, spider-fusion
    corollary). Reuses pyzx's own verified rule
    (pyzx.rewrite_rules.remove_id_rule.check_remove_id) as the feasibility
    check, rather than re-deriving it, for the same reason the rest of this
    project prefers pyzx's own routines wherever one already exists and
    matches: it's the exact rule pyzx's own id_simp is built from
    (simplify.py), so reusing it inherits that correctness rather than
    risking a subtly different reimplementation. Not restricted to interior
    spiders the way LC/pivot are here -- an identity spider adjacent to an
    input/output boundary is equally safe to remove, and pyzx's own check
    does not distinguish the two cases."""
    from pyzx.rewrite_rules.remove_id_rule import check_remove_id
    return [v for v in g.vertices() if g.type(v) == VertexType.Z and check_remove_id(g, v)]


def find_feasible_pivot_boundary(g: BaseGraph) -> List[Tuple[int, int]]:
    """Riu et al.'s (p1) pivot-boundary variant: pivots an interior Pauli
    spider `v` against a non-Pauli Z-spider `w` that has exactly one
    boundary (input/output) neighbour. Reuses pyzx's own
    `check_pivot_boundary`, which is role-asymmetric in (v, w) -- v must be
    the interior Pauli vertex, w the boundary-adjacent non-Pauli one -- so
    both orderings of each Hadamard-connected pair are checked."""
    from pyzx.rewrite_rules.pivot_rule import check_pivot_boundary

    candidates = []
    for v1, v2 in g.edges():
        if check_pivot_boundary(g, v1, v2):
            candidates.append((v1, v2))
        elif check_pivot_boundary(g, v2, v1):
            candidates.append((v2, v1))
    return candidates


def find_feasible_pivot_gadget(g: BaseGraph) -> List[Tuple[int, int]]:
    """Riu et al.'s (p2)/(p3) pivot-gadget variants: pivots an interior
    Pauli spider `v` against an interior non-Pauli spider `w` (e.g. a T
    spider). pyzx's own `check_pivot_gadget` handles this by gadgetizing
    `w` -- moving its non-Pauli phase onto a fresh leaf spider -- so the
    pivot itself still only ever acts on a Pauli pair; this is what avoids
    needing a separate XZ/YZ-plane spider representation. Role-asymmetric
    in (v, w) like pivot-boundary above, so both orderings are checked."""
    from pyzx.rewrite_rules.pivot_rule import check_pivot_gadget

    candidates = []
    for v1, v2 in g.edges():
        if check_pivot_gadget(g, v1, v2):
            candidates.append((v1, v2))
        elif check_pivot_gadget(g, v2, v1):
            candidates.append((v2, v1))
    return candidates


# ---------------------------------------------------------------------------
# Rule application (graph surgery)
# ---------------------------------------------------------------------------


def apply_local_complementation(g: BaseGraph, v: int) -> None:
    """Eq. 2, Riu et al. Removes `v`, complements the edge set among its
    neighbours, and updates their phases by -phase(v)."""
    phase_v = g.phase(v)
    neighbours = list(g.neighbors(v))

    for n in neighbours:
        g.add_to_phase(n, -phase_v)

    for i in range(len(neighbours)):
        for j in range(i + 1, len(neighbours)):
            a, b = neighbours[i], neighbours[j]
            if g.connected(a, b):
                g.remove_edge(g.edge(a, b))
            else:
                g.add_edge(g.edge(a, b), EdgeType.HADAMARD)

    g.remove_vertex(v)


def apply_pivot(g: BaseGraph, v1: int, v2: int) -> None:
    """Eq. 3, Riu et al. Removes `v1`, `v2` and complements edges between
    the three neighbourhood subsets: unique-to-v1, unique-to-v2, shared."""
    n1 = set(g.neighbors(v1)) - {v2}
    n2 = set(g.neighbors(v2)) - {v1}
    shared = n1 & n2
    only1 = n1 - shared
    only2 = n2 - shared

    j = g.phase(v1)  # coefficient of pi for v1 (0 or 1)
    k = g.phase(v2)

    for n in only1:
        g.add_to_phase(n, k)
    for n in only2:
        g.add_to_phase(n, j)
    for n in shared:
        g.add_to_phase(n, j + k + 1)

    def toggle_between(set_a, set_b):
        for a in set_a:
            for b in set_b:
                if a == b:
                    continue
                if g.connected(a, b):
                    g.remove_edge(g.edge(a, b))
                else:
                    g.add_edge(g.edge(a, b), EdgeType.HADAMARD)

    toggle_between(only1, only2)
    toggle_between(only1, shared)
    toggle_between(only2, shared)

    g.remove_vertex(v1)
    g.remove_vertex(v2)


def apply_identity_removal(g: BaseGraph, v: int) -> None:
    """Removes identity spider `v` (degree 2, phase 0) and fuses its two
    neighbours with a single edge directly -- delegates to pyzx's own
    verified `unsafe_remove_id` (same rationale as `find_feasible_identity`
    above: reuse the exact rule pyzx's `id_simp` is built from, rather than
    reimplementing the neighbour-fusion edge-type logic by hand). Callers
    must have already confirmed `v` is feasible via
    `find_feasible_identity`; this performs no check of its own, matching
    the `apply_*` convention used elsewhere in this module.

    Unlike LC/pivot, this is NOT guaranteed to preserve graph-like form on
    its own: `unsafe_remove_id` composes the two edges around `v` using
    real ZX-calculus semantics (Hadamard . Hadamard = a plain wire), so
    removing an identity spider between two Hadamard edges produces a
    plain (non-Hadamard) edge -- correct algebra, but a violation of the
    "every wire is Hadamard" graph-like invariant `pyzx.extract_circuit`
    requires. Confirmed directly: extraction failed immediately on the
    first spot-checked circuit without this fix. `to_graph_like` restores
    the invariant (inserting a fresh identity spider to split the
    plain edge back into two Hadamard edges where needed), so it is called
    here unconditionally rather than left to each caller to remember --
    the same class of easy-to-forget post-step that caused the
    `teleport_reduce` combo bug found earlier this session."""
    from pyzx.rewrite_rules.remove_id_rule import unsafe_remove_id
    unsafe_remove_id(g, v)
    zx.simplify.to_graph_like(g)


def apply_pivot_boundary(g: BaseGraph, v: int, w: int) -> None:
    """Applies pyzx's own `unsafe_pivot_boundary(g, v, w)`: gadgetizes `w`
    (moves its non-Pauli phase onto a fresh leaf spider attached via a new
    ancilla) and then pivots `(v, w)` as an ordinary Pauli pair. Callers
    must have confirmed `(v, w)` is feasible via `find_feasible_pivot_boundary`
    (which fixes the v=interior-Pauli, w=boundary-adjacent-non-Pauli role
    order this function assumes). Empirically confirmed (see zx_utils tests
    run this session) that the result is already graph-like and
    tensor-equal to the input on every spot-checked circuit, unlike
    `apply_identity_removal` above -- no extra `to_graph_like` call needed."""
    from pyzx.rewrite_rules.pivot_rule import unsafe_pivot_boundary
    unsafe_pivot_boundary(g, v, w)


def apply_pivot_gadget(g: BaseGraph, v: int, w: int) -> None:
    """Applies pyzx's own `unsafe_pivot_gadget(g, v, w)`: gadgetizes `w`
    the same way as `apply_pivot_boundary` (moves its non-Pauli phase onto a
    fresh leaf spider), then pivots `(v, w)`. Callers must have confirmed
    `(v, w)` is feasible via `find_feasible_pivot_gadget` (role order:
    v=interior Pauli, w=interior non-Pauli). Also confirmed graph-like and
    tensor-equal on every spot-checked circuit with no extra cleanup call."""
    from pyzx.rewrite_rules.pivot_rule import unsafe_pivot_gadget
    unsafe_pivot_gadget(g, v, w)


_GRAPHS_CLONE_ATTRS = (
    "_vindex", "nedges", "_maxq", "_maxr", "_grounds", "track_phases",
    "phase_master", "max_phase_index",
)
_GRAPHS_CLONE_DICT_ATTRS = (
    "graph", "ty", "_phase", "_qindex", "_rindex", "_vdata", "_edata", "phase_index", "phase_mult",
)


def replace_in_place(g: BaseGraph, source: BaseGraph) -> None:
    """Overwrite `g`'s own internal state so its contents become identical to
    `source`'s, without reassigning the Python-level `g` reference itself --
    i.e. an in-place structural replace, mirroring `.replace()` in Holker
    (2024)'s pyzx fork (`calumholker/pyzx`, `graph/base.py`), which mainline
    pyzx (this project's dependency, pinned to 0.10.5) does not ship.

    Motivation (docs/project_log.md entry 27): the earlier, abandoned
    multi-step cflow-zx chaining attempt advanced the working graph via plain
    reference reassignment (`g = best_graph`, where `best_graph` came from
    `g.copy()`), and this is the one identified structural difference from
    Holker's own reference algorithm, which instead commits an accepted move
    via `check_g = g.clone(); ...; g.replace(check_g)`. Two things matter
    here, not one: `.clone()` (already available in mainline pyzx, unlike
    `.replace()`) preserves vertex IDs exactly, whereas `.copy()` rebuilds the
    graph with freshly renumbered, consecutive vertex indices -- silently
    invalidating any vertex ID recorded before the copy (e.g. a
    previously-scored candidate move's target vertex). `.replace()` on top of
    that keeps the graph object's own identity stable across many chained
    moves, which extract_circuit's internal bookkeeping may implicitly rely
    on in ways not confirmed by reading pyzx's source alone.

    Backend-specific: only implemented for GraphS (`backend='simple'`), the
    only backend used anywhere in this project. Raises if given anything
    else, rather than silently doing the wrong thing.
    """
    if type(g).__name__ != "GraphS" or type(source).__name__ != "GraphS":
        raise NotImplementedError(
            "replace_in_place only supports the GraphS ('simple') backend, "
            f"got {type(g).__name__} / {type(source).__name__}"
        )
    for attr in _GRAPHS_CLONE_ATTRS:
        setattr(g, attr, getattr(source, attr))
    for attr in _GRAPHS_CLONE_DICT_ATTRS:
        setattr(g, attr, getattr(source, attr).copy())
    g.scalar = source.scalar.copy()
    g._inputs = tuple(list(source._inputs))
    g._outputs = tuple(list(source._outputs))


def apply_gadget_fusion(g: BaseGraph, axel1: int, leaf1: int, axel2: int, leaf2: int) -> None:
    """Eq. 8, Riu et al. Fuses two same-target phase gadgets: sums their leaf
    phases onto leaf1 (mod 2pi) and removes the second gadget's axel and leaf
    entirely -- axel1's connections to the shared target set are untouched."""
    g.set_phase(leaf1, (g.phase(leaf1) + g.phase(leaf2)) % 2)
    g.remove_vertex(leaf2)
    g.remove_vertex(axel2)
