"""Causal flow (cflow) computation for graph-like ZX-diagrams, and the direct
two-qubit-gate-count formula it enables.

Implements Mhalla & Perdrix (2008), "Finding Optimal Flows Efficiently"
(arXiv:0709.2670), Algorithm 1 -- the O(k*n + m) causal-flow-finding algorithm.
Only the flow *existence* check and successor function are needed here (not the
partial order / layering, which that paper also computes but which this project
has no use for), so the labelling step is omitted.

Motivation (see docs/project_log.md): Riu et al. (2025) Section 3.3.2 states their
terminal reward bonus compares the agent against `cflow-zx` -- "the most
competitive ZX-calculus based optimizer for two-qubit gate reduction" -- and
explicitly warns that without a comparably strong baseline, "the agent tends to
rapidly converge to a policy of not applying actions to the circuit." This
project's environment previously compared against clifford_simp/teleport_reduce
only, which (confirmed empirically) essentially never beat the starting circuit on
several training regimes -- reproducing exactly that collapse. This module,
together with baselines.baseline_cflow_opt, is a scoped reimplementation of the
paper's approach: a real, non-trivial deterministic optimizer to compare against,
built on Holker (2024) "Causal flow preserving optimisation of quantum circuits in
the zx-calculus" (arXiv:2312.02793), Lemma 17 in particular:

    N2Q = |E| - |V| + |I|

for a graph-like ZX-diagram (without phase gadgets) that admits a causal flow --
letting the two-qubit gate count of the would-be extracted circuit be read
directly off the graph, with no extraction needed per candidate move.
"""

from typing import Dict, Optional, Set

from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType


def causal_flow(v_set: Set, i_set: Set, o_set: Set, neighbours) -> Optional[Dict]:
    """Mhalla & Perdrix (2008) Algorithm 1. `v_set` is the full vertex set of the
    open graph (V), `i_set`/`o_set` its designated inputs/outputs (I, O subset of
    V), and `neighbours(v)` returns v's neighbour set within v_set. Returns the
    causal-flow successor function {u: g(u)} (g: V\\O -> V\\I) if a causal flow
    exists, else None. Pure graph-theoretic function, no pyzx dependency, so it's
    testable against the paper's own hand-worked example independent of how the
    open-graph view of a ZX-diagram is constructed."""
    successor: Dict = {}
    out_set = set(o_set)
    candidates = out_set - i_set  # only non-inputs can correct (paper: O \ I)

    while True:
        out_prime: Set = set()
        c_prime: Set = set()
        for v in candidates:
            outside = neighbours(v) - out_set
            if len(outside) == 1:
                (u,) = outside
                successor[u] = v
                out_prime.add(u)
                c_prime.add(v)
        if not out_prime:
            return successor if out_set == v_set else None
        out_set |= out_prime
        candidates = (candidates - c_prime) | (out_prime - i_set)


def _flow_vertex_sets(g: BaseGraph):
    """Interior Z-spiders (V), and those adjacent to an input/output boundary
    vertex (I, O) -- the open-graph view of a graph-like ZX-diagram, matching
    Holker (2024) Definition 5. Assumes g is already in graph-like form."""
    v_set = {v for v in g.vertices() if g.type(v) == VertexType.Z}
    i_set = set()
    o_set = set()
    for b in g.inputs():
        i_set.update(n for n in g.neighbors(b) if g.type(n) == VertexType.Z)
    for b in g.outputs():
        o_set.update(n for n in g.neighbors(b) if g.type(n) == VertexType.Z)
    return v_set, i_set, o_set


def has_phase_gadget(g: BaseGraph, v_set: Set, i_set: Set, o_set: Set) -> bool:
    """Lemma 17 / causal flow as defined here only applies to graph-like diagrams
    without phase gadgets (Holker 2024, note after Definition 11): a spider with
    total degree 1 whose single edge is *not* to a boundary vertex -- i.e. its
    only neighbour is another interior spider, not an input/output. (A boundary
    spider with no other interior connections also has degree 1, but that's just
    an idle wire, not a gadget -- excluding i_set/o_set is what tells them apart.)
    Extending cflow to labelled graphs with gadgets (Appendix B of that paper) is
    out of scope here; callers should treat a True return as "this optimizer
    doesn't apply, fall back"."""
    boundary_adjacent = i_set | o_set
    return any(
        len(list(g.neighbors(v))) == 1 and v not in boundary_adjacent
        for v in v_set
    )


def n2q_via_causal_flow(g: BaseGraph) -> Optional[int]:
    """Holker (2024) Lemma 17. Returns the two-qubit gate count the extracted
    circuit would have, computed directly from the graph, if `g` (assumed
    graph-like) admits a causal flow and has no phase gadgets; None otherwise
    (caller should fall back to a different baseline / to real extraction)."""
    v_set, i_set, o_set = _flow_vertex_sets(g)
    if not v_set or has_phase_gadget(g, v_set, i_set, o_set):
        return None

    def neighbours(v):
        return set(g.neighbors(v)) & v_set

    if causal_flow(v_set, i_set, o_set, neighbours) is None:
        return None

    n_edges = sum(1 for v1, v2 in g.edges() if v1 in v_set and v2 in v_set)
    return n_edges - len(v_set) + len(i_set)
