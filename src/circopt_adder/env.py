"""Gym environment wrapping a circuit generator for single-episode ZX-diagram optimisation.

Migrated from Section 5 of the dissertation notebook. Observation is a
torch_geometric.data.Data graph: one node per ZX-spider plus one *action node* per
currently-feasible action (LC target, pivot pair, or STOP), connected to the spiders
that define them -- this is the encoding used in Fig. 4 of Riu et al., and it's what
gives you invalid-action masking "for free": the policy only ever outputs logits over
nodes that exist in the current graph.

Note (see docs/deviations_from_paper.md): the reset() baseline uses clifford_simp in
place of the original repo's zx.simplify.greedy_simp, which is not available in
pyzx 0.10.5.
"""

from fractions import Fraction
from typing import Callable, List, Optional

import gymnasium as gym
import pyzx as zx
import torch
from pyzx.graph.base import BaseGraph
from pyzx.utils import EdgeType, VertexType
from torch_geometric.data import Data

from .baselines import baseline_cflow_opt
from .config import Config
from .zx_utils import (
    apply_gadget_fusion,
    apply_identity_removal,
    apply_local_complementation,
    apply_pivot,
    apply_pivot_boundary,
    apply_pivot_gadget,
    circuit_to_graphlike,
    extract_and_cleanup,
    find_feasible_gadget_fusions,
    find_feasible_identity,
    find_feasible_lc,
    find_feasible_pivot_boundary,
    find_feasible_pivot_gadget,
    find_feasible_pivots,
    gate_count,
    t_count,
    two_qubit_gate_count,
)

PHASE_BINS = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4),
              Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]

# Shared offset for both the node action-type one-hot (at index 8+off) and the
# edge action-type one-hot (at index 2+off) below -- keeping the two aligned
# is not required by the model, but avoids one flag layout in the node loop
# and a different one in the edge loop for the same six action kinds plus STOP.
_ACTION_TYPE_OFFSET = {"LC": 0, "PIV": 1, "PIVB": 2, "PIVG": 3, "STOP": 4, "GF": 5, "ID": 6}
N_ACTION_TYPES = len(_ACTION_TYPE_OFFSET)


def _phase_onehot(phase: Fraction) -> List[float]:
    vec = [0.0] * 8
    phase_mod = phase % 2
    for i, p in enumerate(PHASE_BINS):
        if phase_mod == p:
            vec[i] = 1.0
            break
    return vec


class ZXOptEnv(gym.Env):
    """Single-episode ZX-diagram optimization environment. One "step" is
    one LC / pivot / STOP action, mirroring Riu et al. Sec 3.3."""

    metadata = {}

    def __init__(self, circuit_generator: Callable[[], "zx.Circuit"], cfg: Config):
        super().__init__()
        self.circuit_generator = circuit_generator
        self.cfg = cfg
        self.g: Optional[BaseGraph] = None
        self.steps_taken = 0
        self.initial_metric = None
        self.best_metric = None
        self.best_circuit = None
        self._action_index: List = []  # parallel list: "STOP" | ("LC", v) | ("PIV", v1, v2)
        # -- instrumentation --
        self.n_steps_total = 0
        self.n_extraction_failures = 0

    # -- core gym API --

    def reset(self, *, seed=None, options=None):
        circuit = self.circuit_generator()
        self.g = circuit_to_graphlike(circuit)
        self.steps_taken = 0
        c0 = extract_and_cleanup(self.g)
        self.initial_metric = self._metric(c0) if c0 is not None else gate_count(circuit)
        self.best_metric = self.initial_metric
        self.best_circuit = c0 if c0 is not None else circuit

        # Riu et al. Sec 3.3.2 / zx_env.py reset(): two deterministic
        # baselines computed once per episode, used for the terminal STOP
        # bonus in step() below (reward for beating them, not just for
        # reducing gates step-by-step).
        # FIX: pyzx 0.10.5 has no zx.simplify.greedy_simp; substituted clifford_simp
        g_korb = self.g.copy()
        zx.simplify.clifford_simp(g_korb)
        c_korb = extract_and_cleanup(g_korb)
        self.korb_gates = self._metric(c_korb) if c_korb is not None else self.initial_metric

        g_pyzx = self.g.copy()
        zx.simplify.teleport_reduce(g_pyzx)
        c_pyzx = extract_and_cleanup(g_pyzx)
        self.pyzx_gates = self._metric(c_pyzx) if c_pyzx is not None else self.initial_metric

        # A third, stronger deterministic baseline: Riu et al. Sec 3.3.2 states
        # their own terminal bonus compares against `cflow-zx`, "the most
        # competitive ZX-calculus based optimizer for two-qubit gate reduction",
        # and explicitly warns that without a comparably strong baseline "the
        # agent tends to rapidly converge to a policy of not applying actions to
        # the circuit" -- exactly the collapse this project's Agent A hit
        # repeatedly when only clifford_simp/teleport_reduce were available
        # (neither ever beat the starting circuit on several training regimes;
        # confirmed empirically, see docs/project_log.md). baseline_cflow_opt is
        # a scoped, verified (though single-step, not the full chained
        # algorithm -- see its docstring) reimplementation of that idea, only
        # meaningful for the twoqubits objective this project actually trains
        # on; returns None (falls back to initial_metric, i.e. no help) when it
        # doesn't apply or finds nothing better.
        if self.cfg.reward_mode == "twoqubits":
            c_cflow = baseline_cflow_opt(circuit)
            self.cflow_gates = self._metric(c_cflow) if c_cflow is not None else self.initial_metric
        else:
            self.cflow_gates = self.initial_metric

        obs = self._build_observation()
        return obs, {}

    def step(self, action_idx: int):
        action = self._action_index[action_idx]
        terminated = False
        extraction_failed = False

        if action == "STOP":
            terminated = True
            reward = 0.0
        else:
            try:
                if action[0] == "LC":
                    apply_local_complementation(self.g, action[1])
                elif action[0] == "PIV":
                    apply_pivot(self.g, action[1], action[2])
                elif action[0] == "PIVB":
                    apply_pivot_boundary(self.g, action[1], action[2])
                elif action[0] == "PIVG":
                    apply_pivot_gadget(self.g, action[1], action[2])
                elif action[0] == "GF":
                    apply_gadget_fusion(self.g, action[1], action[2], action[3], action[4])
                elif action[0] == "ID":
                    apply_identity_removal(self.g, action[1])
                c_now = extract_and_cleanup(self.g)
            except Exception:
                # Graph surgery itself can raise (e.g. a pivot target whose
                # neighbourhood shifted since find_feasible_pivots() enumerated
                # it) -- without this, an uncaught exception here kills the
                # whole training process silently. Treat exactly like a failed
                # extraction: a bad terminal move, not a crash.
                c_now = None

            if c_now is None:
                # extraction failed -- treat as a bad terminal move
                reward = -1.0
                terminated = True
                extraction_failed = True
                self.n_extraction_failures += 1
            else:
                metric_now = self._metric(c_now)
                reward = (self.best_metric - metric_now) / max(self._normalisation(), 1)
                if metric_now < self.best_metric:
                    self.best_metric = metric_now
                    self.best_circuit = c_now

        self.steps_taken += 1
        self.n_steps_total += 1
        truncated = self.steps_taken >= self.cfg.max_episode_steps
        # Riu et al. Sec 3.3.2 / zx_env.py step(): terminal bonus for
        # beating (or losing to) the deterministic pyzx/korbinian
        # baselines and the uncompressed initial circuit, on top of the
        # per-step shaping reward, whenever the episode actually ends.
        # Skipped on extraction failure, which already carries its own
        # fixed penalty above.
        if (terminated and not extraction_failed) or truncated:
            baseline = min(self.pyzx_gates, self.korb_gates, self.cflow_gates, self.initial_metric)
            reward += (baseline - self.best_metric) / self._normalisation()

        obs = self._build_observation() if not (terminated or truncated) else None
        info = {"best_metric": self.best_metric, "initial_metric": self.initial_metric, "extraction_failed": extraction_failed, "pyzx_gates": self.pyzx_gates, "korb_gates": self.korb_gates}
        return obs, reward, terminated, truncated, info

    # -- helpers --

    def _metric(self, circuit: "zx.Circuit") -> int:
        # matches Riu et al.'s released env, which hardcodes this metric to
        # two-qubit gate count (gate_type="twoqubits" in zx_env.py).
        if self.cfg.reward_mode == "t_count":
            return t_count(circuit)
        elif self.cfg.reward_mode == "total_gates":
            return gate_count(circuit)
        return two_qubit_gate_count(circuit)

    def _normalisation(self) -> float:
        # Riu et al. Sec 3.3.2 / zx_env.py: fixed compression constant,
        # not proportional to initial_metric (matches released code).
        return float(self.cfg.max_compression)

    def _build_observation(self) -> Data:
        g = self.g
        vertices = list(g.vertices())
        v_index = {v: i for i, v in enumerate(vertices)}

        node_feats = []
        for v in vertices:
            feat = _phase_onehot(g.phase(v)) if g.type(v) == VertexType.Z else [0.0] * 8
            is_boundary = 1.0 if g.type(v) == VertexType.BOUNDARY else 0.0
            feat += [is_boundary, 0.0, 0.0]  # [boundary, input, output] -- refine if I/O distinction needed
            feat += [0.0] * N_ACTION_TYPES  # action-type flags: not-an-action-node
            node_feats.append(feat)

        edge_index = [[], []]
        edge_attr = []
        for v1, v2 in g.edges():
            et = g.edge_type(g.edge(v1, v2))
            attr = [1.0, 0.0] if et == EdgeType.SIMPLE else [0.0, 1.0]
            attr += [0.0] * N_ACTION_TYPES  # reserved for action-node edge types
            i, j = v_index[v1], v_index[v2]
            edge_index[0] += [i, j]
            edge_index[1] += [j, i]
            edge_attr += [attr, attr]

        # -- action nodes --
        # All six of Riu et al.'s action types plus STOP. PIVB/PIVG (pivot-
        # boundary, pivot-gadget) reuse pyzx's own gadgetize-then-pivot
        # rewrite (see zx_utils.py) rather than a bespoke non-Clifford
        # spider representation.
        self._action_index = ["STOP"]
        for v in find_feasible_lc(g):
            self._action_index.append(("LC", v))
        for v1, v2 in find_feasible_pivots(g):
            self._action_index.append(("PIV", v1, v2))
        for v, w in find_feasible_pivot_boundary(g):
            self._action_index.append(("PIVB", v, w))
        for v, w in find_feasible_pivot_gadget(g):
            self._action_index.append(("PIVG", v, w))
        for axel1, leaf1, axel2, leaf2 in find_feasible_gadget_fusions(g):
            self._action_index.append(("GF", axel1, leaf1, axel2, leaf2))
        for v in find_feasible_identity(g):
            self._action_index.append(("ID", v))

        n_spiders = len(vertices)
        for a_offset, action in enumerate(self._action_index):
            a_idx = n_spiders + a_offset
            kind = action if action == "STOP" else action[0]
            off = _ACTION_TYPE_OFFSET[kind]

            flag = [0.0] * (8 + 3 + N_ACTION_TYPES)
            flag[8 + 3 + off] = 1.0
            node_feats.append(flag)

            if action != "STOP":  # STOP's edges are the to-every-node loop below
                edge_flag = [0.0] * (2 + N_ACTION_TYPES)
                edge_flag[2 + off] = 1.0
                for v in action[1:]:
                    i, j = v_index[v], a_idx
                    edge_index[0] += [i, j]; edge_index[1] += [j, i]
                    edge_attr += [edge_flag, edge_flag]

        # connect STOP to every other action node (Fig 4, Riu et al.)
        stop_idx = n_spiders
        stop_edge_flag = [0.0] * (2 + N_ACTION_TYPES)
        stop_edge_flag[2 + _ACTION_TYPE_OFFSET["STOP"]] = 1.0
        for a_offset in range(1, len(self._action_index)):
            a_idx = n_spiders + a_offset
            edge_index[0] += [stop_idx, a_idx]; edge_index[1] += [a_idx, stop_idx]
            edge_attr += [stop_edge_flag, stop_edge_flag]

        x = torch.tensor(node_feats, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        ea = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 2 + N_ACTION_TYPES))

        n_actions = len(self._action_index)
        action_node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        action_node_mask[n_spiders: n_spiders + n_actions] = True

        data = Data(x=x, edge_index=ei, edge_attr=ea)
        data.action_node_mask = action_node_mask
        data.n_actions = n_actions
        return data
