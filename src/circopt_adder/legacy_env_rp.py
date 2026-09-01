"""Faithful reproduction of the observation/action space Agent R and Agent P
were actually trained under, for evaluating their saved checkpoints.

Why this exists: env.py has since grown gadget fusion (added in project_log
entry 24, after R/P's training runs completed), identity removal, and the two
non-Clifford pivot variants (entry 46) -- each expansion widened the node/edge
feature dims in place, with no versioned snapshot of the intermediate states
(env.py/config.py were never committed at each stage; git history only has
one very old pre-training commit, long before R/P trained). Loading
`agent_R_random.pt`/`agent_P_pure_random.pt` into the *current* `ZXOptEnv`
would either crash on a shape mismatch or silently run them through an action
space with node/edge feature columns their weights were never trained
against.

Confirmed directly from the checkpoints' own tensors before writing this:
`input_proj.weight` is `[32, 16]` and `convs.*.lin_edge.weight` is `[32, 6]`
for both R and P -- i.e. `actor_node_feat_dim=16`, `edge_feat_dim=6`, the
same dims env.py used before today's identity-removal/pivot-boundary/
pivot-gadget work (which bumped them to 18/9). That 16/6 sizing already
reserved a 5th node action-type-flag slot and a 4th edge action-type slot
(matching the original observation-encoding text in chapter03_methodology.tex
predating any of this session's changes), but gadget fusion's candidate-
generation call was not present in the action-index loop at R/P's training
time (entry 24 confirms it was added afterward) -- so those reserved
columns were *always* zero for every training example R/P ever saw. A
`nn.Linear` column whose input is identically zero across all training
examples receives no gradient and never leaves its random initialisation, so
including gadget-fusion (or any of ID/PIVB/PIVG) action nodes at evaluation
time would route them through untrained weights, not a genuine but
untested capability. LegacyRPEnv therefore restricts the action space to
exactly STOP/LC/PIV, matching what R/P's weights actually learned from.
"""

from typing import List

import torch
from pyzx.utils import EdgeType, VertexType
from torch_geometric.data import Data

from .env import PHASE_BINS, ZXOptEnv, _phase_onehot
from .zx_utils import find_feasible_lc, find_feasible_pivots

LEGACY_ACTION_TYPE_OFFSET = {"LC": 0, "PIV": 1, "STOP": 2}  # offsets 3 (GF), 4 unused -- always zero, matching training


class LegacyRPEnv(ZXOptEnv):
    """Drop-in ZXOptEnv restricted to Agent R/P's actual training-time action
    space (STOP, LC, PIV) and observation dims (16 node / 6 edge). Everything
    else -- reset()'s baseline computation, step()'s reward/terminal-bonus
    logic -- is inherited unchanged, since none of it depends on action-space
    width."""

    def _build_observation(self) -> Data:
        g = self.g
        vertices = list(g.vertices())
        v_index = {v: i for i, v in enumerate(vertices)}

        node_feats = []
        for v in vertices:
            feat = _phase_onehot(g.phase(v)) if g.type(v) == VertexType.Z else [0.0] * 8
            is_boundary = 1.0 if g.type(v) == VertexType.BOUNDARY else 0.0
            feat += [is_boundary, 0.0, 0.0]
            feat += [0.0] * 5  # 5 reserved action-type flags (LC/PIV/STOP used; GF/unused always zero, matching training)
            node_feats.append(feat)

        edge_index = [[], []]
        edge_attr = []
        for v1, v2 in g.edges():
            et = g.edge_type(g.edge(v1, v2))
            attr = [1.0, 0.0] if et == EdgeType.SIMPLE else [0.0, 1.0]
            attr += [0.0, 0.0, 0.0, 0.0]  # 4 reserved action-edge slots (LC/PIV/STOP used; GF always zero)
            i, j = v_index[v1], v_index[v2]
            edge_index[0] += [i, j]
            edge_index[1] += [j, i]
            edge_attr += [attr, attr]

        self._action_index: List = ["STOP"]
        for v in find_feasible_lc(g):
            self._action_index.append(("LC", v))
        for v1, v2 in find_feasible_pivots(g):
            self._action_index.append(("PIV", v1, v2))

        n_spiders = len(vertices)
        for a_offset, action in enumerate(self._action_index):
            a_idx = n_spiders + a_offset
            kind = action if action == "STOP" else action[0]
            off = LEGACY_ACTION_TYPE_OFFSET[kind]

            flag = [0.0] * 16
            flag[8 + off] = 1.0
            node_feats.append(flag)

            if action != "STOP":
                edge_flag = [0.0] * 6
                edge_flag[2 + off] = 1.0
                for v in action[1:]:
                    i, j = v_index[v], a_idx
                    edge_index[0] += [i, j]; edge_index[1] += [j, i]
                    edge_attr += [edge_flag, edge_flag]

        stop_idx = n_spiders
        stop_edge_flag = [0.0] * 6
        stop_edge_flag[2 + LEGACY_ACTION_TYPE_OFFSET["STOP"]] = 1.0
        for a_offset in range(1, len(self._action_index)):
            a_idx = n_spiders + a_offset
            edge_index[0] += [stop_idx, a_idx]; edge_index[1] += [a_idx, stop_idx]
            edge_attr += [stop_edge_flag, stop_edge_flag]

        x = torch.tensor(node_feats, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        ea = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 6))

        n_actions = len(self._action_index)
        action_node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        action_node_mask[n_spiders: n_spiders + n_actions] = True

        data = Data(x=x, edge_index=ei, edge_attr=ea)
        data.action_node_mask = action_node_mask
        data.n_actions = n_actions
        return data
