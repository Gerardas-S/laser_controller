"""
Iarussi/WrapIt-style stochastic graph labeling for ILDA pen-up minimization.

Reference
---------
    Iarussi et al., "WrapIt: Computer-Assisted Crafting of Wire Wrapped
    Jewelry" (SIGGRAPH Asia 2015).  The labeling formulation in §4.2
    generalizes to any line-art pen-up minimization problem: partition the
    edge set of a skeleton graph into K colored connected subgraphs G_k,
    each semi-Eulerian (≤2 odd-degree vertices), minimizing an energy that
    balances K (the number of pen-ups), tangent smoothness at shared
    junctions, and length-balance to discourage degenerate stubs.

Algorithm
---------
    1. Seed: one color per connected chain component.  Repeatedly bisect any
       color that is not semi-Eulerian until all colors are valid.
    2. Simulated annealing over three moves:
         - Split   : bisect a color into two valid subgraphs.
         - Join    : merge two color-adjacent colors (must stay semi-Eulerian).
         - Permute : flip one shared-junction chain from k1 to k2.
    3. Per-color Hierholzer extracts the Eulerian path → one polyline per
       color.  Closed pure-loop chains bypass SA entirely (already trivial
       Euler circuits).

Output matches the InternalPolyline contract expected by Stage 4:
    {'path': [(x, y), ...], 'closed': bool,
     'node_ids': (first_node_id|None, last_node_id|None)}

v2 — incremental SA state
--------------------------
Root cause of v1 slowness: every SA step called _color_chains_map (O(N)),
_compute_node_degs (O(N)), and _energy (O(N×K)) from scratch, even though
each move touches only 1–2 colors.  For a ZS+HED 1080p frame with N≈1000
chains and K≈300 colors, that is ~300 000 Python dict ops per step × 2000
steps = 600 M ops ≈ 30 s.

Fix: maintain color_to_chains, node_degs, color_energy, color_lengths, and
node_to_colors as live dicts updated surgically per accepted move.  Cost per
step is O(chains_in_affected_colors) for the tangent term and O(K) for the
length-balance term — typically 100–1000× less work per step.

Note on the user-proposed adjacency-skip for join: in v2, node_to_colors is
always live so the adjacency check is O(1) (pick a random shared node and
read its color set).  Skipping adjacency would increase infeasible-proposal
rejection rate and does not improve speed here.
"""

import math
import random
import time
from typing import Dict, FrozenSet, List, Set, Tuple
from collections import defaultdict, deque

import numpy as np

from .thinning import chain_tangent


# =============================================================================
# Chain / graph helpers
# =============================================================================

def _chain_length(chain) -> float:
    pts = chain['pixels']
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        total += math.hypot(dx, dy)
    return total


def _chain_endpoints(c) -> List[int]:
    """Return the graph node ids touched by this chain (0, 1, or 2 entries)."""
    out = []
    if c['node_a'] is not None:
        out.append(c['node_a'])
    if c['node_b'] is not None:
        out.append(c['node_b'])
    return out


def _is_semi_eulerian(node_deg: Dict[int, int]) -> bool:
    return sum(1 for d in node_deg.values() if d % 2 == 1) <= 2


def _compute_deg_for_chains(chains, chain_idxs) -> Dict[int, int]:
    """Compute {node_id: degree} over the given chain indices (open chains only)."""
    deg: Dict[int, int] = {}
    for ci in chain_idxs:
        if chains[ci].get('closed', False):
            continue
        for nid in _chain_endpoints(chains[ci]):
            deg[nid] = deg.get(nid, 0) + 1
    return deg


# =============================================================================
# Tangent helpers
# =============================================================================

def _build_tangents(chains, tangent_k: int = 5):
    out = [None] * len(chains)
    for ci, c in enumerate(chains):
        if c.get('closed', False):
            out[ci] = {'a': None, 'b': None}
            continue
        out[ci] = {
            'a': chain_tangent(c['pixels'], end='head', k=tangent_k),
            'b': chain_tangent(c['pixels'], end='tail', k=tangent_k),
        }
    return out


def _per_color_tangent_energy(chains, chain_idxs, tangents) -> float:
    """Tangent energy for a single color's chain set.

    Σ_{nodes with ≥2 incident same-color half-edges} (1 + cos θ) over
    greedily-matched anti-parallel pairs.  Anti-parallel = good (cost 0);
    parallel = bad (cost 2).
    """
    node_halves: Dict[int, list] = defaultdict(list)
    for ci in chain_idxs:
        c = chains[ci]
        if c.get('closed', False):
            continue
        if c['node_a'] is not None:
            t = tangents[ci]['a']
            if t is not None:
                node_halves[c['node_a']].append(t)
        if c['node_b'] is not None:
            t = tangents[ci]['b']
            if t is not None:
                node_halves[c['node_b']].append(t)

    e = 0.0
    for halves in node_halves.values():
        if len(halves) < 2:
            continue
        angles = sorted(range(len(halves)),
                        key=lambda i: math.atan2(halves[i][1], halves[i][0]))
        for i in range(0, len(angles) - 1, 2):
            a = halves[angles[i]]
            b = halves[angles[i + 1]]
            dot = max(-1.0, min(1.0, a[0]*b[0] + a[1]*b[1]))
            e += 1.0 + dot
    return e


# =============================================================================
# Length balance (O(K) — acceptable because K ≪ N in practice)
# =============================================================================

def _length_balance(color_lengths: Dict[int, float], K: int) -> float:
    if K == 0:
        return 0.0
    vals = list(color_lengths.values())
    Lbar = sum(vals) / K
    if Lbar < 1e-6:
        return 0.0
    return sum((Lk / Lbar - 1.0) ** 2 for Lk in vals)


# =============================================================================
# Initial labeling (unchanged from v1)
# =============================================================================

def _bfs_bipartition(chains, color_chain_idxs: List[int]):
    """Bisect a color's open-chain set into two BFS halves via two-source BFS."""
    if len(color_chain_idxs) < 2:
        return None, None

    n2c: Dict[int, List[int]] = defaultdict(list)
    for ci in color_chain_idxs:
        for nid in _chain_endpoints(chains[ci]):
            n2c[nid].append(ci)

    def bfs_from(start):
        dist = {start: 0}
        q = deque([start])
        while q:
            ci = q.popleft()
            for nid in _chain_endpoints(chains[ci]):
                for nb in n2c[nid]:
                    if nb not in dist:
                        dist[nb] = dist[ci] + 1
                        q.append(nb)
        return dist

    seed_a = color_chain_idxs[0]
    dist_a = bfs_from(seed_a)
    if not dist_a:
        return None, None
    seed_b = max(dist_a, key=dist_a.get)
    if seed_b == seed_a:
        return None, None

    owner: Dict[int, int] = {seed_a: 0, seed_b: 1}
    q = deque([seed_a, seed_b])
    while q:
        ci = q.popleft()
        for nid in _chain_endpoints(chains[ci]):
            for nb in n2c[nid]:
                if nb not in owner:
                    owner[nb] = owner[ci]
                    q.append(nb)

    set_a = frozenset(ci for ci, o in owner.items() if o == 0)
    set_b = frozenset(ci for ci, o in owner.items() if o == 1)
    if len(set_a) + len(set_b) != len(color_chain_idxs) or not set_a or not set_b:
        return None, None
    return set_a, set_b


def _make_color_to_chains(label: np.ndarray) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for ci, k in enumerate(label):
        out[int(k)].append(ci)
    return dict(out)


def _seed_labeling(chains, deadline: float = None) -> np.ndarray:
    """One color per connected chain component, then bisect non-semi-Eulerian colors.

    `deadline` is an absolute time.perf_counter() value; if the repair loop
    has not finished by then it bails early (the remaining non-Eulerian colors
    will either be repaired by SA or left as-is — still a valid starting point).
    """
    N = len(chains)
    label = np.full(N, -1, dtype=np.int32)

    n2c_all: Dict[int, List[int]] = defaultdict(list)
    for ci, c in enumerate(chains):
        if c.get('closed', False):
            continue
        for nid in _chain_endpoints(c):
            n2c_all[nid].append(ci)

    next_k = 0
    for ci in range(N):
        if chains[ci].get('closed', False):
            label[ci] = next_k
            next_k += 1
            continue
        if label[ci] >= 0:
            continue
        q = deque([ci])
        label[ci] = next_k
        while q:
            cur = q.popleft()
            for nid in _chain_endpoints(chains[cur]):
                for nb in n2c_all[nid]:
                    if label[nb] < 0:
                        label[nb] = next_k
                        q.append(nb)
        next_k += 1

    repaired = True
    while repaired:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        repaired = False
        ctc = _make_color_to_chains(label)
        for k, idxs in list(ctc.items()):
            open_idxs = [i for i in idxs if not chains[i].get('closed', False)]
            if not open_idxs:
                continue
            if _is_semi_eulerian(_compute_deg_for_chains(chains, open_idxs)):
                continue
            sa, sb = _bfs_bipartition(chains, open_idxs)
            if sa is None:
                continue
            new_k = int(label.max()) + 1
            for ci in sb:
                label[ci] = new_k
            repaired = True
            break

    return label


# =============================================================================
# Incremental SA state
# =============================================================================

class _SAState:
    """Mutable SA state with O(local) incremental updates per accepted move.

    Fields maintained live:
        color_to_chains  — dict[int, list[int]]
        node_degs        — dict[int, dict[int, int]]  (per color, open chains only)
        color_energy     — dict[int, float]            (tangent term per color)
        color_lengths    — dict[int, float]            (total length per color)
        node_to_colors   — dict[int, set[int]]         (which colors share each node)
        K                — int                         (number of colors)
        _lb              — float                       (current length-balance value)
        E                — float                       (current total energy)
    """

    def __init__(self, chains, label: np.ndarray, tangents, lengths, cfg):
        self.chains   = chains
        self.tangents = tangents
        self.lengths  = lengths
        self.w_K = cfg['w_K']
        self.w_T = cfg['w_T']
        self.w_B = cfg['w_B']

        self.label           = label.copy()
        self.color_to_chains : Dict[int, List[int]]      = _make_color_to_chains(label)
        self.node_degs       : Dict[int, Dict[int, int]] = {}
        self.color_energy    : Dict[int, float]           = {}
        self.color_lengths   : Dict[int, float]           = {}
        self.node_to_colors  : Dict[int, Set[int]]        = defaultdict(set)
        self.K               = len(self.color_to_chains)

        for k, idxs in self.color_to_chains.items():
            open_idxs = [i for i in idxs if not chains[i].get('closed', False)]
            self.node_degs[k]     = _compute_deg_for_chains(chains, open_idxs)
            self.color_energy[k]  = _per_color_tangent_energy(chains, idxs, tangents)
            self.color_lengths[k] = sum(lengths[i] for i in idxs)
            for ci in open_idxs:
                for nid in _chain_endpoints(chains[ci]):
                    self.node_to_colors[nid].add(k)

        self._lb = _length_balance(self.color_lengths, self.K)
        self.E   = self._full_energy()

    def _full_energy(self) -> float:
        return (self.w_K * self.K
                + self.w_T * sum(self.color_energy.values())
                + self.w_B * self._lb)

    def _delta_lb(self, tmp_lengths: Dict[int, float], new_K: int) -> float:
        return _length_balance(tmp_lengths, new_K) - self._lb

    # ------------------------------------------------------------------
    # Permute: move chain ci from k_from → k_to
    # ------------------------------------------------------------------

    def do_permute(self, ci: int, k_from: int, k_to: int, rng, T_eff: float) -> bool:
        chains  = self.chains
        c       = chains[ci]
        eps     = _chain_endpoints(c)
        length  = self.lengths[ci]

        # Proposed new node_degs (computed without mutating)
        new_deg_from = dict(self.node_degs[k_from])
        new_deg_to   = dict(self.node_degs[k_to])
        for nid in eps:
            v = new_deg_from.get(nid, 0) - 1
            if v:
                new_deg_from[nid] = v
            else:
                new_deg_from.pop(nid, None)
            new_deg_to[nid] = new_deg_to.get(nid, 0) + 1

        if not _is_semi_eulerian(new_deg_from) or not _is_semi_eulerian(new_deg_to):
            return False

        new_idxs_from = [i for i in self.color_to_chains[k_from] if i != ci]
        new_idxs_to   = self.color_to_chains[k_to] + [ci]
        new_e_from = _per_color_tangent_energy(chains, new_idxs_from, self.tangents)
        new_e_to   = _per_color_tangent_energy(chains, new_idxs_to,   self.tangents)

        new_len_from = self.color_lengths[k_from] - length
        new_len_to   = self.color_lengths[k_to]   + length
        tmp = dict(self.color_lengths)
        tmp[k_from] = new_len_from
        tmp[k_to]   = new_len_to

        dE = (self.w_T * ((new_e_from - self.color_energy[k_from])
                          + (new_e_to  - self.color_energy[k_to]))
              + self.w_B * self._delta_lb(tmp, self.K))

        if dE >= 0 and rng.random() >= math.exp(-dE / max(1e-9, T_eff)):
            return False

        # Apply
        self.label[ci] = k_to
        self.color_to_chains[k_from].remove(ci)
        self.color_to_chains[k_to].append(ci)
        self.node_degs[k_from]    = new_deg_from
        self.node_degs[k_to]      = new_deg_to
        self.color_energy[k_from] = new_e_from
        self.color_energy[k_to]   = new_e_to
        self.color_lengths[k_from]= new_len_from
        self.color_lengths[k_to]  = new_len_to
        for nid in eps:
            if new_deg_from.get(nid, 0) == 0:
                self.node_to_colors[nid].discard(k_from)
            self.node_to_colors[nid].add(k_to)

        self._lb = _length_balance(self.color_lengths, self.K)
        self.E   = self._full_energy()
        return True

    # ------------------------------------------------------------------
    # Join: merge k_b into k_a
    # ------------------------------------------------------------------

    def do_join(self, k_a: int, k_b: int, rng, T_eff: float) -> bool:
        # Merged node degrees
        new_deg_a = dict(self.node_degs[k_a])
        for nid, d in self.node_degs[k_b].items():
            new_deg_a[nid] = new_deg_a.get(nid, 0) + d
        if not _is_semi_eulerian(new_deg_a):
            return False

        merged_idxs = self.color_to_chains[k_a] + self.color_to_chains[k_b]
        new_e_a   = _per_color_tangent_energy(self.chains, merged_idxs, self.tangents)
        new_len_a = self.color_lengths[k_a] + self.color_lengths[k_b]

        tmp = {k: v for k, v in self.color_lengths.items() if k != k_b}
        tmp[k_a] = new_len_a
        new_K = self.K - 1

        dE = (self.w_K * (-1)
              + self.w_T * (new_e_a - self.color_energy[k_a] - self.color_energy[k_b])
              + self.w_B * self._delta_lb(tmp, new_K))

        if dE >= 0 and rng.random() >= math.exp(-dE / max(1e-9, T_eff)):
            return False

        # Apply
        chains = self.chains
        b_idxs = list(self.color_to_chains[k_b])
        for ci in b_idxs:
            self.label[ci] = k_a
            for nid in _chain_endpoints(chains[ci]):
                self.node_to_colors[nid].discard(k_b)
                self.node_to_colors[nid].add(k_a)

        self.color_to_chains[k_a]  = merged_idxs
        del self.color_to_chains[k_b]
        self.node_degs[k_a]        = new_deg_a
        del self.node_degs[k_b]
        self.color_energy[k_a]     = new_e_a
        del self.color_energy[k_b]
        self.color_lengths[k_a]    = new_len_a
        del self.color_lengths[k_b]
        self.K  -= 1
        self._lb = _length_balance(self.color_lengths, self.K)
        self.E   = self._full_energy()
        return True

    # ------------------------------------------------------------------
    # Split: bisect color k.
    # open_set_b  — frozenset of open chain idxs that move to the new color.
    # open_set_a  — the remaining open chain idxs (stay with k).
    # closed_idxs — closed chains in k; always stay with k.
    # ------------------------------------------------------------------

    def do_split(self, k: int,
                 open_set_a: FrozenSet[int],
                 open_set_b: FrozenSet[int],
                 closed_idxs: List[int],
                 rng, T_eff: float) -> bool:
        chains = self.chains
        idxs_a = list(open_set_a) + closed_idxs   # stay with k
        idxs_b = list(open_set_b)                  # move to new_k

        new_deg_a = _compute_deg_for_chains(chains, idxs_a)
        new_deg_b = _compute_deg_for_chains(chains, idxs_b)
        if not _is_semi_eulerian(new_deg_a) or not _is_semi_eulerian(new_deg_b):
            return False

        new_e_a   = _per_color_tangent_energy(chains, idxs_a, self.tangents)
        new_e_b   = _per_color_tangent_energy(chains, idxs_b, self.tangents)
        new_len_a = sum(self.lengths[i] for i in idxs_a)
        new_len_b = sum(self.lengths[i] for i in idxs_b)

        _PH = -1  # placeholder key for the not-yet-created new color
        tmp = dict(self.color_lengths)
        tmp[k]   = new_len_a
        tmp[_PH] = new_len_b
        new_K = self.K + 1

        dE = (self.w_K * 1
              + self.w_T * (new_e_a + new_e_b - self.color_energy[k])
              + self.w_B * self._delta_lb(tmp, new_K))

        if dE >= 0 and rng.random() >= math.exp(-dE / max(1e-9, T_eff)):
            return False

        # Apply
        new_k = max(self.color_to_chains) + 1
        for ci in idxs_b:
            self.label[ci] = new_k

        self.color_to_chains[k]     = idxs_a
        self.color_to_chains[new_k] = idxs_b
        self.node_degs[k]           = new_deg_a
        self.node_degs[new_k]       = new_deg_b
        self.color_energy[k]        = new_e_a
        self.color_energy[new_k]    = new_e_b
        self.color_lengths[k]       = new_len_a
        self.color_lengths[new_k]   = new_len_b

        # Update node_to_colors: set_b nodes gain new_k and may lose k
        for ci in idxs_b:
            for nid in _chain_endpoints(chains[ci]):
                self.node_to_colors[nid].discard(k)
                self.node_to_colors[nid].add(new_k)
        # set_a nodes ensure k is (re-)registered (handles bridge nodes)
        for ci in idxs_a:
            if chains[ci].get('closed', False):
                continue
            for nid in _chain_endpoints(chains[ci]):
                self.node_to_colors[nid].add(k)

        self.K  += 1
        self._lb = _length_balance(self.color_lengths, self.K)
        self.E   = self._full_energy()
        return True


# =============================================================================
# Hierholzer Eulerian extraction (per color)  — unchanged from v1
# =============================================================================

def _hierholzer_extract(chains, chain_idxs: List[int],
                         nodes: dict) -> List[List[Tuple[float, float]]]:
    """Extract one or more Eulerian walks over the given chain set."""
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for ci in chain_idxs:
        c = chains[ci]
        a, b = c['node_a'], c['node_b']
        if a is None or b is None:
            continue
        adj[a].append((ci, b))
        adj[b].append((ci, a))

    if not adj:
        return [list(chains[ci]['pixels']) for ci in chain_idxs]

    consumed: Set[int] = set()

    def pick_start() -> int:
        for nid, edges in adj.items():
            live = sum(1 for ci, _ in edges if ci not in consumed)
            if live > 0 and live % 2 == 1:
                return nid
        for nid, edges in adj.items():
            if any(ci not in consumed for ci, _ in edges):
                return nid
        return -1

    def directed_pixels(ci: int, start_nid: int) -> List[Tuple[float, float]]:
        c = chains[ci]
        if c['node_a'] == start_nid:
            return list(c['pixels'])
        return list(reversed(c['pixels']))

    paths: List[List[Tuple[float, float]]] = []

    while True:
        start = pick_start()
        if start < 0:
            break
        stack       = [start]
        chain_stack : List[int] = []
        circuit     : List[int] = []
        circuit_nodes: List[int] = [start]

        while stack:
            v = stack[-1]
            chosen = None
            for ci, other in adj[v]:
                if ci not in consumed:
                    chosen = (ci, other)
                    break
            if chosen is None:
                circuit_nodes.append(stack.pop())
                if chain_stack:
                    circuit.append(chain_stack.pop())
                continue
            ci, other = chosen
            consumed.add(ci)
            stack.append(other)
            chain_stack.append(ci)

        circuit.reverse()
        circuit_nodes.reverse()
        if not circuit:
            continue

        path_pixels: List[Tuple[float, float]] = []
        cur_node = circuit_nodes[0]
        for step_idx, ci in enumerate(circuit):
            seg = directed_pixels(ci, cur_node)
            path_pixels.extend(seg if step_idx == 0 else seg[1:])
            c = chains[ci]
            cur_node = c['node_b'] if c['node_a'] == cur_node else c['node_a']
        if len(path_pixels) >= 2:
            paths.append(path_pixels)

    for ci in chain_idxs:
        if chains[ci].get('closed', False) and ci not in consumed:
            paths.append(list(chains[ci]['pixels']))
            consumed.add(ci)

    return paths


# =============================================================================
# Main entry — simulated annealing + Hierholzer
# =============================================================================

def iarussi_label(graph: dict, cfg: dict) -> List[dict]:
    """Apply Iarussi-style SA + Hierholzer; return InternalPolyline list."""
    chains = graph['chains']
    nodes  = graph['nodes']
    if not chains:
        return []

    rng = random.Random(0xC0FFEE)   # deterministic for reproducibility

    max_seconds = cfg.get('max_seconds', None)
    t_start     = time.perf_counter()
    deadline    = (t_start + max_seconds) if max_seconds else None

    tangents = _build_tangents(chains, cfg.get('tangent_k', 5))
    lengths  = [_chain_length(c) for c in chains]

    label = _seed_labeling(chains, deadline=deadline)
    state = _SAState(chains, label, tangents, lengths, cfg)

    best_label = state.label.copy()
    best_E     = state.E

    T          = cfg.get('T0', 1.0)
    T_min      = cfg.get('T_min', 0.01)
    cooling    = cfg.get('cooling', 0.995)
    n_proposals= cfg.get('n_proposals', 2000)
    early_stop = cfg.get('early_stop_rejections', 500)
    kick_thresh= cfg.get('kick_threshold', 300)
    kick_temp  = cfg.get('kick_temperature', 0.3)
    kick_dur   = cfg.get('kick_duration', 200)
    p_split    = cfg.get('p_split', 0.4)
    p_join     = cfg.get('p_join',  0.4)
    # Cool every ~5% of proposals so temperature actually drops within budget.
    N_cool     = max(1, n_proposals // 20)

    rejections = 0
    kick_left  = 0

    for step in range(n_proposals):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        T_eff = kick_temp if kick_left > 0 else T
        if kick_left > 0:
            kick_left -= 1

        if rejections >= early_stop and T_eff < T_min * 2:
            break

        r        = rng.random()
        accepted = False

        if r < p_split:
            # ---- Split ----
            candidates = [
                k for k, idxs in state.color_to_chains.items()
                if sum(1 for i in idxs if not chains[i].get('closed', False)) >= 2
            ]
            if candidates:
                k          = rng.choice(candidates)
                all_idxs   = state.color_to_chains[k]
                open_idxs  = [i for i in all_idxs if not chains[i].get('closed', False)]
                closed_idxs= [i for i in all_idxs if     chains[i].get('closed', False)]
                sa, sb     = _bfs_bipartition(chains, open_idxs)
                if sa is not None:
                    accepted = state.do_split(k, sa, sb, closed_idxs, rng, T_eff)

        elif r < p_split + p_join:
            # ---- Join ----
            # node_to_colors is live → adjacency check is O(1)
            shared = [nid for nid, cols in state.node_to_colors.items()
                      if len(cols) >= 2]
            if shared:
                nid    = rng.choice(shared)
                cols   = list(state.node_to_colors[nid])
                k_a, k_b = rng.sample(cols, 2)
                accepted = state.do_join(k_a, k_b, rng, T_eff)

        else:
            # ---- Permute ----
            shared = [nid for nid, cols in state.node_to_colors.items()
                      if len(cols) >= 2]
            if shared:
                nid      = rng.choice(shared)
                cols     = list(state.node_to_colors[nid])
                k_from, k_to = rng.sample(cols, 2)
                cands    = [ci for ci in state.color_to_chains[k_from]
                            if not chains[ci].get('closed', False)
                            and nid in _chain_endpoints(chains[ci])]
                if cands:
                    ci   = rng.choice(cands)
                    accepted = state.do_permute(ci, k_from, k_to, rng, T_eff)

        if accepted:
            rejections = 0
            if state.E < best_E:
                best_E    = state.E
                best_label= state.label.copy()
        else:
            rejections += 1
            if rejections == kick_thresh and kick_left == 0:
                kick_left = kick_dur

        if (step + 1) % N_cool == 0 and T > T_min:
            T *= cooling

    # Extract polylines from the best labeling found
    best_ctc = _make_color_to_chains(best_label)

    polylines: List[dict] = []
    for k, chain_idxs in best_ctc.items():
        paths = _hierholzer_extract(chains, chain_idxs, nodes)
        for path in paths:
            if len(path) < 2:
                continue
            polylines.append({
                'path':     path,
                'closed':   False,
                'node_ids': (None, None),
            })

    return polylines
