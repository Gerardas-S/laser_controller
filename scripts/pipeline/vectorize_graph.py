"""
Stage 3 — Graph → InternalPolyline list.

Three methods, each taking the unified `Graph` from Stage 2:

    dp_vw            — per-chain Douglas-Peucker / Visvalingam-Whyatt.  No
                       graph traversal; emits one polyline per chain.
    greedy_eulerian  — graph traversal that prefers anti-parallel continuations
                       at junctions.  Closed loops emit directly.  Currently
                       the production baseline.
    iarussi          — true Iarussi/WrapIt stochastic graph labeling +
                       per-color Hierholzer.  See iarussi.py.

Each method returns a list of InternalPolyline dicts:
    {'path': [(x, y), ...], 'closed': bool, 'node_ids': (int|None, int|None)}

`path` is in pixel coordinates; Stage 4 (postprocess) handles V-W cleanup,
intensity / color sampling, and [-1, 1] normalization.  Methods that already
need V-W during their traversal MAY apply it here (greedy_eulerian's
behavior) but the canonical V-W invocation is post-graph.
"""

import math
from typing import List
import numpy as np

from .defaults import (
    VEC_DP_DEFAULTS,
    VEC_GREEDY_EULERIAN_DEFAULTS,
    VEC_IARUSSI_DEFAULTS,
)
from .thinning import chain_tangent
from .iarussi import iarussi_label


# =============================================================================
# dp  — per-chain polyline emission (no graph traversal)
# =============================================================================
# Historically named `dp_vw` after Douglas-Peucker + Visvalingam-Whyatt; the
# V-W simplification was moved to Stage 4 (and runs for every vec method
# uniformly), and no DP step actually runs here either.  Kept under the `dp`
# label as the canonical "no graph traversal" baseline that distinguishes it
# from greedy_eulerian and iarussi (which DO traverse the graph).

def vec_dp(graph: dict, cfg: dict = None) -> List[dict]:
    """Emit one InternalPolyline per Chain.  No graph traversal.

    The actual V-W simplification + length / point filtering happens in the
    Stage 4 postprocess pass.  This stage just stitches the chain pixels
    into a path and forwards node_ids for downstream chaining.
    """
    cfg = cfg or VEC_DP_DEFAULTS
    out = []
    for c in graph['chains']:
        path = list(c['pixels'])
        if len(path) < 2:
            continue
        out.append({
            'path':     path,
            'closed':   bool(c.get('closed', False)),
            'node_ids': (c.get('node_a'), c.get('node_b')),
        })
    return out


# =============================================================================
# greedy_eulerian  — anti-parallel continuation graph traversal
# =============================================================================

def vec_greedy_eulerian(graph: dict, cfg: dict = None) -> List[dict]:
    """Greedy graph traversal: at every junction pick the unused incident
    chain whose tangent is most anti-parallel to the one just left.

    Closed-loop chains emit directly as standalone polylines.  Start nodes
    prioritise odd degree (path endpoints) then high-degree even nodes so
    polyline endpoints land on shared graph nodes whenever possible —
    encode.py uses this to chain polylines through junctions with zero
    blanking.
    """
    cfg = cfg or VEC_GREEDY_EULERIAN_DEFAULTS
    chains  = graph['chains']
    nodes   = graph['nodes']
    junction_angle_deg = cfg.get('junction_angle_deg', 30.0)
    tangent_k          = cfg.get('tangent_k', 5)

    tan_a = [None] * len(chains)
    tan_b = [None] * len(chains)
    for ci, c in enumerate(chains):
        if c.get('closed'):
            continue
        tan_a[ci] = chain_tangent(c['pixels'], end='head', k=tangent_k)
        tan_b[ci] = chain_tangent(c['pixels'], end='tail', k=tangent_k)

    incidence = {nid: [] for nid in nodes}
    for ci, c in enumerate(chains):
        if c.get('closed'):
            continue
        if c['node_a'] is not None:
            incidence[c['node_a']].append((ci, 'a'))
        if c['node_b'] is not None:
            incidence[c['node_b']].append((ci, 'b'))

    cos_th = -math.cos(math.radians(junction_angle_deg))

    visited = [False] * len(chains)
    polylines = []

    for ci, c in enumerate(chains):
        if c.get('closed') and not visited[ci]:
            visited[ci] = True
            polylines.append({
                'path':     list(c['pixels']),
                'closed':   True,
                'node_ids': (None, None),
            })

    # --- O(1) priority-bucket start-node selection ----------------------------
    # Maintain three disjoint sets of live nodes partitioned by current degree.
    # Preferred order: odd-degree (path endpoints) → high even (≥4, junctions)
    # → degree-2 (simple pass-through).  Each accepted move decrements the
    # degrees of the two endpoint nodes of the consumed chain and migrates
    # those nodes between buckets in O(1).  pick_start_node() is then O(1)
    # amortised instead of the previous O(|active|) full scan, which was
    # catastrophic for NMS-fragmented graphs (thousands of isolated 2-node
    # chains → O(N²) total).
    cur_deg: dict = {
        nid: sum(1 for ci2, _ in edges if not visited[ci2])
        for nid, edges in incidence.items()
    }

    _odd:       set = set()
    _high_even: set = set()
    _other:     set = set()
    for nid, d in cur_deg.items():
        if d == 0:
            continue
        if d % 2 == 1:
            _odd.add(nid)
        elif d >= 4:
            _high_even.add(nid)
        else:
            _other.add(nid)

    def _consume_chain(ci):
        """Mark chain visited and update degree buckets for its two endpoints."""
        visited[ci] = True
        for nid in (chains[ci]['node_a'], chains[ci]['node_b']):
            if nid is None or nid not in cur_deg:
                continue
            old_d = cur_deg[nid]
            new_d = old_d - 1
            cur_deg[nid] = new_d
            # Evict from old bucket
            if old_d % 2 == 1:   _odd.discard(nid)
            elif old_d >= 4:     _high_even.discard(nid)
            else:                _other.discard(nid)
            # Insert into new bucket (or discard if exhausted)
            if new_d > 0:
                if new_d % 2 == 1:   _odd.add(nid)
                elif new_d >= 4:     _high_even.add(nid)
                else:                _other.add(nid)

    def pick_start_node():
        for bucket in (_odd, _high_even, _other):
            while bucket:
                nid = next(iter(bucket))
                if cur_deg.get(nid, 0) > 0:
                    return nid
                bucket.discard(nid)   # stale entry
        return None

    def directed_pixels(ci, start_end):
        pixs = chains[ci]['pixels']
        return pixs if start_end == 'a' else list(reversed(pixs))

    def tangent_at(ci, end):
        return tan_a[ci] if end == 'a' else tan_b[ci]

    while True:
        start = pick_start_node()
        if start is None:
            break
        cands = [(ci, e) for ci, e in incidence[start] if not visited[ci]]
        if not cands:
            continue           # stale node slipped through; keep searching
        ci, end = cands[0]
        path = []
        node_first = start
        node_last = start
        while True:
            _consume_chain(ci)
            seg = directed_pixels(ci, end)
            if path:
                path.extend(seg[1:])
            else:
                path.extend(seg)
            far = 'b' if end == 'a' else 'a'
            cur_node = chains[ci]['node_a'] if far == 'a' else chains[ci]['node_b']
            if cur_node is None:
                node_last = None
                break
            node_last = cur_node
            cur_tan = tangent_at(ci, far)
            best = None
            best_dot = float('inf')
            for ci2, e2 in incidence[cur_node]:
                if visited[ci2]:
                    continue
                t2 = tangent_at(ci2, e2)
                if t2 is None or cur_tan is None:
                    continue
                d = cur_tan[0] * t2[0] + cur_tan[1] * t2[1]
                if d < best_dot:
                    best_dot = d
                    best = (ci2, e2)
            if best is None or best_dot > cos_th:
                break
            ci, end = best
        polylines.append({
            'path':     path,
            'closed':   False,
            'node_ids': (node_first, node_last),
        })

    return polylines


# =============================================================================
# iarussi  — stochastic graph-labeling with Eulerian extraction per color
# =============================================================================

def vec_iarussi(graph: dict, cfg: dict = None) -> List[dict]:
    """Iarussi/WrapIt stochastic graph labeler + per-color Hierholzer.

    Detailed algorithm and SA schedule in iarussi.py.  This entry point
    forwards the graph and config; the heavy lifting lives in the optimizer.
    """
    cfg = cfg or VEC_IARUSSI_DEFAULTS
    return iarussi_label(graph, cfg)
