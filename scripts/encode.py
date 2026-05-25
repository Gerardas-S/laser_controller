#!/usr/bin/env python3
"""
Stage 3 — Temporal filter + Natural-cost ILDA bake
==================================================

Reads logical polylines (output of vectorize.py) and produces an ILDA file
whose every point is a point the DAC will play.  Each frame is emitted at
its NATURAL cost — only the points the geometry actually demands — so the
DAC firmware can auto-loop simple frames at hundreds of Hz between
RenderThread upload ticks.  No artificial frame-rate throttle, no
budget-fill resampling, no preferred-tempo metadata.

Pipeline:

    1.  Temporal persistence filter            (drop one-frame flickers)
    2.  node_ids-based polyline chaining       (zero-blanking through junctions)
    3.  Curvature-aware sample spacing         (v ≤ √(a_max/κ); straight = 2 pts)
    4.  Angle-weighted nearest-neighbour TSP   (optional, minimise inter-stroke travel)
    5.  Per-stroke quintic-eased blank travel
    6.  Pre-on / post-on / blank dwells at stroke boundaries
    7.  Menger-curvature corner dwells (with intensity compensation)

Reads
-----
    --polylines    resources/polylines/{stem}_{tag}.json   (from vectorize.py)

Writes
------
    --output       resources/animations/{stem}_{tag}.ild
"""

import argparse
import json
import math
import os
import struct
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Polylines → ILDA encoder + physics baker — Stage 3')
    p.add_argument('--polylines',  required=True, help='Input .json polylines')
    p.add_argument('--output',     required=True, help='Output .ild path')

    # Outer SAM2 contour (stylistic — bright, point-expensive ring around the
    # subject silhouette).  Drop it from the encoded animation when undesired.
    p.add_argument('--exclude-outer',  action='store_true',
                   help='Drop polylines tagged outer=True (the SAM2 silhouette).')

    # Temporal persistence
    p.add_argument('--persist-frames', type=int,   default=2)
    p.add_argument('--persist-dist',   type=int,   default=40)

    # Playback rate is decided at runtime by the C++ side (g_config.target_fps
    # = host upload cadence; DAC always plays at max_pps).  The .ild file
    # carries no tempo metadata, so there are no --target-fps / --max-pps
    # arguments here — they would be vestigial since the encoder doesn't
    # bake a per-frame point ceiling anymore.

    # Path reorder (angle-weighted TSP)
    p.add_argument('--reorder',         action='store_true',
                   help='Re-order strokes to minimise inter-stroke travel + '
                        'direction-change penalty.')
    p.add_argument('--reorder-angle-w', type=float, default=100.0,
                   help='Angle penalty weight (12-bit ILDA units per (1 - cos θ)).')

    # node_ids-based zero-blanking chaining (consumes vectorize.py metadata)
    p.add_argument('--no-chain-node-ids', action='store_true',
                   help='Disable joining of polylines that share endpoint '
                        'node_ids (debugging).  Default: chain them with no '
                        'inter-polyline travel/blank/dwell.')

    # Per-stroke boundary dwells (count in points)
    p.add_argument('--blank-points',    type=int, default=10)
    p.add_argument('--pre-on-points',   type=int, default=5)
    p.add_argument('--post-on-points',  type=int, default=5)

    # Eased blank travel
    p.add_argument('--move-speed',         type=float, default=100.0,
                   help='12-bit ILDA units per travel point.  Smaller = '
                        'more points spent on travel = smoother but slower.')
    p.add_argument('--min-travel-points',  type=int,   default=8)
    p.add_argument('--max-travel-points',  type=int,   default=80)

    # Corner dwell (Menger curvature).  Dwell points are emitted with
    # intensity scaled by 1/D so the corner does not look brighter than the
    # surrounding line — matches the advisor brightness-uniformity requirement.
    p.add_argument('--min-vertex-hold',  type=int,   default=1)
    p.add_argument('--max-vertex-hold',  type=int,   default=10)
    p.add_argument('--curve-threshold',  type=float, default=20.0,
                   help='Bend angle (degrees) below which no dwell is inserted.')
    p.add_argument('--kappa-scale',      type=float, default=500.0,
                   help='Dwell count = κ · kappa_scale, clamped to '
                        '[min_vertex_hold, max_vertex_hold].')

    # Curvature-aware lit-point spacing.  Replaces the legacy density floor
    # and arc-length-proportional budget allocator.  Each lit polyline keeps
    # only the points its curvature demands: straight segments emit ~2
    # endpoints, sharp curves pack samples densely so v ≤ sqrt(a_max / κ).
    p.add_argument('--max-accel',    type=float, default=5000.0,
                   help='Galvo acceleration ceiling in 12-bit-units per point². '
                        'Caps spacing on sharp curves via Δs = sqrt(a_max / κ). '
                        'Higher = looser corners, fewer points; lower = tighter, '
                        'more points.')
    p.add_argument('--min-spacing',  type=float, default=4.0,
                   help='Floor on per-sample step in 12-bit units (~0.1%% of '
                        'frame).  Prevents pathological over-sampling on noise.')
    p.add_argument('--max-spacing',  type=float, default=100.0,
                   help='Ceiling on per-sample step in 12-bit units.  Long '
                        'straight edges still get a handful of samples for '
                        'color interpolation.')

    # Per-frame point budget (projector viability)
    p.add_argument('--scan-rate', type=int, default=30_000,
                   help='Projector max points per second (default 30000).  '
                        'Combined with --fps to derive per-frame point budget.')
    p.add_argument('--fps',       type=int, default=30,
                   help='Playback frame rate used to derive per-frame budget '
                        '(default 30).  budget = scan_rate // fps.')
    p.add_argument('--max-pts',   type=int, default=0,
                   help='Direct per-frame point budget override.  '
                        '0 (default) = derive from scan-rate / fps.')

    return p.parse_args()


# =============================================================================
# Coordinate space
# =============================================================================
#
#   JSON (vectorize.py)   :   normalised [-1, 1]            (image y=+1 = top)
#   Encoder internals     :   12-bit ILDA float [0, 4095]   (matches HeliosConfig
#                                                            parameter scale —
#                                                            move_speed = 100,
#                                                            blank_points = 20,
#                                                            etc. all in this
#                                                            range)
#   .ild file on disk     :   16-bit signed [-32767, 32767]
#
# The 12-bit-internal choice keeps the parameter defaults familiar to anyone
# who has tuned HeliosConfig in the past — a convenient bridge during the
# migration.

def _norm_to_ilda12(x_norm, y_norm):
    return ((x_norm + 1.0) * 0.5 * 4095.0,
            (y_norm + 1.0) * 0.5 * 4095.0)


def _ilda12_to_int16(x12, y12):
    # 12-bit (0-4095, centre 2048) → 16-bit signed (-32767..32767, centre 0)
    xi = int(round((x12 - 2048.0) * 16.0))
    yi = int(round((y12 - 2048.0) * 16.0))
    return (max(-32767, min(32767, xi)),
            max(-32767, min(32767, yi)))


# =============================================================================
# Geometry helpers (operate on 12-bit ILDA float coordinates)
# =============================================================================

def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _polyline_length(pts):
    total = 0.0
    for i in range(1, len(pts)):
        total += _dist(pts[i - 1], pts[i])
    return total


def _estimate_poly_cost(p, args):
    """
    Estimate the ILDA point count for one logical polyline dict.

    Converts to 12-bit and runs _curvature_resample for an exact drawing-point
    count, then adds fixed per-stroke overhead (travel + blank dwells + pre/post
    on).  Corner dwells are omitted — small second-order effect, keeps estimate
    conservative (actual cost ≥ estimate is the safe direction for filtering).
    """
    xy     = [_norm_to_ilda12(pt[0], pt[1]) for pt in p['pts']]
    cs     = p.get('colors', [[1.0, 1.0, 1.0]] * len(p['pts']))
    pts_rs = _curvature_resample(xy, cs, args.max_accel, args.min_spacing,
                                 args.max_spacing)
    n_draw = len(pts_rs)
    fixed  = (args.min_travel_points
              + 2 * args.blank_points
              + args.pre_on_points
              + args.post_on_points)
    return n_draw + fixed


def _select_polylines_for_budget(frame_polys, budget, args):
    """
    Greedy knapsack: keep the most visually important polylines that fit within
    `budget` estimated ILDA points for the frame.

    Importance score = arc length in 12-bit units × 10 for outer=True polylines
    (SAM2 silhouette), × 1 otherwise.  Original frame order is restored in the
    returned list so downstream chaining/TSP sees a consistent sequence.
    """
    if not frame_polys or budget <= 0:
        return frame_polys

    scored = []
    for p in frame_polys:
        cost = _estimate_poly_cost(p, args)
        xy   = [_norm_to_ilda12(pt[0], pt[1]) for pt in p['pts']]
        arc  = sum(_dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1))
        score = arc * (10.0 if p.get('outer', False) else 1.0)
        scored.append((score, cost, p))

    scored.sort(key=lambda x: -x[0])   # highest importance first

    kept, total = [], 0
    for _score, cost, p in scored:
        if total + cost <= budget:
            kept.append(p)
            total += cost

    # Restore original frame order
    orig_idx = {id(p): i for i, p in enumerate(frame_polys)}
    kept.sort(key=lambda p: orig_idx[id(p)])
    return kept


def _quint_ease(t):
    """ Quintic ease-in-out — matches HeliosOutput::QuintEaseInOut. """
    if t < 0.5:
        return 16.0 * t**5
    f = (2.0 * t) - 2.0
    return 0.5 * f**5 + 1.0


def _calc_travel_points(p_from, p_to, move_speed, min_pts, max_pts):
    n = int(_dist(p_from, p_to) / move_speed)
    return max(min_pts, min(max_pts, n))


def _menger_curvature(a, b, c):
    """
    κ = 2·|cross(AB, BC)| / (|AB| · |BC| · |AC|)

    Reciprocal of the circumradius of the triangle abc — physically
    correct measure of how fast the galvo direction changes per unit
    distance.
    """
    ax, ay = b[0] - a[0], b[1] - a[1]
    bx, by = c[0] - b[0], c[1] - b[1]
    len_a  = math.hypot(ax, ay)
    len_b  = math.hypot(bx, by)
    if len_a < 1.0 or len_b < 1.0:
        return None, 0.0  # degenerate, no kappa

    # Scale-independent angle for the curve_threshold gate
    dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (len_a * len_b)))
    angle_deg = math.degrees(math.acos(dot))

    # κ
    cross_z = ax * by - ay * bx
    len_ac  = _dist(a, c)
    if len_ac > 0.5:
        kappa = 2.0 * abs(cross_z) / (len_a * len_b * len_ac)
    else:
        kappa = 2.0 / max(len_a, len_b)
    return angle_deg, kappa


def _corner_dwell(a, b, c, kappa_scale, threshold_deg, min_hold, max_hold):
    angle_deg, kappa = _menger_curvature(a, b, c)
    if angle_deg is None:
        return min_hold
    if angle_deg < threshold_deg:
        return 0
    return max(min_hold, min(max_hold, int(kappa * kappa_scale)))


# =============================================================================
# Curvature-aware lit-point spacing
# =============================================================================
#
# Every input vertex from vectorize.py is preserved (V-W already simplified
# the polyline; the encoder must not throw vertices away).  Between vertices
# we INSERT extra samples only when local curvature demands them — straight
# segments get zero inserts, sharp curves get tightly packed inserts so the
# galvo can decelerate via v ≤ sqrt(max_accel / κ).
#
# Colors are linearly interpolated along arc length per segment.
# `intensities` is read but ignored (same caveat as the previous resampler:
# raw edge-map probabilities are too low to drive the laser; would need a
# gamma/floor before re-enabling).

def _curvature_resample(pts_xy, colors, max_accel, min_spacing, max_spacing):
    """
    pts_xy      : list of (x12, y12) tuples
    colors      : list of [r, g, b] in [0, 1], same length as pts_xy
    max_accel   : galvo accel ceiling, 12-bit-units / point²
    min_spacing : floor on inserted-sample spacing, 12-bit units
    max_spacing : ceiling on inserted-sample spacing, 12-bit units

    Returns     : list of (x12, y12, r, g, b) — input vertices preserved,
                  plus curvature-driven inserts between them.
    """
    n = len(pts_xy)
    if n < 2:
        return [(x, y, c[0], c[1], c[2])
                for (x, y), c in zip(pts_xy, colors)]
    if n == 2:
        # Single straight segment — emit as-is.
        return [(pts_xy[0][0], pts_xy[0][1],
                 colors[0][0], colors[0][1], colors[0][2]),
                (pts_xy[1][0], pts_xy[1][1],
                 colors[1][0], colors[1][1], colors[1][2])]

    KAPPA_FLOOR = 1e-6   # avoid div-by-zero on collinear vertices

    def _kappa_at(i):
        """Per-vertex Menger κ; endpoints have no neighbour on one side."""
        if i <= 0 or i >= n - 1:
            return KAPPA_FLOOR
        _, k = _menger_curvature(pts_xy[i - 1], pts_xy[i], pts_xy[i + 1])
        return max(k, KAPPA_FLOOR)

    out = []
    k_left = _kappa_at(0)  # κ at the segment's left endpoint, carried across iters

    for i in range(n - 1):
        ax, ay = pts_xy[i]
        bx, by = pts_xy[i + 1]
        ca = colors[i]
        cb = colors[i + 1]

        out.append((ax, ay, ca[0], ca[1], ca[2]))

        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len >= min_spacing:
            k_right = _kappa_at(i + 1)
            # Spacing for this segment uses the worse of the two endpoint κs.
            kappa = max(k_left, k_right)
            ds    = max(min_spacing, min(max_spacing, math.sqrt(max_accel / kappa)))
            n_ins = int(seg_len / ds) - 1   # endpoint is itself a sample
            for j in range(1, n_ins + 1):
                t  = j / (n_ins + 1)
                x  = ax + (bx - ax) * t
                y  = ay + (by - ay) * t
                r  = ca[0] + (cb[0] - ca[0]) * t
                g  = ca[1] + (cb[1] - ca[1]) * t
                bc = ca[2] + (cb[2] - ca[2]) * t
                out.append((x, y, r, g, bc))
            k_left = k_right
        else:
            # Sub-min-spacing segment: don't insert anything, advance κ window.
            k_left = _kappa_at(i + 1)

    # Final vertex
    last = pts_xy[-1]
    cl   = colors[-1]
    out.append((last[0], last[1], cl[0], cl[1], cl[2]))
    return out


# =============================================================================
# Angle-weighted TSP path reorder (12-bit ILDA float space)
# =============================================================================

def _reorder_polylines(polys12, start_pos, angle_weight):
    """
    polys12      : list of dicts { 'pts': [(x12, y12), ...], 'closed': bool }
    start_pos    : (x12, y12) where the galvo is assumed to start
    angle_weight : 12-bit-ILDA-units penalty for direction reversals

    Returns      : reordered list (each polyline may have been reversed; the
                   'closed' flag prevents reversal of closed polylines).
    """
    if len(polys12) <= 1:
        return polys12

    n        = len(polys12)
    visited  = [False] * n
    ordered  = []
    cur      = start_pos
    cur_dir  = (1.0, 0.0)   # unknown initial direction → no penalty on first pick

    # Pre-compute exit directions
    fwd_exit = []   # direction of last step when traversed forward
    rev_exit = []   # direction of last step when traversed reversed
    for p in polys12:
        pts = p['pts']
        if len(pts) >= 2:
            dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
            l = math.hypot(dx, dy) or 1.0
            fwd_exit.append((dx / l, dy / l))
            dx, dy = pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]
            l = math.hypot(dx, dy) or 1.0
            rev_exit.append((dx / l, dy / l))
        else:
            fwd_exit.append((1.0, 0.0))
            rev_exit.append((-1.0, 0.0))

    for step in range(n):
        best_cost = math.inf
        best_idx  = -1
        best_rev  = False

        for i in range(n):
            if visited[i]:
                continue

            # Forward traversal: travel cur -> pts[0]
            for endpoint, traversal_rev in ((polys12[i]['pts'][0], False),
                                            (polys12[i]['pts'][-1], True)):
                if traversal_rev and polys12[i].get('closed', False):
                    continue   # don't reverse closed polylines

                dx = endpoint[0] - cur[0]
                dy = endpoint[1] - cur[1]
                dist = math.hypot(dx, dy)
                if step > 0 and dist > 0.5:
                    travel_dx = dx / dist
                    travel_dy = dy / dist
                    angle_pen = angle_weight * (1.0 - (cur_dir[0] * travel_dx +
                                                       cur_dir[1] * travel_dy))
                else:
                    angle_pen = 0.0
                cost = dist + angle_pen
                if cost < best_cost:
                    best_cost = cost
                    best_idx  = i
                    best_rev  = traversal_rev

        if best_idx < 0:
            break
        visited[best_idx] = True
        poly = polys12[best_idx]
        if best_rev:
            poly = {**poly, 'pts': list(reversed(poly['pts']))}
            cur_dir = rev_exit[best_idx]
        else:
            cur_dir = fwd_exit[best_idx]
        ordered.append(poly)
        cur = poly['pts'][-1]

    return ordered


# =============================================================================
# Natural-cost frame builder
# =============================================================================
#
# Same per-polyline insertion structure as the old BuildFrame mirror, but:
#   • Each "polyline" passed in here is a super-polyline produced by
#     _chain_by_node_ids — internally already a continuous run of pixels
#     (no inter-segment blanking).  Transitions are only inserted BETWEEN
#     super-polylines.
#   • Corner-dwell points are emitted with intensity = 1/D so a D-fold
#     repeat at a vertex doesn't visually brighten the vertex.

def _build_natural_frame(super_polys, prev_end_pos, cfg):
    """
    super_polys  : list of dicts {'pts': [(x12, y12, r, g, b), ...],
                                   'closed': bool}
                   from _chain_by_node_ids → _curvature_resample.
    prev_end_pos : (x12, y12) galvo position at frame start.
    cfg          : args namespace (blank/dwell/travel/corner params).

    Returns      : (physical_points, end_pos)
                   physical_points = [(x12, y12, r, g, b, blank), ...]
                   end_pos = position of final emitted point.
    """
    out = []
    cur = prev_end_pos

    for poly in super_polys:
        pts = poly['pts']
        if len(pts) < 2:
            continue

        first = pts[0]
        last  = pts[-1]
        first_xy = (first[0], first[1])
        last_xy  = (last[0],  last[1])

        # 1. Eased blank travel from current galvo position to stroke start
        n_travel = _calc_travel_points(cur, first_xy,
                                       cfg.move_speed,
                                       cfg.min_travel_points,
                                       cfg.max_travel_points)
        for i in range(n_travel):
            t = i / max(1, n_travel - 1)
            e = _quint_ease(t)
            x = cur[0] + (first_xy[0] - cur[0]) * e
            y = cur[1] + (first_xy[1] - cur[1]) * e
            out.append((x, y, 0.0, 0.0, 0.0, True))    # blank

        # 2. Blank dwell at stroke start
        for _ in range(cfg.blank_points):
            out.append((first_xy[0], first_xy[1], 0.0, 0.0, 0.0, True))

        # 3. Pre-on dwell
        for _ in range(cfg.pre_on_points):
            out.append((first_xy[0], first_xy[1],
                        first[2], first[3], first[4], False))

        # 4. Lit drawing with corner-dwell insertions (intensity-compensated)
        for i in range(len(pts)):
            x, y, r, g, b = pts[i]
            out.append((x, y, r, g, b, False))

            if 0 < i < len(pts) - 1:
                a  = (pts[i - 1][0], pts[i - 1][1])
                bp = (pts[i][0],     pts[i][1])
                c  = (pts[i + 1][0], pts[i + 1][1])
                dwell = _corner_dwell(
                    a, bp, c,
                    cfg.kappa_scale, cfg.curve_threshold,
                    cfg.min_vertex_hold, cfg.max_vertex_hold)
                if dwell > 0:
                    # Intensity compensation: the original vertex + dwell
                    # repeats means (1 + dwell) hits at this position.  Scale
                    # so total emitted energy ≈ a single point's energy.
                    scale = 1.0 / (dwell + 1)
                    # Retro-scale the just-emitted vertex too.
                    px, py, pr, pg, pb, pbl = out[-1]
                    out[-1] = (px, py, pr * scale, pg * scale, pb * scale, pbl)
                    for _ in range(dwell):
                        out.append((x, y, r * scale, g * scale, b * scale, False))

        # 5. Post-on dwell
        for _ in range(cfg.post_on_points):
            out.append((last_xy[0], last_xy[1],
                        last[2], last[3], last[4], False))

        # 6. Blank dwell at end
        for _ in range(cfg.blank_points):
            out.append((last_xy[0], last_xy[1], 0.0, 0.0, 0.0, True))

        cur = last_xy

    return out, cur


# =============================================================================
# Polyline chaining by shared node_ids
# =============================================================================
#
# vectorize.py / interior_skeleton_graph emits node_ids: (first_node, last_node)
# on every interior polyline.  Two polylines whose endpoint node_ids match
# end and start at the same sub-pixel position and can be drawn back-to-back
# with ZERO blanking — no travel, no blank dwell, no pre/post-on.
#
# We chain such polylines into "super-polylines" up front so the downstream
# builder treats each super as one continuous stroke.

def _chain_by_node_ids(polys12):
    """
    polys12 : list of dicts {'pts': [(x12, y12, r, g, b), ...],
                              'closed': bool,
                              'node_ids': (a, b) or None}
              (pts already curvature-resampled into the 5-tuple form).

    Returns : list of dicts {'pts': [...],
                              'closed': bool,
                              'first_node_id': int or None,
                              'last_node_id': int or None}
              Each super-poly's pts are a single concatenated stroke.
    """
    n = len(polys12)
    if n == 0:
        return []

    # Map node_id -> list of (poly_idx, 'a' or 'b') for polys that have one.
    # Closed polylines and those without node_ids are excluded from chaining.
    incidence = {}
    chainable = [False] * n
    for i, p in enumerate(polys12):
        nid = p.get('node_ids')
        if p.get('closed', False) or not nid:
            continue
        a, b = nid
        if a is None and b is None:
            continue
        chainable[i] = True
        if a is not None:
            incidence.setdefault(a, []).append((i, 'a'))
        if b is not None:
            incidence.setdefault(b, []).append((i, 'b'))

    visited = [False] * n
    out = []

    def _reversed_pts(pts):
        return list(reversed(pts))

    def _emit_pts(pts, oriented_forward):
        return list(pts) if oriented_forward else _reversed_pts(pts)

    def _grow(seed_idx):
        # Initial orientation: forward
        chain_pts = list(polys12[seed_idx]['pts'])
        head_nid  = polys12[seed_idx]['node_ids'][0]
        tail_nid  = polys12[seed_idx]['node_ids'][1]
        visited[seed_idx] = True

        # Extend at the tail.
        while tail_nid is not None:
            cands = [(j, end) for (j, end) in incidence.get(tail_nid, [])
                     if not visited[j] and chainable[j]]
            if len(cands) != 1:
                break  # 0 = nowhere to go, >1 = ambiguous junction
            j, end = cands[0]
            jpts = polys12[j]['pts']
            if end == 'a':
                # jpts[0] meets tail_nid → append forward, skip duplicate seam
                chain_pts.extend(jpts[1:])
                tail_nid = polys12[j]['node_ids'][1]
            else:
                # jpts[-1] meets tail_nid → append reversed
                chain_pts.extend(_reversed_pts(jpts)[1:])
                tail_nid = polys12[j]['node_ids'][0]
            visited[j] = True

        # Extend at the head (build prefix in reverse).
        prefix = []
        while head_nid is not None:
            cands = [(j, end) for (j, end) in incidence.get(head_nid, [])
                     if not visited[j] and chainable[j]]
            if len(cands) != 1:
                break
            j, end = cands[0]
            jpts = polys12[j]['pts']
            if end == 'b':
                # jpts[-1] meets head_nid → prepend forward
                prefix = list(jpts[:-1]) + prefix  # drop duplicate seam
                head_nid = polys12[j]['node_ids'][0]
            else:
                # jpts[0] meets head_nid → prepend reversed
                prefix = _reversed_pts(jpts)[:-1] + prefix
                head_nid = polys12[j]['node_ids'][1]
            visited[j] = True

        return {'pts': prefix + chain_pts,
                'closed': False,
                'first_node_id': head_nid,
                'last_node_id':  tail_nid}

    for i in range(n):
        if visited[i]:
            continue
        if not chainable[i]:
            # Closed polylines and node-less polylines pass through verbatim.
            p = polys12[i]
            out.append({'pts': list(p['pts']),
                        'closed': p.get('closed', False),
                        'first_node_id': None,
                        'last_node_id':  None})
            visited[i] = True
        else:
            out.append(_grow(i))

    return out


# =============================================================================
# Temporal persistence filter (unchanged from original encode.py)
# =============================================================================

def _poly_centroid_px(pts, frame_w, frame_h):
    xs = [(p[0] + 1.0) * 0.5 * frame_w for p in pts]
    ys = [(1.0 - p[1]) * 0.5 * frame_h for p in pts]
    return float(np.mean(xs)), float(np.mean(ys))


def temporal_persistence_filter(frames, frame_w, frame_h,
                                persist_frames, persist_dist):
    if persist_frames <= 1:
        return frames
    n     = len(frames)
    dist2 = persist_dist ** 2

    centroids = []
    for frame_polys in frames:
        centroids.append([
            _poly_centroid_px(poly['pts'], frame_w, frame_h)
            for poly in frame_polys
        ])

    result = [[] for _ in range(n)]
    for fi in range(n):
        required = persist_frames - 1
        for pi, poly in enumerate(frames[fi]):
            if poly.get('outer', False):
                result[fi].append(poly); continue
            cx, cy  = centroids[fi][pi]
            matched = 0
            for back in range(1, persist_frames):
                prev_fi = fi - back
                if prev_fi < 0: break
                for pcx, pcy in centroids[prev_fi]:
                    if (pcx - cx) ** 2 + (pcy - cy) ** 2 <= dist2:
                        matched += 1; break
            if matched >= required:
                result[fi].append(poly)

    before = sum(len(f) for f in frames)
    after  = sum(len(f) for f in result)
    print(f'[encode] Persistence filter: {before} → {after} polylines '
          f'(removed {before - after})', flush=True)
    return result


# =============================================================================
# ILDA Format 5 writer
# =============================================================================
#
# Each frame is now a flat list of (x12, y12, r, g, b, blank) tuples — every
# point in the file is a point the DAC will play.  Blank flag is the actual
# laser state, not a polyline separator.

def _ilda_header(frame_idx, total_frames, num_points):
    # ILDA Format 5 header (32 bytes, all multi-byte fields big-endian):
    #   0– 3  magic "ILDA"
    #   4– 6  reserved \x00\x00\x00
    #      7  format code (5 = 2-D true-colour)
    #   8–15  frame name  (8 ASCII bytes, space-padded)
    #  16–23  company name (8 ASCII bytes)
    #  24–25  number of records in this frame
    #  26–27  frame number (0-based)
    #  28–29  total frames
    #     30  projector number
    #     31  reserved
    # Company field == b'LZRBAKED' marks files as "physically baked" — every
    # point is already a DAC point.  Loaders detect this and skip BuildFrame.
    hdr  = b'ILDA'
    hdr += b'\x00\x00\x00'
    hdr += struct.pack('B', 5)
    hdr += b'frame   '
    hdr += b'LZRBAKED'
    hdr += struct.pack('>HHH', num_points, frame_idx, total_frames)
    hdr += b'\x00\x00'
    return hdr


def _ilda_point(x12, y12, r, g, b, blank):
    xi, yi = _ilda12_to_int16(x12, y12)
    status = 0x40 if blank else 0x00
    rb = max(0, min(255, int(r * 255)))
    gb = max(0, min(255, int(g * 255)))
    bb = max(0, min(255, int(b * 255)))
    return struct.pack('>hh', xi, yi) + struct.pack('BBBB',
                                                     status, bb, gb, rb)


def write_ilda_physical(path, physical_frames):
    """
    Empty frames (no polylines after filtering) are emitted as a 2-point
    blank hold at centre — this preserves the original animation frame count
    so playback timing matches the source video.  Without this, an empty
    frame is dropped from the file and the animation plays faster than
    intended.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    total = len(physical_frames)
    BLANK_HOLD = [(2048.0, 2048.0, 0.0, 0.0, 0.0, True)] * 2
    written = 0
    with open(path, 'wb') as f:
        for fi, pts in enumerate(physical_frames):
            if not pts:
                pts = BLANK_HOLD
            n = min(len(pts), 65535)
            f.write(_ilda_header(fi, total, n))
            for i in range(n):
                x, y, r, g, b, blank = pts[i]
                f.write(_ilda_point(x, y, r, g, b, blank))
            written += 1
        f.write(_ilda_header(0, total, 0))   # EOF
    print(f'[encode] Frames written: {written} / {total}', flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if not os.path.exists(args.polylines):
        print(f'[encode] Polylines file not found: {args.polylines}', flush=True)
        sys.exit(1)

    print(f'[encode] Loading {args.polylines} ...', flush=True)
    with open(args.polylines, 'r') as f:
        doc = json.load(f)

    meta   = doc['meta']
    frames = doc['frames']

    frame_w = meta['frame_w']
    frame_h = meta['frame_h']
    method  = meta.get('method', '?')

    print(f'[encode] {len(frames)} input frames  method={method}  '
          f'({frame_w}×{frame_h})', flush=True)

    # ── Optional: drop SAM2 outer-silhouette polylines ────────────────────────
    if args.exclude_outer:
        before = sum(len(fp) for fp in frames)
        frames = [[p for p in fp if not p.get('outer', False)] for fp in frames]
        after  = sum(len(fp) for fp in frames)
        print(f'[encode] --exclude-outer: dropped {before - after} polylines '
              f'({before} → {after})', flush=True)

    # ── 1. Temporal persistence filter ────────────────────────────────────────
    filtered = temporal_persistence_filter(
        frames, frame_w, frame_h,
        args.persist_frames, args.persist_dist)

    # ── 2. Per-frame baking pipeline ──────────────────────────────────────────
    # Each source animation frame is encoded at its NATURAL point count.
    # The C++ runtime plays every frame at max_pps and the DAC firmware
    # auto-loops the buffered frame between RenderThread uploads, so simple
    # frames refresh at hundreds of Hz without any in-file looping.
    #
    # The animation loops, so frame 0's entry travel comes from the LAST
    # frame's exit position.

    budget = args.max_pts if args.max_pts > 0 else args.scan_rate // args.fps
    print(f'[encode] Point budget: {budget} pts/frame  '
          f'(scan_rate={args.scan_rate}  fps={args.fps})', flush=True)

    centre = (2048.0, 2048.0)

    def _prepare_supers(frame_polys, prev_end):
        # 12-bit conversion + curvature-aware resample + node_ids chaining.
        if not frame_polys:
            return []
        polys12 = []
        for p in frame_polys:
            xy = [_norm_to_ilda12(pt[0], pt[1]) for pt in p['pts']]
            cs = p.get('colors', [[1.0, 1.0, 1.0]] * len(p['pts']))
            pts_rs = _curvature_resample(xy, cs,
                                          args.max_accel,
                                          args.min_spacing,
                                          args.max_spacing)
            polys12.append({
                'pts':      pts_rs,
                'closed':   p.get('closed', False),
                'node_ids': p.get('node_ids') if not args.no_chain_node_ids else None,
            })
        supers = _chain_by_node_ids(polys12)
        if args.reorder:
            supers = _reorder_polylines(supers, prev_end, args.reorder_angle_w)
        return supers

    # Single forward pass.  Frame 0's entry travel is baked from centre
    # (one ~16-pt blank travel on the loop-wrap — invisible).  Every later
    # frame's entry travel is baked from the prior frame's actual exit, so
    # steady-state playback is exact.
    physical_frames = []
    prev_end = centre
    n_dropped_total = 0
    for frame_polys in filtered:
        if not frame_polys:
            physical_frames.append([])
            continue
        n_before = len(frame_polys)
        frame_polys = _select_polylines_for_budget(frame_polys, budget, args)
        n_dropped_total += n_before - len(frame_polys)
        supers = _prepare_supers(frame_polys, prev_end)
        physical, prev_end = _build_natural_frame(supers, prev_end, args)
        physical_frames.append(physical)

    if n_dropped_total:
        print(f'[encode] Budget filter dropped {n_dropped_total} polylines '
              f'across all frames to stay within {budget} pts/frame.', flush=True)

    # ── 3. Write ILDA ─────────────────────────────────────────────────────────
    total_pts = sum(len(p) for p in physical_frames)
    print(f'[encode] Writing ILDA: {len(physical_frames)} frames  '
          f'{total_pts} total physical points '
          f'(avg {total_pts // max(1, len(physical_frames))}/frame)', flush=True)
    over_budget = sum(1 for p in physical_frames if len(p) > budget)
    if over_budget:
        print(f'[encode] WARNING: {over_budget} frame(s) still exceed {budget} pts '
              f'(corner-dwell overhead — within expected margin).', flush=True)

    write_ilda_physical(args.output, physical_frames)

    size_kb = os.path.getsize(args.output) / 1024
    print(f'[encode] Saved: {args.output}  ({size_kb:.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
