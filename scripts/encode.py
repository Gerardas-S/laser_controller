#!/usr/bin/env python3
"""
Stage 3 — Temporal filter + Physical-Path Bake + ILDA encode
============================================================

Reads logical polylines (output of vectorize.py) and produces an ILDA file
that contains the *exact* point sequence the laser will play.  All geometric
optimisation that used to happen at runtime in HeliosOutput::BuildFrame is
performed here:

    1.  Temporal persistence filter            (drop one-frame flickers)
    2.  Angle-weighted nearest-neighbour TSP   (minimise inter-stroke travel)
    3.  Brightness-uniform point allocation    (longer strokes get more pts)
    4.  Quintic-eased blank travel between strokes
    5.  Pre-on / post-on / blank dwells at stroke boundaries
    6.  Menger-curvature corner dwells inside strokes

The resulting .ild file is "physically faithful" — every point in the file is
a point the DAC will play.  Other ILDA viewers will show what your laser will
draw.  HeliosOutput becomes a thin wrapper around the Helios SDK.

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

    # Temporal persistence
    p.add_argument('--persist-frames', type=int,   default=2)
    p.add_argument('--persist-dist',   type=int,   default=40)

    # Frame budget — sized to maintain a target playback rate.  The lit-point
    # budget is what's LEFT after per-polyline overhead (blanks + dwells +
    # travel) is subtracted.  This matches the legacy HeliosOutput::SendFrame
    # budgeting exactly, so output frame sizes — and therefore effective FPS —
    # are preserved across the migration.
    p.add_argument('--target-fps',  type=int, default=60,
                   help='Target playback frame rate.  budget = max-pps / fps.')
    p.add_argument('--max-pps',     type=int, default=30000,
                   help='Hardware PPS ceiling — Helios DAC = 30000.')
    p.add_argument('--min-lit-per-poly', type=int, default=2,
                   help='Floor for lit-point count per polyline when budget is '
                        'overhead-saturated.  Matches old polys*2 fallback.')

    # Path reorder (angle-weighted TSP)
    p.add_argument('--reorder',         action='store_true',
                   help='Re-order strokes to minimise inter-stroke travel + '
                        'direction-change penalty.')
    p.add_argument('--reorder-angle-w', type=float, default=100.0,
                   help='Angle penalty weight (12-bit ILDA units per (1 - cos θ)).')

    # Per-stroke boundary dwells (count in points)
    p.add_argument('--blank-points',    type=int, default=20)
    p.add_argument('--pre-on-points',   type=int, default=8)
    p.add_argument('--post-on-points',  type=int, default=8)

    # Eased blank travel
    p.add_argument('--move-speed',         type=float, default=100.0,
                   help='12-bit ILDA units per travel point.  Smaller = '
                        'more points spent on travel = smoother but slower.')
    p.add_argument('--min-travel-points',  type=int,   default=8)
    p.add_argument('--max-travel-points',  type=int,   default=80)

    # Corner dwell (Menger curvature)
    p.add_argument('--min-vertex-hold',  type=int,   default=2)
    p.add_argument('--max-vertex-hold',  type=int,   default=20)
    p.add_argument('--curve-threshold',  type=float, default=20.0,
                   help='Bend angle (degrees) below which no dwell is inserted.')
    p.add_argument('--kappa-scale',      type=float, default=500.0,
                   help='Dwell count = κ · kappa_scale, clamped to '
                        '[min_vertex_hold, max_vertex_hold].')

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
# Polyline resampling — arc-length proportional, gives uniform brightness
# =============================================================================

def _resample_polyline(pts_xy, target_count, colors=None):
    """
    pts_xy   : list of (x12, y12) tuples
    colors   : optional list of (r, g, b) per input point, will be interpolated
    Returns  : list of (x12, y12, r, g, b) tuples with len = target_count
    """
    n = len(pts_xy)
    if n < 2 or target_count < 2:
        if colors is None:
            return [(x, y, 1.0, 1.0, 1.0) for x, y in pts_xy]
        return [(x, y, *c) for (x, y), c in zip(pts_xy, colors)]

    cum = [0.0] * n
    for i in range(1, n):
        cum[i] = cum[i - 1] + _dist(pts_xy[i - 1], pts_xy[i])
    total = cum[-1]
    if total < 0.001:
        if colors is None:
            return [(x, y, 1.0, 1.0, 1.0) for x, y in pts_xy[:target_count]]
        return [(x, y, *c) for (x, y), c in
                zip(pts_xy[:target_count], colors[:target_count])]

    out      = []
    interval = total / (target_count - 1)
    seg      = 0
    for i in range(target_count):
        target_len = interval * i
        while seg < n - 2 and cum[seg + 1] < target_len:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        t = 0.0 if seg_len < 0.001 else (target_len - cum[seg]) / seg_len
        t = max(0.0, min(1.0, t))

        x = pts_xy[seg][0] + (pts_xy[seg + 1][0] - pts_xy[seg][0]) * t
        y = pts_xy[seg][1] + (pts_xy[seg + 1][1] - pts_xy[seg][1]) * t
        if colors is None:
            out.append((x, y, 1.0, 1.0, 1.0))
        else:
            r = colors[seg][0] + (colors[seg + 1][0] - colors[seg][0]) * t
            g = colors[seg][1] + (colors[seg + 1][1] - colors[seg][1]) * t
            b = colors[seg][2] + (colors[seg + 1][2] - colors[seg][2]) * t
            out.append((x, y, r, g, b))
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
# Physical frame builder — mirrors HeliosOutput::BuildFrame, in Python.
# =============================================================================

def _build_physical_frame(polys12, prev_end_pos, cfg):
    """
    polys12       : list of dicts {'pts': [(x12, y12, r, g, b), ...]}
                    pts already include per-point color; pts already
                    resampled to allocated length.
    prev_end_pos  : (x12, y12) position the galvo will be at when this
                    frame begins (last point of the previous frame).
    cfg           : namespace with all the dwell/blank/travel params.

    Returns       : (physical_points, end_pos)
                    physical_points = list of (x12, y12, r, g, b, blank)
                    end_pos = position after final lit point of the frame
    """
    out = []
    cur = prev_end_pos

    for poly in polys12:
        pts = poly['pts']
        if len(pts) < 2:
            continue

        first = pts[0]                            # (x, y, r, g, b)
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

        # 2. Blank dwell at stroke start (galvo settles, laser still off)
        for _ in range(cfg.blank_points):
            out.append((first_xy[0], first_xy[1], 0.0, 0.0, 0.0, True))

        # 3. Pre-on dwell — laser turns on, galvo already at position
        for _ in range(cfg.pre_on_points):
            out.append((first_xy[0], first_xy[1], first[2], first[3], first[4], False))

        # 4. Lit drawing with corner-dwell insertions
        for i in range(len(pts)):
            x, y, r, g, b = pts[i]
            out.append((x, y, r, g, b, False))

            if 0 < i < len(pts) - 1:
                a = (pts[i - 1][0], pts[i - 1][1])
                bp = (pts[i][0],     pts[i][1])
                c = (pts[i + 1][0], pts[i + 1][1])
                dwell = _corner_dwell(
                    a, bp, c,
                    cfg.kappa_scale, cfg.curve_threshold,
                    cfg.min_vertex_hold, cfg.max_vertex_hold)
                for _ in range(dwell):
                    out.append((x, y, r, g, b, False))

        # 5. Post-on dwell at stroke end — laser stays on briefly
        for _ in range(cfg.post_on_points):
            out.append((last_xy[0], last_xy[1], last[2], last[3], last[4], False))

        # 6. Blank dwell at end — laser off before next move
        for _ in range(cfg.blank_points):
            out.append((last_xy[0], last_xy[1], 0.0, 0.0, 0.0, True))

        cur = last_xy

    return out, cur


# =============================================================================
# Frame-budget allocator — matches HeliosOutput::SendFrame logic exactly.
# =============================================================================
#
# The frame budget is derived from PPS and FPS:  budget = max_pps / target_fps
#
# Per-polyline OVERHEAD is the cost of getting in and out of the stroke:
#       eased blank travel + start blank dwell + pre-on + post-on + end blank dwell
#
# The lit budget is what's left:
#       lit = max(budget - sum(overhead), polys × min_lit_per_poly)
#
# Lit points are then allocated proportionally to arc length so that all
# strokes draw at uniform velocity (uniform brightness).
#
# The key behaviour preserved from the legacy runtime:  when overhead exceeds
# budget (frames with many small polylines), lit collapses to the polys × 2
# floor.  The total frame size is then dominated by overhead, FPS drops
# gracefully, and the visual character — short flicks of light bridged by
# many brief blank moves — is identical to the old runtime output.

def _estimate_overhead(prev_pos, first_xy, cfg):
    """Per-polyline overhead estimate in points (matches BuildFrame exactly)."""
    travel = _calc_travel_points(prev_pos, first_xy,
                                 cfg.move_speed,
                                 cfg.min_travel_points,
                                 cfg.max_travel_points)
    # BuildFrame inserts: travel + blank-start + pre-on + post-on + blank-end
    return (travel
            + cfg.blank_points       # start blank dwell
            + cfg.pre_on_points
            + cfg.post_on_points
            + cfg.blank_points)      # end blank dwell


def _allocate_and_resample(polys_norm, prev_end_pos, cfg, frame_budget):
    """
    polys_norm    : input polylines in normalised [-1,1]
    prev_end_pos  : (x12, y12) galvo position at frame start (for travel calc)
    cfg           : args namespace
    frame_budget  : total points for the whole frame (max_pps / target_fps)
    Returns       : list of {'pts': [(x12, y12, r, g, b), ...], 'closed': bool}
                    with arc-length-proportional lit-point counts.
    """
    if not polys_norm:
        return []

    # 12-bit ILDA conversion
    xys = []
    for p in polys_norm:
        xy = [_norm_to_ilda12(pt[0], pt[1]) for pt in p['pts']]
        xys.append(xy)
    n = len(polys_norm)

    # ── Estimate per-polyline overhead in playback order ──────────────────────
    # The encoder's reorder pass (if enabled) already determined draw order.
    # Travel cost depends on previous endpoint, so walk linearly through.
    cur = prev_end_pos
    overhead_total = 0
    for xy in xys:
        if not xy:
            continue
        overhead_total += _estimate_overhead(cur, xy[0], cfg)
        cur = xy[-1]

    # ── Compute lit budget (matches OLD drawBudget) ───────────────────────────
    lit_budget = max(frame_budget - overhead_total,
                     n * cfg.min_lit_per_poly)

    # ── Allocate proportionally to arc length ─────────────────────────────────
    lengths = [_polyline_length(xy) for xy in xys]
    total_len = sum(lengths)

    out = []
    if total_len < 0.001:
        # Degenerate frame (zero-length polylines).  Distribute evenly.
        per = max(cfg.min_lit_per_poly, lit_budget // max(1, n))
        for i, xy in enumerate(xys):
            res = _resample_polyline(xy, per) if len(xy) >= 2 else []
            if res:
                out.append({'pts': res,
                            'closed': polys_norm[i].get('closed', False)})
        return out

    for i, (xy, length) in enumerate(zip(xys, lengths)):
        if len(xy) < 2:
            continue
        share = int(lit_budget * (length / total_len))
        share = max(cfg.min_lit_per_poly, share)
        # No upsample cap — letting short polylines receive their length-
        # proportional share is what makes uniform-brightness allocation work.
        res = _resample_polyline(xy, share)
        out.append({'pts': res,
                    'closed': polys_norm[i].get('closed', False)})
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

    # ── 1. Temporal persistence filter ────────────────────────────────────────
    filtered = temporal_persistence_filter(
        frames, frame_w, frame_h,
        args.persist_frames, args.persist_dist)

    # ── 2. Per-frame baking pipeline ──────────────────────────────────────────
    # The animation loops, so frame 0's entry travel comes from the LAST
    # frame's exit position.  For stationary patterns this is a no-op (frame's
    # last lit point is also its first if it's closed).  For animations it
    # eliminates the loop discontinuity after the first iteration.

    centre        = (2048.0, 2048.0)
    frame_budget  = max(64, args.max_pps // max(1, args.target_fps))

    print(f'[encode] Frame budget: {frame_budget} pts '
          f'({args.max_pps} PPS / {args.target_fps} FPS)', flush=True)

    # First pass: walk frames in order to determine each frame's exit
    # position (needed so the next frame's entry-travel is correct).  We use
    # placeholder end positions on this pass — they get refined on the second
    # pass.  For frame 0, treat the animation as a loop and use the LAST
    # frame's exit.
    end_positions = []
    cur = centre
    for frame_polys in filtered:
        if not frame_polys:
            end_positions.append(cur); continue
        allocated = _allocate_and_resample(frame_polys, cur, args, frame_budget)
        if args.reorder:
            allocated = _reorder_polylines(allocated, cur, args.reorder_angle_w)
        if allocated and allocated[-1]['pts']:
            last = allocated[-1]['pts'][-1]
            cur = (last[0], last[1])
        end_positions.append(cur)

    # Second pass: actually build, using previous-frame exit as entry-travel
    # source.  For frame 0, use the LAST frame's exit (looping behaviour).
    physical_frames = []
    n_frames = len(filtered)
    for fi, frame_polys in enumerate(filtered):
        if not frame_polys:
            physical_frames.append([]); continue

        prev_end = end_positions[(fi - 1) % n_frames] if n_frames > 0 else centre

        allocated = _allocate_and_resample(frame_polys, prev_end,
                                            args, frame_budget)
        if args.reorder:
            allocated = _reorder_polylines(allocated, prev_end, args.reorder_angle_w)

        physical, _ = _build_physical_frame(allocated, prev_end, args)
        physical_frames.append(physical)

    # ── 3. Write ILDA ─────────────────────────────────────────────────────────
    total_pts = sum(len(p) for p in physical_frames)
    print(f'[encode] Writing ILDA: {len(physical_frames)} frames  '
          f'{total_pts} total physical points '
          f'(avg {total_pts // max(1, len(physical_frames))}/frame)', flush=True)

    write_ilda_physical(args.output, physical_frames)

    size_kb = os.path.getsize(args.output) / 1024
    print(f'[encode] Saved: {args.output}  ({size_kb:.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
