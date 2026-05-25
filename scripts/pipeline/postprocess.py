"""
Stage 4 — Postprocess (shared by every pipeline).

Takes the list of InternalPolyline dicts emitted by Stage 3 and produces the
final JSONPolyline list ready for vectorize.py to serialise:

    1. Visvalingam-Whyatt simplification per polyline.
    2. Min-points / min-length filtering (drops residual speckle).
    3. Per-vertex intensity sampling from the underlying soft edge map.
    4. Per-vertex color sampling (Option D: step perpendicular into the mask).
    5. Coordinate normalisation: pixel (x, y) → [-1, 1] with y-up.
    6. Optional Catmull-Rom spline upsampling.

Also exposes `mask_outer_contours` for the SAM2 silhouette ring that
encode.py treats specially (protected from temporal-persistence drops).
"""

import os
import math
from typing import List, Optional
import cv2
import numpy as np


# =============================================================================
# Visvalingam-Whyatt simplification
# =============================================================================

def visvalingam_whyatt(pts, area_threshold):
    """Iteratively remove the interior point whose triangle (with its two
    immediate neighbours) has the smallest area, until every remaining
    triangle has area >= area_threshold.  Preserves first and last points.
    """
    if len(pts) < 3 or area_threshold <= 0:
        return list(pts)
    pts = list(pts)
    while len(pts) > 2:
        min_area = float('inf'); min_i = -1
        for i in range(1, len(pts) - 1):
            ax, ay = pts[i - 1]; bx, by = pts[i]; cx, cy = pts[i + 1]
            area = abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 0.5
            if area < min_area:
                min_area = area; min_i = i
        if min_area >= area_threshold:
            break
        del pts[min_i]
    return pts


# =============================================================================
# Per-vertex color sampling — Option D
# =============================================================================
# Edge pixels themselves carry a blended/muddy color from two neighbouring
# regions.  Option D steps perpendicular to the local edge tangent by
# `color_step` pixels, uses the SAM2 mask to pick the inside side, and
# samples a patch there.

def _sample_patch_rgb(bgr_frame, x, y, patch_size):
    H, W = bgr_frame.shape[:2]
    half = patch_size // 2
    patch = bgr_frame[max(0, y - half):min(H, y + half + 1),
                      max(0, x - half):min(W, x + half + 1)]
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    return [float(mean_bgr[2]) / 255.0,
            float(mean_bgr[1]) / 255.0,
            float(mean_bgr[0]) / 255.0]


def _sample_vertex_color(bgr_frame, mask, pts_px, i, step, patch_size):
    H, W = bgr_frame.shape[:2]
    n = len(pts_px)
    px, py = pts_px[i]

    if n == 1:
        return _sample_patch_rgb(bgr_frame, px, py, patch_size)

    if i == 0:
        dx, dy = pts_px[1][0]  - pts_px[0][0],   pts_px[1][1]  - pts_px[0][1]
    elif i == n - 1:
        dx, dy = pts_px[-1][0] - pts_px[-2][0],  pts_px[-1][1] - pts_px[-2][1]
    else:
        dx, dy = pts_px[i+1][0] - pts_px[i-1][0], pts_px[i+1][1] - pts_px[i-1][1]

    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return _sample_patch_rgb(bgr_frame, px, py, patch_size)

    nx1, ny1 = -dy / length,  dx / length
    nx2, ny2 =  dy / length, -dx / length

    x1s = int(round(px + nx1 * step)); y1s = int(round(py + ny1 * step))
    x2s = int(round(px + nx2 * step)); y2s = int(round(py + ny2 * step))

    in1 = mask is not None and 0 <= y1s < H and 0 <= x1s < W and bool(mask[y1s, x1s])
    in2 = mask is not None and 0 <= y2s < H and 0 <= x2s < W and bool(mask[y2s, x2s])

    if in1 and not in2:
        return _sample_patch_rgb(bgr_frame, x1s, y1s, patch_size)
    if in2 and not in1:
        return _sample_patch_rgb(bgr_frame, x2s, y2s, patch_size)
    if in1 and in2:
        c1 = _sample_patch_rgb(bgr_frame, x1s, y1s, patch_size)
        c2 = _sample_patch_rgb(bgr_frame, x2s, y2s, patch_size)
        return c1 if (max(c1) - min(c1)) >= (max(c2) - min(c2)) else c2
    return _sample_patch_rgb(bgr_frame, px, py, patch_size)


# =============================================================================
# Unified Stage 4 entry point
# =============================================================================

def apply_postprocess(polylines: List[dict],
                       frame_w: int, frame_h: int,
                       *,
                       soft_map: Optional[np.ndarray] = None,
                       bgr_frame: Optional[np.ndarray] = None,
                       mask: Optional[np.ndarray] = None,
                       min_pts: int = 3,
                       min_len: float = 0.0,
                       simplify_area: float = 0.0,
                       spread: float = 1.0,
                       color_step: int = 10,
                       color_patch: int = 5,
                       use_white: bool = False) -> List[dict]:
    """Convert Stage 3's InternalPolyline list to the final JSONPolyline list.

    Pipeline per polyline:
      1. Visvalingam-Whyatt simplify (if simplify_area > 0).
      2. Min-points / min-length filter.
      3. Per-vertex intensity from soft_map (or 0.7 fallback).
      4. Per-vertex color (Option D from bgr_frame + mask; else white).
      5. Normalize to [-1, 1] with y-up.
      6. Carry node_ids and closed flags through.
    """
    out: List[dict] = []
    H = W = None
    if soft_map is not None:
        H, W = soft_map.shape[:2]

    for pl in polylines:
        pts = pl['path']
        closed = bool(pl.get('closed', False))

        if simplify_area > 0 and len(pts) >= 3:
            pts = visvalingam_whyatt(pts, simplify_area)
        if len(pts) < min_pts:
            continue
        if min_len > 0:
            arc = sum(math.hypot(pts[i+1][0] - pts[i][0],
                                 pts[i+1][1] - pts[i][1])
                      for i in range(len(pts) - 1))
            if arc < min_len:
                continue

        # Intensity
        if soft_map is None:
            intensities = [0.7] * len(pts)
        else:
            intensities = []
            for (x, y) in pts:
                xi = max(0, min(W - 1, int(x)))
                yi = max(0, min(H - 1, int(y)))
                v = float(soft_map[yi, xi])
                intensities.append(max(0.0, min(1.0, v)))
            if spread != 1.0:
                m = sum(intensities) / len(intensities)
                intensities = [max(0.0, min(1.0, m + (v - m) * spread))
                               for v in intensities]

        # Color
        if use_white or bgr_frame is None:
            colors = [[1.0, 1.0, 1.0] for _ in pts]
        else:
            cH, cW = bgr_frame.shape[:2]
            pts_px_list = [(max(0, min(cW - 1, int(p[0]))),
                            max(0, min(cH - 1, int(p[1])))) for p in pts]
            colors = [_sample_vertex_color(bgr_frame, mask, pts_px_list, i,
                                            color_step, color_patch)
                      for i in range(len(pts_px_list))]

        # Normalize
        normed = [
            [ (float(pt[0]) / frame_w) * 2.0 - 1.0,
             -(float(pt[1]) / frame_h) * 2.0 + 1.0 ]
            for pt in pts
        ]

        record = {
            'pts':         normed,
            'intensities': intensities,
            'colors':      colors,
            'closed':      closed,
        }
        if pl.get('node_ids') is not None:
            record['node_ids'] = pl['node_ids']
        out.append(record)

    return out


# =============================================================================
# Outer SAM2 silhouette
# =============================================================================

def mask_outer_contours(mask_bool, frame_w, frame_h, min_area_px, epsilon,
                         *, bgr_frame=None, mask_for_color=None,
                         color_step=10, color_patch=5, use_white=False):
    """Extract the SAM2 mask outer contour as one or more closed polylines.

    Tagged `outer=True` so encode.py's temporal-persistence filter never
    drops them.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in cnts if cv2.contourArea(c) >= min_area_px]

    out = []
    for cnt in filtered:
        if len(cnt) < 2:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True) if epsilon > 0 else cnt
        pts = approx.reshape(-1, 2)
        if len(pts) < 2:
            continue

        intensities = [1.0] * len(pts)
        if use_white or bgr_frame is None:
            colors = [[1.0, 1.0, 1.0] for _ in pts]
        else:
            cH, cW = bgr_frame.shape[:2]
            pts_px_list = [(max(0, min(cW - 1, int(p[0]))),
                            max(0, min(cH - 1, int(p[1])))) for p in pts]
            colors = [_sample_vertex_color(bgr_frame, mask_for_color, pts_px_list, i,
                                            color_step, color_patch)
                      for i in range(len(pts_px_list))]

        normed = [
            [ (float(pt[0]) / frame_w) * 2.0 - 1.0,
             -(float(pt[1]) / frame_h) * 2.0 + 1.0 ]
            for pt in pts
        ]
        out.append({
            'pts':         normed,
            'intensities': intensities,
            'colors':      colors,
            'closed':      True,
            'outer':       True,
        })
    return out


# =============================================================================
# Catmull-Rom spline upsampling
# =============================================================================

def _catmull_rom_chain(pts, n_per_seg, closed):
    n = len(pts)
    if n < 2 or n_per_seg <= 1:
        return pts

    arr = np.array(pts, dtype=np.float64)

    def _get(i):
        if closed:
            return arr[i % n]
        if i < 0:
            return 2.0 * arr[0] - arr[min(-i, n - 1)]
        if i >= n:
            return 2.0 * arr[-1] - arr[max(2 * n - 2 - i, 0)]
        return arr[i]

    ts = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
    segments = n if closed else n - 1
    result = []

    for seg in range(segments):
        p0, p1, p2, p3 = _get(seg - 1), _get(seg), _get(seg + 1), _get(seg + 2)
        for t in ts:
            t2, t3 = t * t, t * t * t
            q = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            result.append([float(q[0]), float(q[1])])

    if not closed:
        result.append(list(pts[-1]))

    return result


def apply_spline_fitting(polys, n_per_seg):
    """Catmull-Rom upsample every polyline; linearly interpolate intensities
    and colors at matched t values.  No-op when n_per_seg <= 1.
    """
    if n_per_seg <= 1:
        return polys
    out = []
    for poly in polys:
        pts         = poly['pts']
        intensities = poly['intensities']
        colors      = poly.get('colors')
        closed      = poly.get('closed', False)
        if len(pts) < 2:
            out.append(poly)
            continue
        new_pts = _catmull_rom_chain(pts, n_per_seg, closed)
        n        = len(pts)
        segments = n if closed else n - 1
        ts       = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
        new_int    = []
        new_colors = [] if colors is not None else None
        for seg in range(segments):
            i_a = intensities[seg % n]
            i_b = intensities[(seg + 1) % n]
            if colors is not None:
                c_a = colors[seg % n]
                c_b = colors[(seg + 1) % n]
            for t in ts:
                new_int.append(float(i_a + (i_b - i_a) * t))
                if colors is not None:
                    new_colors.append([
                        float(c_a[0] + (c_b[0] - c_a[0]) * t),
                        float(c_a[1] + (c_b[1] - c_a[1]) * t),
                        float(c_a[2] + (c_b[2] - c_a[2]) * t),
                    ])
        if not closed:
            new_int.append(float(intensities[-1]))
            if colors is not None:
                new_colors.append([float(colors[-1][0]),
                                   float(colors[-1][1]),
                                   float(colors[-1][2])])
        new_poly = {'pts': new_pts, 'intensities': new_int, 'closed': closed}
        if new_colors is not None:
            new_poly['colors'] = new_colors
        if poly.get('outer'):
            new_poly['outer'] = True
        if poly.get('node_ids') is not None:
            new_poly['node_ids'] = poly['node_ids']
        out.append(new_poly)
    return out


# =============================================================================
# Debug rendering helper
# =============================================================================

def render_polys_to_png(polys, frame_w, frame_h, path):
    canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for p in polys:
        if not p.get('pts'):
            continue
        norm = np.array(p['pts'], dtype=np.float32)
        pix = np.empty_like(norm)
        pix[:, 0] = (norm[:, 0] + 1.0) * 0.5 * frame_w
        pix[:, 1] = (1.0 - norm[:, 1]) * 0.5 * frame_h
        pix = pix.astype(np.int32)
        cv2.polylines(canvas, [pix], bool(p.get('closed', False)), 255, 1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)
