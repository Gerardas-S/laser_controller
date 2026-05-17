#!/usr/bin/env python3
"""
Catmull-Rom spline demo
=======================
Generates two polylines JSON files in the same format as vectorize.py:

  resources/polylines/catmull_demo_sam2-tiny_raw.json     <- raw polylines (no spline)
  resources/polylines/catmull_demo_sam2-tiny_spline.json  <- same polylines after Catmull-Rom

Open both in the viewer and compare.  The raw file has hard corners wherever
the control points are; the spline file smooths through them.

HOW CATMULL-ROM WORKS (in our pipeline)
----------------------------------------
A standard polyline is a sequence of straight line segments connecting control
points.  Sharp direction changes at those points look mechanical and are bad for
galvanometers (hard corners → overshoot artefacts + dwell compensation needed).

Catmull-Rom replaces each straight segment with a smooth cubic curve.  For each
segment from control point P1 to control point P2, the algorithm borrows the
two *neighbouring* points (P0 before P1, P3 after P2) and fits a cubic
polynomial that:
  - passes exactly through P1 at t=0
  - passes exactly through P2 at t=1
  - arrives at P1 with a tangent parallel to the chord P0→P2
  - leaves P2 with a tangent parallel to the chord P1→P3

The formula for a point at parameter t (0..1) along the segment P1→P2 is:

  q(t) = 0.5 * (  2*P1
                 + (-P0 + P2)          * t
                 + (2*P0 - 5*P1 + 4*P2 - P3) * t²
                 + (-P0 + 3*P1 - 3*P2 + P3)  * t³  )

n_per_seg samples of t are taken uniformly over [0, 1), producing n_per_seg
new points between each pair of original control points.  The final endpoint
is appended once at the end.

Boundary handling for open curves: the phantom points P0 (before the start)
and P3 (past the end) are reflected across the endpoint so the curve reaches
the endpoint with zero curvature (natural spline behaviour).

For closed curves: indices wrap around modulo n so the curve closes smoothly.
"""

import json
import math
import os
import numpy as np

# ---------------------------------------------------------------------------
# Output paths — same naming convention as vectorize.py
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'resources', 'polylines')
RAW_PATH = os.path.join(OUT_DIR, 'catmull_demo_sam2-tiny_raw.json')
SPL_PATH = os.path.join(OUT_DIR, 'catmull_demo_sam2-tiny_spline.json')

FRAME_W = 1920
FRAME_H = 1080
N_PER_SEG = 8   # Catmull-Rom samples per segment — high value makes smoothing very visible


# ---------------------------------------------------------------------------
# Catmull-Rom — copied verbatim from vectorize.py so behaviour is identical
# ---------------------------------------------------------------------------

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

    ts       = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
    segments = n if closed else n - 1
    result   = []

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


def apply_spline(polys, n_per_seg):
    out = []
    for poly in polys:
        pts    = poly['pts']
        closed = poly.get('closed', False)
        if len(pts) < 2:
            out.append(poly)
            continue
        out.append({'pts': _catmull_rom_chain(pts, n_per_seg, closed),
                    'closed': closed})
    return out


# ---------------------------------------------------------------------------
# Demo shape generators
# All coordinates are in normalized laser space: [-1, 1] x [-1, 1], Y-up.
# ---------------------------------------------------------------------------

def make_square(cx, cy, half, closed=True):
    """A square with perfectly sharp 90-degree corners.
    Catmull-Rom will round these into smooth arcs — very visible effect."""
    pts = [
        [cx - half, cy + half],   # top-left
        [cx + half, cy + half],   # top-right
        [cx + half, cy - half],   # bottom-right
        [cx - half, cy - half],   # bottom-left
    ]
    return {'pts': pts, 'closed': closed}


def make_star(cx, cy, r_outer, r_inner, n_points, closed=True):
    """An n-pointed star with very sharp tips.
    Each tip is a hard angle — Catmull-Rom smooths them into gentle curves,
    visibly rounding off the points of the star."""
    pts = []
    for i in range(n_points * 2):
        angle = math.pi / 2 + i * math.pi / n_points   # start at top
        r = r_outer if i % 2 == 0 else r_inner
        pts.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
    return {'pts': pts, 'closed': closed}


def make_zigzag(x_start, x_end, y_center, amplitude, n_teeth, closed=False):
    """A sharp zigzag / sawtooth line.
    All vertices are hard direction reversals — Catmull-Rom turns this into
    a smooth sine-wave-like curve."""
    pts = []
    n_pts = n_teeth * 2 + 1
    for i in range(n_pts):
        x = x_start + (x_end - x_start) * i / (n_pts - 1)
        y = y_center + amplitude * (1 if i % 2 == 0 else -1)
        pts.append([x, y])
    return {'pts': pts, 'closed': closed}


def make_polygon(cx, cy, r, n_sides, closed=True):
    """A regular polygon.  With few sides (triangle, pentagon) the corners are
    wide but still softened by Catmull-Rom.  Good for comparing with the star."""
    pts = []
    for i in range(n_sides):
        angle = math.pi / 2 + 2 * math.pi * i / n_sides
        pts.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
    return {'pts': pts, 'closed': closed}


def make_l_shape(cx, cy, size, closed=False):
    """An L-shape: two sharp 90-degree corners.
    Classic example — shows how Catmull-Rom rounds interior corners."""
    h = size
    pts = [
        [cx - h,       cy + h * 0.5],   # top of vertical arm
        [cx - h,       cy - h * 0.5],   # bottom-left corner (90 deg)
        [cx + h * 0.5, cy - h * 0.5],   # bottom-right corner (90 deg)
        [cx + h * 0.5, cy - h],         # end of horizontal arm
    ]
    return {'pts': pts, 'closed': closed}


# ---------------------------------------------------------------------------
# Build frames
# ---------------------------------------------------------------------------

def build_frame():
    """One frame containing several demo shapes, spread across the canvas."""
    polys = [
        # Top-left: sharp square  →  will become rounded rectangle
        make_square(-0.65, 0.55, 0.25, closed=True),

        # Top-right: 5-pointed star  →  tips become smooth bumps
        make_star(0.6, 0.55, 0.30, 0.12, 5, closed=True),

        # Centre: zigzag  →  becomes a smooth wave
        make_zigzag(-0.80, 0.80, 0.0, 0.15, 6, closed=False),

        # Bottom-left: pentagon  →  gentle rounding of corners
        make_polygon(-0.62, -0.55, 0.28, 5, closed=True),

        # Bottom-right: L-shape  →  two interior 90-deg corners softened
        make_l_shape(0.60, -0.50, 0.25, closed=False),
    ]
    return polys


# Build N_FRAMES identical frames so the viewer can play through
N_FRAMES = 30
frames_raw = [build_frame() for _ in range(N_FRAMES)]
frames_spl = [apply_spline(build_frame(), N_PER_SEG) for _ in range(N_FRAMES)]


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def save_json(path, method, frames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {
        'meta': {
            'video':        'catmull_demo.mp4',
            'masks':        'catmull_demo_sam2-tiny.npz',
            'method':       method,
            'frame_w':      FRAME_W,
            'frame_h':      FRAME_H,
            'total_frames': len(frames),
        },
        'frames': frames,
    }
    with open(path, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f'Saved: {path}  ({size_kb:.1f} KB,  {len(frames)} frames)')


save_json(RAW_PATH, 'raw',    frames_raw)
save_json(SPL_PATH, 'spline', frames_spl)

print()
print(f'n_per_seg = {N_PER_SEG}')
print('Raw   pts per frame:', sum(len(p["pts"]) for p in frames_raw[0]))
print('Spline pts per frame:', sum(len(p["pts"]) for p in frames_spl[0]))
print()
print('Open both files in the viewer and compare.')
print('  raw    — hard corners everywhere')
print('  spline — corners rounded, zigzag becomes a wave')
