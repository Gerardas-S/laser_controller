#!/usr/bin/env python3
"""
Stage 2 — Vectorization
========================
Convert SAM2 masks + original video frames into normalized laser polylines
and save them as JSON.

Reads
-----
  --masks   resources/masks/{stem}_sam2-{model}.npz   (from segment.py)
  --video   original video file                        (needed for pixel data)

Writes
------
  Single method  (e.g. --method thin):
    resources/polylines/{stem}_sam2-{model}_thin.json

  All methods   (--method all):
    resources/polylines/{stem}_sam2-{model}_canny.json
    resources/polylines/{stem}_sam2-{model}_hed.json
    resources/polylines/{stem}_sam2-{model}_edter.json
    resources/polylines/{stem}_sam2-{model}_diffusion_edge.json

JSON schema
-----------
{
  "meta": {
    "video"   : "clip.mp4",
    "masks"   : "clip_sam2-tiny.npz",
    "method"  : "thin",
    "frame_w" : 1920,
    "frame_h" : 1080,
    "total_frames": 300
  },
  "frames": [
    [
      {"pts": [[x, y], ...], "intensities": [1.0, ...],
       "colors": [[r, g, b], ...], "closed": true, "outer": true},
      {"pts": [[x, y], ...], "intensities": [...],
       "colors": [[r, g, b], ...], "closed": false}
    ],
    ...
  ]
}

Coordinates are normalized to [-1, 1] with Y-up (laser convention).

'intensities' is per-vertex (same length as 'pts'), each value in [0, 1].
For neural edge methods it is sampled from the float edge probability map
(post-blur, pre-threshold) and optionally re-scaled by --*-spread.  For Canny
and outer SAM2 contours all intensities are 1.0.

'colors' is per-vertex RGB (same length as 'pts'), each channel in [0, 1].
Sampled from the source BGR frame using Option D: step perpendicular to the
local edge tangent by --color-step pixels, pick whichever side the SAM2 mask
marks as inside the subject, average a --color-patch BGR neighbourhood.
When --no-color is set, every vertex gets [1.0, 1.0, 1.0].

encode.py multiplies the linearly-interpolated per-point RGB by the
linearly-interpolated intensity to produce the final laser output color.
"""

import os
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                # Load the c10 runtime explicitly from the venv torch/lib folder
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        # best-effort; fall back to normal import behavior
        pass

# now safe to import torch and other modules
import torch

import argparse
import json
import math
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2

from edge_models import HEDRunner, EDTERRunner, DiffusionEdgeRunner


# =============================================================================
# Args
# =============================================================================

_ALL_METHODS = ['canny', 'hed', 'edter', 'diffusion_edge']

def parse_args():
    p = argparse.ArgumentParser(description='Mask → polyline vectorization — Stage 2')
    # I/O
    p.add_argument('--video',           required=True,
                   help='Original video file (needed for pixel data)')
    p.add_argument('--masks',           required=True,
                   help='Input .npz mask file (from segment.py)')
    p.add_argument('--output',          required=True,
                   help='Output .json path.  For --method all, _method suffixes are inserted.')
    # Method
    p.add_argument('--method',          default='hed',
                   choices=_ALL_METHODS + ['all'],
                   help='Interior edge method(s)')
    # Model paths
    p.add_argument('--hed-model',            default='bsds500')
    p.add_argument('--edter-model',          default='models/edter/EDTER-BSDS-VOC-StageI.pth')
    p.add_argument('--diffusion-edge-model',             default='models/diffusion_edge/bsds.pt')
    p.add_argument('--diffusion-edge-first-stage-model', default='models/diffusion_edge/first_stage_total_320.pt')
    p.add_argument('--diffusion-edge-config',            default='models/diffusion_edge/bsds_sample.yaml')
    p.add_argument('--device',               default='cuda' if __import__('torch').cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    # Outer contour
    p.add_argument('--min-area',        type=float, default=0.001,
                   help='Minimum mask area as fraction of frame area')
    p.add_argument('--smooth-epsilon',  type=float, default=0.0009,
                   help='approxPolyDP epsilon for outer contours as fraction of '
                        'image diagonal.  Converted to pixels at runtime.')
    # Frame averaging
    p.add_argument('--frame-avg-alpha', type=float, default=0.7,
                   help='Signal temporal blend: 1.0=off, lower=more averaging')
    # Canny
    p.add_argument('--canny-low',       type=int,   default=40)
    p.add_argument('--canny-high',      type=int,   default=120)
    p.add_argument('--canny-blur',      type=int,   default=3)
    p.add_argument('--canny-epsilon',   type=float, default=0.0007,
                   help='approxPolyDP epsilon as fraction of image diagonal.')
    p.add_argument('--canny-min-pts',   type=int,   default=4)
    # HED  (NMS + gap-bridging closing + component filter pipeline)
    p.add_argument('--hed-threshold',       type=float, default=0.10,
                   help='Noise-floor threshold on the NMS-thinned HED map. '
                        'NMS-surviving pixels below this value are dropped '
                        'before gap-bridging.  Lower = retain dimmer ridge '
                        'continuations.')
    p.add_argument('--hed-threshold-high',  type=float, default=0.25,
                   help='Anchor threshold on the NMS-thinned HED map.  After '
                        'morphological closing bridges small NMS gaps, '
                        'connected components are retained only if they '
                        'contain at least one pixel at or above this value. '
                        'This rejects isolated noise fragments that NMS '
                        'preserved but are not part of real edges.')
    p.add_argument('--hed-bridge-kernel',   type=int,   default=8,
                   help='Maximum bridge distance (in pixels) for soft-guided '
                        'gap bridging.  Each NMS pixel is allowed to extend '
                        'up to this many pixels outward, but ONLY into areas '
                        'where the soft HED map indicates an edge (>= '
                        '--hed-threshold).  Bridges form where two NMS '
                        'fragments are connected by a high-probability '
                        'corridor in the soft map.  Unrelated nearby ridges '
                        'do NOT merge because the soft map is low between '
                        'them.  Set to 1 to disable bridging.')
    p.add_argument('--hed-blur',            type=int,   default=1,
                   help='(Unused.)  Retained for argparse compatibility; the '
                        'HED pipeline does no pre-blur.')
    p.add_argument('--hed-epsilon',     type=float, default=0.0007,
                   help='approxPolyDP epsilon as fraction of image diagonal. '
                        'Lower = more detail preserved.')
    p.add_argument('--hed-min-pts',     type=int,   default=3)
    p.add_argument('--hed-min-len',     type=int,   default=40,
                   help='Minimum arc length in pixels. Strokes shorter than this '
                        'are discarded as speckle.')
    p.add_argument('--hed-spread',      type=float, default=1.0,
                   help='Per-polyline intensity contrast: '
                        'new = clamp(mean + (raw - mean) * spread, 0, 1). '
                        '1.0=no change, >1=more contrast, <1=softer, 0=uniform.')
    # EDTER
    p.add_argument('--edter-threshold',  type=float, default=0.35)
    p.add_argument('--edter-blur',       type=int,   default=3)
    p.add_argument('--edter-epsilon',    type=float, default=0.0007,
                   help='approxPolyDP epsilon as fraction of image diagonal.')
    p.add_argument('--edter-min-pts',    type=int,   default=3)
    p.add_argument('--edter-min-len',    type=int,   default=40)
    p.add_argument('--edter-spread',     type=float, default=1.0,
                   help='Per-polyline intensity contrast modulator (see --hed-spread).')
    # DiffusionEdge
    p.add_argument('--diffusion-edge-threshold', type=float, default=0.35)
    p.add_argument('--diffusion-edge-blur',      type=int,   default=3)
    p.add_argument('--diffusion-edge-epsilon',   type=float, default=0.0007,
                   help='approxPolyDP epsilon as fraction of image diagonal.')
    p.add_argument('--diffusion-edge-min-pts',   type=int,   default=3)
    p.add_argument('--diffusion-edge-min-len',   type=int,   default=40)
    p.add_argument('--diffusion-edge-spread',    type=float, default=1.0,
                   help='Per-polyline intensity contrast modulator (see --hed-spread).')
    p.add_argument('--diffusion-edge-steps',     type=int,   default=5,
                   help='Denoising steps for DiffusionEdge (fewer = faster, less crisp)')
    # Spline fitting (applied to all methods)
    p.add_argument('--spline-samples',  type=int,   default=1,
                   help='Catmull-Rom samples per segment between DP vertices '
                        '(1=off, 4=smooth, 8=very smooth)')
    # Color sampling (Option D: step perpendicular into the SAM2 mask)
    p.add_argument('--color-step',  type=int,  default=10,
                   help='Pixels to step from edge vertex into the SAM2 mask for '
                        'color sampling.  Larger values clear the blended edge '
                        'fringe but risk landing on a differently-coloured '
                        'sub-region.  Default: 10.')
    p.add_argument('--color-patch', type=int,  default=5,
                   help='Patch size (NxN) for BGR averaging at each sample '
                        'point.  Default: 5.')
    p.add_argument('--no-color',    action='store_true',
                   help='Disable color sampling; every vertex emits white '
                        '(1,1,1).  Useful for white-light mode.')
    return p.parse_args()


# =============================================================================
# Shared helpers
# =============================================================================

_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def _erode_mask(mask_u8):
    inner = cv2.erode(mask_u8, _ERODE_KERNEL, iterations=1)
    return inner if cv2.countNonZero(inner) > 0 else None


def _nms_edge_map(prob_map):
    """Non-maximum suppression on a soft edge-probability map.

    The map is treated as a ridge field: the gradient of the map points
    perpendicular to each edge ridge.  At every pixel we compare its value
    to the two neighbours along that gradient direction (quantised to 4
    directions: H, V, /, \\).  Pixels that are not local maxima along the
    gradient are zeroed.  The output is a thin (1-pixel-wide) ridge map
    that preserves the original float intensities at surviving pixels.

    This is the right thin-ing operation for HED/EDTER output because it
    operates on the float map *before* any binary decision -- weak-but-real
    edges that would have been discarded by a hard threshold survive as
    long as they form a local ridge.

    Input:  float32 [0, 1] of shape (H, W)
    Output: float32 [0, 1] of shape (H, W), thinned to single-pixel ridges.
    """
    # Use a larger Sobel kernel: gradient direction at the *peak* of a soft
    # ridge is dominated by noise with ksize=3 because the magnitude is
    # near zero there.  ksize=5 averages over a wider neighbourhood and
    # gives a stable perpendicular-to-ridge direction even at ridge tops.
    gx = cv2.Sobel(prob_map, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(prob_map, cv2.CV_32F, 0, 1, ksize=5)
    angle = np.arctan2(gy, gx)
    # Quantise to 4 directions: 0=H, 1=/, 2=V, 3=\
    q = (np.round(angle / (np.pi / 4)) % 4).astype(np.int8)

    # Where the gradient is too weak to give a reliable direction (flat
    # plateaus, exact ridge tops) we skip suppression and keep the pixel
    # unconditionally if its value clears the threshold downstream.  This
    # preserves the centerline of wide soft ridges that would otherwise be
    # perforated by random-direction comparisons.
    gmag = cv2.magnitude(gx, gy)
    weak_grad = gmag < 1e-3

    p = prob_map
    out = np.zeros_like(p)

    # Asymmetric comparison (>= on one side, > on the other) ensures exactly
    # one pixel survives per flat plateau -- with strict > on both sides,
    # quantisation-induced plateaus suppress everything.

    # Direction 0 (horizontal gradient -> vertical ridge): compare left/right
    m = (q == 0)
    keep = (p >= np.roll(p, 1, axis=1)) & (p > np.roll(p, -1, axis=1))
    out[m & keep] = p[m & keep]

    # Direction 1 (/ gradient): compare TL <-> BR
    m = (q == 1)
    keep = (p >= np.roll(np.roll(p, 1, axis=0), 1, axis=1)) & \
           (p >  np.roll(np.roll(p, -1, axis=0), -1, axis=1))
    out[m & keep] = p[m & keep]

    # Direction 2 (vertical gradient -> horizontal ridge): compare up/down
    m = (q == 2)
    keep = (p >= np.roll(p, 1, axis=0)) & (p > np.roll(p, -1, axis=0))
    out[m & keep] = p[m & keep]

    # Direction 3 (\ gradient): compare TR <-> BL
    m = (q == 3)
    keep = (p >= np.roll(np.roll(p, 1, axis=0), -1, axis=1)) & \
           (p >  np.roll(np.roll(p, -1, axis=0), 1, axis=1))
    out[m & keep] = p[m & keep]

    # Flat-top pixels: keep their value unconditionally
    out[weak_grad] = p[weak_grad]

    return out


def _render_polys_to_png(polys, frame_w, frame_h, path):
    """Render normalized polylines (coords in [-1,1], y flipped) to a black
    canvas and write a debug PNG.  Used for inspecting approxPolyDP / spline
    output on frame 0."""
    canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for p in polys:
        if not p.get('pts'):
            continue
        norm = np.array(p['pts'], dtype=np.float32)
        # invert: x_px = (x_norm + 1)/2 * W ;  y_px = (1 - y_norm)/2 * H
        pix = np.empty_like(norm)
        pix[:, 0] = (norm[:, 0] + 1.0) * 0.5 * frame_w
        pix[:, 1] = (1.0 - norm[:, 1]) * 0.5 * frame_h
        pix = pix.astype(np.int32)
        cv2.polylines(canvas, [pix], bool(p.get('closed', False)), 255, 1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)


# -----------------------------------------------------------------------------
# Per-vertex color sampling (Option D)
# -----------------------------------------------------------------------------
# Edge pixels themselves carry a blended/muddy color from two neighbouring
# regions.  Option D steps perpendicular to the local edge tangent by
# `color_step` pixels, uses the SAM2 mask to determine which side is inside
# the subject, and samples a small patch there.  Fallbacks: if both sides
# are inside the mask the higher-saturation patch wins; if neither side is
# inside, sample at the vertex itself.

def _sample_patch_rgb(bgr_frame, x, y, patch_size):
    """Mean BGR patch at (x, y) -> [r, g, b] float [0, 1]."""
    H, W = bgr_frame.shape[:2]
    half = patch_size // 2
    patch = bgr_frame[max(0, y - half):min(H, y + half + 1),
                      max(0, x - half):min(W, x + half + 1)]
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    return [float(mean_bgr[2]) / 255.0,
            float(mean_bgr[1]) / 255.0,
            float(mean_bgr[0]) / 255.0]


def _sample_vertex_color(bgr_frame, mask, pts_px, i, step, patch_size):
    """Option D: step both edge normals, pick the in-mask side, sample patch."""
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

    nx1, ny1 = -dy / length,  dx / length   # CCW normal
    nx2, ny2 =  dy / length, -dx / length   # CW  normal

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


def _contours_to_polys(contours, frame_w, frame_h, min_pts, epsilon, closed,
                        edge_map=None, spread=1.0, min_len=0,
                        bgr_frame=None, mask=None,
                        color_step=10, color_patch=5,
                        debug_prefix=None):
    """OpenCV contours → list of {'pts': [[x,y],...], 'intensities': [...],
                                    'colors': [[r,g,b],...], 'closed': bool}.

    edge_map   : optional uint8 [H, W] edge-probability map (post-blur,
                 pre-threshold) used to sample per-vertex intensity.  When
                 None, every vertex gets intensity 1.0 (Canny / outer SAM2).
    spread     : per-polyline intensity contrast modulator.
                    new = clamp(mean + (raw - mean) * spread, 0, 1)
                 1.0 = no change; >1 expands range; <1 compresses; 0 flattens.
    min_len    : minimum arc length in pixels.  Contours shorter than this
                 are discarded after approxPolyDP so speckle fragments don't
                 survive.
    bgr_frame  : optional [H, W, 3] uint8 BGR frame for per-vertex color
                 sampling (Option D).  When None, every vertex gets
                 [1.0, 1.0, 1.0] (white).
    mask       : optional bool [H, W] SAM2 mask used to disambiguate which
                 perpendicular side of an edge is inside the subject.  Both
                 sides outside → fallback to vertex sample.
    color_step : pixels to step from each vertex along the edge normal.
    color_patch: NxN patch averaged for the BGR sample.
    """
    out = []
    if edge_map is not None:
        H, W = edge_map.shape[:2]
    for cnt in contours:
        if len(cnt) < min_pts:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed) if epsilon > 0 else cnt
        pts    = approx.reshape(-1, 2)
        if len(pts) < min_pts:
            continue
        if min_len > 0 and cv2.arcLength(approx, closed) < min_len:
            continue

        # Per-vertex intensity sampling
        if edge_map is None:
            intensities = [1.0] * len(pts)
        else:
            intensities = []
            for pt in pts:
                px = max(0, min(W - 1, int(pt[0])))
                py = max(0, min(H - 1, int(pt[1])))
                intensities.append(edge_map[py, px] / 255.0)
            if spread != 1.0:
                m = sum(intensities) / len(intensities)
                intensities = [max(0.0, min(1.0, m + (v - m) * spread))
                               for v in intensities]

        # Per-vertex color sampling (Option D)
        if bgr_frame is None:
            colors = [[1.0, 1.0, 1.0] for _ in pts]
        else:
            cH, cW = bgr_frame.shape[:2]
            pts_px_list = [(max(0, min(cW - 1, int(p[0]))),
                            max(0, min(cH - 1, int(p[1])))) for p in pts]
            colors = [_sample_vertex_color(bgr_frame, mask, pts_px_list, i,
                                           color_step, color_patch)
                      for i in range(len(pts_px_list))]

        normed = [
            [ (float(pt[0]) / frame_w) * 2.0 - 1.0,
             -(float(pt[1]) / frame_h) * 2.0 + 1.0 ]
            for pt in pts
        ]
        out.append({'pts': normed,
                    'intensities': intensities,
                    'colors': colors,
                    'closed': bool(closed)})

    if debug_prefix:
        _render_polys_to_png(out, frame_w, frame_h,
                             f'{debug_prefix}_06_approx.png')
    return out


# =============================================================================
# Outer contours  (SAM2 mask boundary)
# =============================================================================

def mask_outer_contours(mask_bool, frame_w, frame_h, min_area_px, epsilon,
                        bgr_frame=None, color_step=10, color_patch=5):
    mask_u8  = mask_bool.astype(np.uint8) * 255
    cnts, _  = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
    polys    = _contours_to_polys(filtered, frame_w, frame_h, 2, epsilon, closed=True,
                                  bgr_frame=bgr_frame, mask=mask_bool,
                                  color_step=color_step, color_patch=color_patch)
    # Tag as outer so the persistence filter in encode.py never drops them
    for p in polys:
        p['outer'] = True
    return polys


# =============================================================================
# Interior edge extractors
# =============================================================================

def interior_canny(mask_bool, gray_blended, frame_w, frame_h,
                   low, high, blur_k, epsilon, min_pts,
                   bgr_frame=None, color_step=10, color_patch=5):
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []
    img = np.clip(gray_blended, 0, 255).astype(np.uint8)
    if blur_k > 1:
        k = blur_k | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    edges = cv2.Canny(img, low, high)
    edges = cv2.bitwise_and(edges, inner)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon, closed=False,
                              bgr_frame=bgr_frame, mask=mask_bool,
                              color_step=color_step, color_patch=color_patch)


def interior_edgemap(mask_bool, edge_map, frame_w, frame_h,
                     threshold, blur_k, epsilon, min_pts, min_len, spread,
                     preprocess='diffusion_edge',
                     threshold_high=None,
                     bridge_kernel=3,
                     bgr_frame=None, color_step=10, color_patch=5,
                     debug_prefix=None):
    """
    Interior edge extraction from a float [0,1] edge-probability map.
    Used by all three neural edge detectors.  Common skeleton: erode mask
    -> [SEAM: map -> binary] -> AND with mask -> findContours -> polylines.
    Only the seam varies per model -- branches are named after the model
    they serve, since each model has different output characteristics and
    the pipelines diverge accordingly.

    preprocess='diffusion_edge':
        Hard threshold only.  DiffusionEdge already outputs near-binary
        single-pixel ridges, so no extra thinning is needed.

    preprocess='edter':
        NMS on the soft float map (before any threshold).  EDTER outputs
        moderately wide soft ridges; NMS collapses them to 1-pixel-wide
        skeletons while preserving the float intensity at the ridge centre.
        A single threshold then acts as a noise-floor kill.

    preprocess='hed':
        NMS -> noise-floor threshold -> soft-map-guided gap bridging ->
        component filter -> Zhang-Suen.  HED outputs wide soft probability
        ridges.  NMS extracts thin ridge centerlines directly from the
        soft map (correct curve geometry, not blob medial axes), but
        produces fragmented output where gradient direction is unreliable
        at ridge peaks.  Naive morphological closing cannot bridge the
        resulting gaps without also merging unrelated nearby ridges.
        Instead: use the SOFT HED MAP ITSELF as a connectivity guide --
        extend each NMS pixel outward by up to `bridge_kernel` pixels,
        but ONLY into areas where the soft map exceeds `threshold`.
        Bridges form along soft-supported corridors (real edges); they
        don't form across unrelated structures (low soft map between
        them).  A component filter then drops fragments not anchored by
        any `threshold_high` pixel.  Final Zhang-Suen ensures 1-pixel
        width.

    `threshold_high` is only consulted for preprocess='hed'.
    `bridge_kernel` is only consulted for preprocess='hed'.
    """
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    # Keep a high-precision float copy of the raw map for NMS.  The u8 cast
    # below quantises to 256 levels, which creates plateaus that break NMS
    # tie-breaking; NMS must run on the original floats.
    edge_float_raw = np.clip(edge_map, 0.0, 1.0).astype(np.float32)
    edge_u8        = (edge_float_raw * 255).astype(np.uint8)
    if debug_prefix:
        cv2.imwrite(f'{debug_prefix}_01_edge_u8.png', edge_u8)

    # ── seam: probability map -> binary ──────────────────────────────────
    # blur_k accepted but unused for the neural-edge paths: visual
    # inspection showed blur at small kernels has negligible effect on the
    # probability map, and median blur actively degrades NMS by creating
    # plateaus.  (Canny in interior_canny still uses blur -- gradient
    # operators on raw grayscale genuinely benefit from pre-smoothing.)
    if preprocess == 'hed':
        # Step 1: NMS extracts thin ridge centerlines from the soft map.
        # This gives the correct curve geometry -- 1-pixel-wide ridges
        # following actual probability peaks.  Its only weakness is
        # fragmentation where gradient direction is unreliable (ridge
        # peaks where gradient magnitude is near zero).
        thinned_f = _nms_edge_map(edge_float_raw)
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_02_nms.png',
                        (thinned_f * 255).astype(np.uint8))

        th_low  = float(threshold)
        th_high = float(threshold_high) if threshold_high is not None \
                                       else min(1.0, th_low * 3.5)

        # Step 2: noise-floor threshold on the NMS output.  Drops weak
        # NMS survivors below the floor before any bridging happens.
        nms_binary = (thinned_f >= th_low).astype(np.uint8) * 255
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_03_nms_thresholded.png', nms_binary)

        # Step 3: soft-map-guided gap bridging.
        # Naive morphological closing fails for large NMS gaps: a k-sized
        # kernel can only bridge gaps up to (k-2) px, and growing the kernel
        # further also merges UNRELATED nearby ridges (recreating the
        # bubble problem).  Solution: use the original soft HED map as a
        # connectivity guide.  A real edge with NMS gaps has CONTINUOUS
        # high values in the soft map along the gap (HED detected the edge
        # there; NMS just didn't fire consistently).  Two unrelated nearby
        # ridges have a LOW-probability gap between them in the soft map.
        # So: extend each NMS pixel outward up to `bridge_kernel` pixels,
        # but ONLY into pixels where the soft map exceeds `th_low`.  Real
        # gaps get bridged through their soft-supported corridor; unrelated
        # ridges don't merge because no support corridor connects them.
        k = max(1, int(bridge_kernel))
        if k > 1:
            soft_support = (edge_float_raw >= th_low).astype(np.uint8) * 255
            # Distance transform: for each background pixel, how far is the
            # nearest NMS pixel?
            dist     = cv2.distanceTransform(255 - nms_binary,
                                             cv2.DIST_L2, 3)
            near_nms = (dist <= float(k)).astype(np.uint8) * 255
            # Bridge candidates: near an NMS pixel AND soft-map-supported
            extensions = cv2.bitwise_and(near_nms, soft_support)
            bridged    = cv2.bitwise_or(nms_binary, extensions)
            if debug_prefix:
                cv2.imwrite(f'{debug_prefix}_03a_soft_support.png', soft_support)
        else:
            bridged = nms_binary
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_03b_bridged.png', bridged)

        # Step 4: hysteresis-style component filter.  Keep only connected
        # components that contain at least one pixel above th_high.  This
        # rejects noise fragments that survived NMS but aren't part of any
        # strong edge structure.
        strong_mask = thinned_f >= th_high
        _, labels   = cv2.connectedComponents(bridged, connectivity=8)
        keep_labels = np.unique(labels[strong_mask])
        keep_labels = keep_labels[keep_labels > 0]    # drop background label
        filtered = np.isin(labels, keep_labels).astype(np.uint8) * 255
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_03c_anchors.png',
                        strong_mask.astype(np.uint8) * 255)
            cv2.imwrite(f'{debug_prefix}_03d_filtered.png', filtered)

        # Step 5: Zhang-Suen guarantees 1-pixel width.  Closing widened
        # bridges to ~k pixels in places; this collapses any residual
        # thickness.  No branching because the input is curve-shaped.
        try:
            binary = cv2.ximgproc.thinning(
                filtered, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            from skimage.morphology import skeletonize as _skeletonize
            binary = (_skeletonize(filtered > 0).astype(np.uint8)) * 255
        # _contours_to_polys samples per-vertex intensities from edge_u8 --
        # swap in the NMS'd float map so intensities reflect the ridge
        # values at each skeleton pixel.
        edge_u8 = (thinned_f * 255).astype(np.uint8)

    elif preprocess == 'edter':
        # NMS only, single threshold acts as noise-floor kill.
        thinned_f = _nms_edge_map(edge_float_raw)
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_03_nms.png',
                        (thinned_f * 255).astype(np.uint8))
        binary = (thinned_f > threshold).astype(np.uint8) * 255
        edge_u8 = (thinned_f * 255).astype(np.uint8)

    else:  # 'diffusion_edge'
        binary = (edge_u8 > int(threshold * 255)).astype(np.uint8) * 255
    if debug_prefix:
        cv2.imwrite(f'{debug_prefix}_04_preprocessed.png', binary)
    # ─────────────────────────────────────────────────────────────────────

    binary = cv2.bitwise_and(binary, inner)
    if debug_prefix:
        cv2.imwrite(f'{debug_prefix}_05_masked.png', binary)

    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon,
                              closed=False, edge_map=edge_u8, spread=spread,
                              min_len=min_len,
                              bgr_frame=bgr_frame, mask=mask_bool,
                              color_step=color_step, color_patch=color_patch,
                              debug_prefix=debug_prefix)


# =============================================================================
# JSON output helper
# =============================================================================

def _stem_path(base_output, suffix):
    """Insert suffix before .json:  base.json -> base_thin.json"""
    base, ext = os.path.splitext(base_output)
    return base + suffix + (ext or '.json')


def _save_json(path, meta, frames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {'meta': meta, 'frames': frames}
    with open(path, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))   # compact — no extra whitespace
    size_kb = os.path.getsize(path) / 1024
    print(f'[vectorize] Saved: {path}  ({size_kb:.0f} KB)', flush=True)


# =============================================================================
# Catmull-Rom spline fitting
# =============================================================================

def _catmull_rom_chain(pts, n_per_seg, closed):
    """
    Resample a polyline using Catmull-Rom splines.
    n_per_seg : interpolated points inserted between each pair of control vertices.
    Returns a new list of [x, y] points.
    """
    n = len(pts)
    if n < 2 or n_per_seg <= 1:
        return pts

    arr = np.array(pts, dtype=np.float64)

    def _get(i):
        if closed:
            return arr[i % n]
        # Reflect endpoints for natural boundary
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
        result.append(list(pts[-1]))   # final endpoint

    return result


def apply_spline_fitting(polys, n_per_seg):
    """Apply Catmull-Rom spline fitting to every polyline in the list.

    Intensities and colors are linearly interpolated at the same t parameters
    used for geometry; 'outer' tag is preserved.  When n_per_seg <= 1 this is
    a no-op and the original list is returned unchanged.
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
        # Linearly interpolate intensities (and colors) at the same t values
        # used by _catmull_rom_chain so the arrays stay the same length as
        # new_pts.
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
        out.append(new_poly)
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load masks
    # -------------------------------------------------------------------------
    if not os.path.exists(args.masks):
        print(f'[vectorize] Masks file not found: {args.masks}', flush=True)
        sys.exit(1)

    data      = np.load(args.masks)
    all_masks = data['masks']          # bool [N_frames, H, W]
    frame_w   = int(data['frame_w'])
    frame_h   = int(data['frame_h'])
    total_frames = all_masks.shape[0]
    min_area_px  = args.min_area * frame_w * frame_h

    # Convert epsilon fractions → absolute pixels using the image diagonal.
    # All --*-epsilon args are expressed as a fraction of the diagonal so that
    # the same value produces equivalent geometric detail at any resolution.
    _diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    args.smooth_epsilon          = args.smooth_epsilon          * _diag
    args.canny_epsilon           = args.canny_epsilon           * _diag
    args.hed_epsilon             = args.hed_epsilon             * _diag
    args.edter_epsilon           = args.edter_epsilon           * _diag
    args.diffusion_edge_epsilon  = args.diffusion_edge_epsilon  * _diag

    print(f'[vectorize] Masks: {total_frames} frames  ({frame_w}x{frame_h})'
          f'  diagonal={_diag:.0f}px  epsilon={args.hed_epsilon:.1f}px', flush=True)

    # -------------------------------------------------------------------------
    # Determine which methods to run
    # -------------------------------------------------------------------------
    methods = _ALL_METHODS if args.method == 'all' else [args.method]

    # -------------------------------------------------------------------------
    # Load model runners
    # -------------------------------------------------------------------------

    def _try_load(method_name, model_path, runner_cls, **kwargs):
        """Instantiate a model runner; return None and drop the method on failure."""
        if method_name not in methods:
            return None
        try:
            return runner_cls(model_path, args.device, **kwargs)
        except Exception as e:
            print(f'[vectorize] {method_name} load failed: {e}'
                  f' — skipping {method_name}', flush=True)
            methods.remove(method_name)
            return None

    # HED auto-downloads weights from content.sniklaus.com on first run.
    edter_runner          = _try_load('edter',          args.edter_model,          EDTERRunner)
    diffusion_edge_runner = _try_load('diffusion_edge', args.diffusion_edge_model, DiffusionEdgeRunner,
                                      first_stage_path=args.diffusion_edge_first_stage_model,
                                      config_path=args.diffusion_edge_config,
                                      sampling_steps=args.diffusion_edge_steps)
    hed_runner            = _try_load('hed',     args.hed_model,     HEDRunner)

    if not methods:
        print('[vectorize] No methods available — nothing to do.', flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Open video for pixel data
    # -------------------------------------------------------------------------
    if not os.path.exists(args.video):
        print(f'[vectorize] Video not found: {args.video}', flush=True)
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f'[vectorize] Cannot open video: {args.video}', flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Per-method frame storage
    # -------------------------------------------------------------------------
    # Each entry is a list of polyline dicts: {'pts': [[x,y],...], 'closed': bool}
    frames_by_method = {m: [None] * total_frames for m in methods}

    # Frame-blended signal maps (temporal averaging)
    prev_gray           = None
    prev_hed            = None
    prev_edter          = None
    prev_diffusion_edge = None
    alpha               = args.frame_avg_alpha

    print(f'[vectorize] Running  method={args.method}  '
          f'frame_avg_alpha={alpha}  spline_samples={args.spline_samples}',
          flush=True)

    for frame_idx in range(total_frames):
        ret, bgr = cap.read()
        if not ret:
            print(f'[vectorize] Warning: ran out of video frames at {frame_idx}', flush=True)
            break

        mask = all_masks[frame_idx]   # bool [H, W]

        # Forward BGR frame to vectorizers for per-vertex color sampling.
        # --no-color short-circuits to white at the _contours_to_polys layer.
        bgr_frame_arg = None if args.no_color else bgr

        # --- Greyscale, temporally blended (used by Canny) ---
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None and alpha < 1.0:
            gray = alpha * gray + (1.0 - alpha) * prev_gray
        prev_gray = gray

        def _run_and_blend(runner, bgr_frame, prev_map, mask=None):
            """Run a runner and apply temporal blending. Returns (new_map, new_prev).

            ``mask`` is forwarded to the runner so tile-based models (EDTER,
            DiffusionEdge) can skip background tiles.  HED ignores it.
            """
            if runner is None:
                return None, prev_map
            m = runner.infer(bgr_frame, mask=mask)
            if prev_map is not None and alpha < 1.0:
                m = alpha * m + (1.0 - alpha) * prev_map
            return m, m

        hed_map,            prev_hed            = _run_and_blend(hed_runner,            bgr, prev_hed,            mask)
        edter_map,          prev_edter          = _run_and_blend(edter_runner,          bgr, prev_edter,          mask)
        diffusion_edge_map, prev_diffusion_edge = _run_and_blend(diffusion_edge_runner, bgr, prev_diffusion_edge, mask)

        # --- Save raw model output for frame 0 (visual inspection) ---
        if frame_idx == 0:
            _edgemap_saves = [
                ('edter',          edter_map),
                ('diffusion_edge', diffusion_edge_map),
                ('hed',            hed_map),
            ]
            for _em_name, _em in _edgemap_saves:
                if _em_name not in methods or _em is None:
                    continue
                if args.method == 'all':
                    _em_json = _stem_path(args.output, f'_{_em_name}')
                else:
                    _em_json = args.output
                _em_path = os.path.splitext(_em_json)[0] + '_edgemap.png'
                os.makedirs(os.path.dirname(os.path.abspath(_em_path)), exist_ok=True)
                cv2.imwrite(_em_path, (np.clip(_em, 0.0, 1.0) * 255).astype(np.uint8))
                print(f'[vectorize] Raw {_em_name} edge map (frame 0) → {_em_path}',
                      flush=True)

        # --- Outer contour (shared across all methods for this frame) ---
        outer = mask_outer_contours(mask, frame_w, frame_h, min_area_px,
                                    args.smooth_epsilon,
                                    bgr_frame=bgr_frame_arg,
                                    color_step=args.color_step,
                                    color_patch=args.color_patch)

        # --- Interior edges per method ---
        for m in methods:
            # On frame 0, build per-method debug prefix that matches the
            # existing *_edgemap.png stem.  None on other frames disables
            # debug saving everywhere downstream.
            dbg = None
            if frame_idx == 0 and m in ('hed', 'edter', 'diffusion_edge'):
                _m_json = (_stem_path(args.output, f'_{m}')
                           if args.method == 'all' else args.output)
                dbg = os.path.splitext(_m_json)[0]

            interior = []
            if m == 'canny':
                interior = interior_canny(
                    mask, gray, frame_w, frame_h,
                    args.canny_low, args.canny_high, args.canny_blur,
                    args.canny_epsilon, args.canny_min_pts,
                    bgr_frame=bgr_frame_arg,
                    color_step=args.color_step,
                    color_patch=args.color_patch)
            elif m == 'hed' and hed_map is not None:
                interior = interior_edgemap(
                    mask, hed_map, frame_w, frame_h,
                    args.hed_threshold, args.hed_blur,
                    args.hed_epsilon, args.hed_min_pts, args.hed_min_len,
                    args.hed_spread,
                    preprocess='hed',
                    threshold_high=args.hed_threshold_high,
                    bridge_kernel=args.hed_bridge_kernel,
                    bgr_frame=bgr_frame_arg,
                    color_step=args.color_step,
                    color_patch=args.color_patch,
                    debug_prefix=dbg)
            elif m == 'edter' and edter_map is not None:
                interior = interior_edgemap(
                    mask, edter_map, frame_w, frame_h,
                    args.edter_threshold, args.edter_blur,
                    args.edter_epsilon, args.edter_min_pts, args.edter_min_len,
                    args.edter_spread,
                    preprocess='edter',
                    bgr_frame=bgr_frame_arg,
                    color_step=args.color_step,
                    color_patch=args.color_patch,
                    debug_prefix=dbg)
            elif m == 'diffusion_edge' and diffusion_edge_map is not None:
                interior = interior_edgemap(
                    mask, diffusion_edge_map, frame_w, frame_h,
                    args.diffusion_edge_threshold, args.diffusion_edge_blur,
                    args.diffusion_edge_epsilon, args.diffusion_edge_min_pts,
                    args.diffusion_edge_min_len,
                    args.diffusion_edge_spread,
                    preprocess='diffusion_edge',
                    bgr_frame=bgr_frame_arg,
                    color_step=args.color_step,
                    color_patch=args.color_patch,
                    debug_prefix=dbg)

            frame_polys = outer + interior
            frame_polys = apply_spline_fitting(frame_polys, args.spline_samples)
            frames_by_method[m][frame_idx] = frame_polys

            if dbg:
                _render_polys_to_png(frame_polys, frame_w, frame_h,
                                     f'{dbg}_07_splined.png')
                print(f'[vectorize] {m} debug images (frame 0) → '
                      f'{os.path.basename(dbg)}_01..07.png', flush=True)

        if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
            sample = frames_by_method[methods[0]][frame_idx] or []
            pts    = sum(len(p['pts']) for p in sample)
            print(f'[vectorize] {frame_idx + 1}/{total_frames}  '
                  f'polylines={len(sample)}  pts={pts}', flush=True)

    cap.release()

    # -------------------------------------------------------------------------
    # Save one JSON per method
    # -------------------------------------------------------------------------
    video_basename = os.path.basename(args.video)
    masks_basename = os.path.basename(args.masks)

    for m in methods:
        frames_raw = frames_by_method[m]
        # Drop None entries (video ran short)
        frames_clean = [f for f in frames_raw if f is not None]

        meta = {
            'video':        video_basename,
            'masks':        masks_basename,
            'method':       m,
            'frame_w':      frame_w,
            'frame_h':      frame_h,
            'total_frames': len(frames_clean),
        }

        if args.method == 'all':
            out_path = _stem_path(args.output, f'_{m}')
        else:
            out_path = args.output

        _save_json(out_path, meta, frames_clean)


if __name__ == '__main__':
    main()
