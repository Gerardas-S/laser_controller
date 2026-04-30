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
    resources/polylines/{stem}_sam2-{model}_depth.json
    resources/polylines/{stem}_sam2-{model}_thin.json

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
      {"pts": [[x, y], ...], "closed": true},   <- outer SAM2 contour
      {"pts": [[x, y], ...], "closed": false}   <- interior edge path
    ],
    ...
  ]
}

Coordinates are normalized to [-1, 1] with Y-up (laser convention).
Color (always white) is added by encode.py.
"""

import argparse
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2


# =============================================================================
# Args
# =============================================================================

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
                   choices=['canny', 'hed', 'depth', 'thin', 'depth_iso', 'flow',
                            'hatch', 'lum_iso', 'all'],
                   help='Interior edge method(s)')
    # Model paths
    p.add_argument('--hed-model',       default='models/hed/model.onnx')
    p.add_argument('--depth-model',     default='models/depth/model.onnx')
    p.add_argument('--device',          default='cpu', choices=['cpu', 'cuda'])
    # Outer contour
    p.add_argument('--min-area',        type=float, default=0.001,
                   help='Minimum mask area as fraction of frame area')
    p.add_argument('--smooth-epsilon',  type=float, default=2.0,
                   help='approxPolyDP epsilon for outer contours (pixels)')
    # Frame averaging
    p.add_argument('--frame-avg-alpha', type=float, default=0.7,
                   help='Signal temporal blend: 1.0=off, lower=more averaging')
    # Point budget
    p.add_argument('--point-budget',    type=int,   default=600,
                   help='Max laser points per frame before spline expansion '
                        '(0=unlimited). Final count ≈ budget × spline-samples. '
                        'At 30kpps/30fps the hard ceiling is ~1000 points.')
    # Canny
    p.add_argument('--canny-low',       type=int,   default=40)
    p.add_argument('--canny-high',      type=int,   default=120)
    p.add_argument('--canny-blur',      type=int,   default=3)
    p.add_argument('--canny-epsilon',   type=float, default=1.5)
    p.add_argument('--canny-min-pts',   type=int,   default=4)
    # HED
    p.add_argument('--hed-threshold',   type=float, default=0.35)
    p.add_argument('--hed-blur',        type=int,   default=3,
                   help='Median blur radius applied to HED map before thresholding')
    p.add_argument('--hed-epsilon',     type=float, default=1.5,
                   help='approxPolyDP epsilon for HED skeleton strokes (pixels). '
                        'Lower = more detail preserved.')
    p.add_argument('--hed-min-pts',     type=int,   default=3)
    p.add_argument('--hed-min-len',     type=int,   default=4,
                   help='Minimum skeleton stroke length in pixels. '
                        'Lower = fine details (lashes, wrinkles) preserved.')
    # Depth
    p.add_argument('--depth-threshold', type=float, default=0.08)
    p.add_argument('--depth-epsilon',   type=float, default=2.0)
    p.add_argument('--depth-min-pts',   type=int,   default=4)
    # Thinning
    p.add_argument('--thin-thresh',     type=float, default=0.12)
    p.add_argument('--thin-blur',       type=int,   default=3)
    p.add_argument('--thin-epsilon',    type=float, default=2.0)
    p.add_argument('--thin-min-pts',    type=int,   default=5)
    # Depth iso-contours
    p.add_argument('--iso-levels',      type=int,   default=5,
                   help='Number of depth iso-contour bands')
    p.add_argument('--iso-epsilon',     type=float, default=2.0)
    p.add_argument('--iso-min-pts',     type=int,   default=4)
    # Luminance iso-contours  (topographic brightness map, no model required)
    p.add_argument('--lum-iso-levels',  type=int,   default=6,
                   help='Number of luminance iso-contour bands (4-10 typical). '
                        'Each band adds one closed contour ring per luminance zone.')
    p.add_argument('--lum-iso-epsilon', type=float, default=2.0,
                   help='DP simplification tolerance for lum_iso contours (px).')
    p.add_argument('--lum-iso-min-pts', type=int,   default=4)
    p.add_argument('--lum-iso-blur',    type=int,   default=5,
                   help='Gaussian blur half-size before iso-contouring.  '
                        'Higher = smoother, broader contours.  0 = no blur.')
    # Optical flow / motion
    p.add_argument('--flow-threshold',  type=float, default=1.5,
                   help='Optical flow magnitude threshold in pixels/frame')
    p.add_argument('--flow-blur',       type=int,   default=21,
                   help='Gaussian blur radius on flow magnitude before thresholding')
    p.add_argument('--flow-epsilon',    type=float, default=3.0)
    p.add_argument('--flow-min-pts',    type=int,   default=4)
    # Spline fitting (applied to all methods)
    p.add_argument('--spline-samples',  type=int,   default=4,
                   help='Catmull-Rom samples per segment between DP vertices '
                        '(1=off, 4=smooth, 8=very smooth)')
    # Toon preprocessing (bilateral filter + optional colour quantisation)
    p.add_argument('--toon',            action='store_true',
                   help='Apply toon/anime bilateral preprocessing before HED, Canny and '
                        'Thin.  Removes texture noise so edge detectors fire on structural '
                        'edges only.  Two bilateral filter passes at the given sigma.')
    p.add_argument('--toon-d',          type=int,   default=9,
                   help='Bilateral filter neighbourhood diameter (px).  9-15 typical.')
    p.add_argument('--toon-sigma-color',type=float, default=75.0,
                   help='Bilateral colour sigma.  75-150 typical.')
    p.add_argument('--toon-sigma-space',type=float, default=75.0,
                   help='Bilateral spatial sigma.  75-150 typical.')
    p.add_argument('--toon-colors',     type=int,   default=0,
                   help='K-means colour palette size for posterisation (0=disabled). '
                        '8-16 recommended.  Adds ~0.3-0.8 s per frame on CPU.')
    # Artistic hatching (streamline-based engraving)
    p.add_argument('--hatch-blur',      type=float, default=12.0,
                   help='Gaussian sigma for luminance gradient smoothing.  '
                        'Large (8-20) = sweeping strokes following overall form.  '
                        'Small (2-6) = strokes following local surface detail.')
    p.add_argument('--hatch-tones',     type=int,   default=4,
                   help='Number of luminance tone bands.  Each band adds a layer of '
                        'strokes at its own spacing and luminance threshold.  '
                        'More = richer shadows, more points per frame.')
    p.add_argument('--hatch-spacing',   type=int,   default=12,
                   help='Seed grid spacing in px for the darkest tone band.  '
                        'Each successive band doubles the spacing (sparser coverage).')
    p.add_argument('--hatch-max-len',   type=int,   default=80,
                   help='Max Euler integration steps per trace direction.  '
                        'Controls maximum stroke length in pixels.')
    p.add_argument('--hatch-min-sep',   type=int,   default=8,
                   help='Minimum pixel radius kept clear around each placed seed.  '
                        'Prevents strokes from touching each other.')
    p.add_argument('--hatch-epsilon',   type=float, default=2.0,
                   help='Douglas-Peucker simplification epsilon for stroke polylines (px).')
    p.add_argument('--hatch-min-pts',   type=int,   default=4,
                   help='Minimum surviving points per stroke polyline.')
    return p.parse_args()


# =============================================================================
# ONNX inference runners
# =============================================================================

class HEDRunner:
    MEAN_B, MEAN_G, MEAN_R = 104.00698793, 116.66876762, 122.67891434
    INPUT_H, INPUT_W = 480, 480

    def __init__(self, model_path, device='cpu'):
        import onnxruntime as ort
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.sess       = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.out_name   = self.sess.get_outputs()[-1].name
        print(f'[hed] Loaded {model_path}', flush=True)

    def infer(self, bgr_frame):
        h0, w0 = bgr_frame.shape[:2]
        rsz = cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H)).astype(np.float32)
        b, g, r = cv2.split(rsz)
        b -= self.MEAN_B; g -= self.MEAN_G; r -= self.MEAN_R
        blob = np.stack([b, g, r], axis=0)[np.newaxis]
        raw  = self.sess.run([self.out_name], {self.input_name: blob})[0]
        edge = raw.squeeze()
        if edge.ndim == 3:
            edge = edge.squeeze(0)
        return cv2.resize(edge, (w0, h0), interpolation=cv2.INTER_LINEAR)


class DepthRunner:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    INPUT_H, INPUT_W = 518, 518

    def __init__(self, model_path, device='cpu'):
        import onnxruntime as ort
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.sess       = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.out_name   = self.sess.get_outputs()[0].name
        print(f'[depth] Loaded {model_path}', flush=True)

    def infer(self, bgr_frame):
        h0, w0 = bgr_frame.shape[:2]
        rsz = cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H))
        rgb = cv2.cvtColor(rsz, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD
        blob = rgb.transpose(2, 0, 1)[np.newaxis]
        raw  = self.sess.run([self.out_name], {self.input_name: blob})[0]
        d    = raw.squeeze()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lo, hi = d.min(), d.max()
        return (d - lo) / (hi - lo + 1e-6)


# =============================================================================
# Shared helpers
# =============================================================================

_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def _erode_mask(mask_u8):
    inner = cv2.erode(mask_u8, _ERODE_KERNEL, iterations=1)
    return inner if cv2.countNonZero(inner) > 0 else None


def _contours_to_polys(contours, frame_w, frame_h, min_pts, epsilon, closed):
    """OpenCV contours → list of {'pts': [[x,y],...], 'closed': bool}."""
    out = []
    for cnt in contours:
        if len(cnt) < min_pts:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed) if epsilon > 0 else cnt
        pts    = approx.reshape(-1, 2)
        if len(pts) < min_pts:
            continue
        normed = [
            [ (float(pt[0]) / frame_w) * 2.0 - 1.0,
             -(float(pt[1]) / frame_h) * 2.0 + 1.0 ]
            for pt in pts
        ]
        out.append({'pts': normed, 'closed': bool(closed)})
    return out


# =============================================================================
# Outer contours  (SAM2 mask boundary)
# =============================================================================

def mask_outer_contours(mask_bool, frame_w, frame_h, min_area_px, epsilon):
    mask_u8  = mask_bool.astype(np.uint8) * 255
    cnts, _  = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
    polys    = _contours_to_polys(filtered, frame_w, frame_h, 2, epsilon, closed=True)
    # Tag as outer so the persistence filter in encode.py never drops them
    for p in polys:
        p['outer'] = True
    return polys


# =============================================================================
# Interior edge extractors
# =============================================================================

def interior_canny(mask_bool, gray_blended, frame_w, frame_h,
                   low, high, blur_k, epsilon, min_pts):
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
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon, closed=False)


def interior_hed(mask_bool, hed_blended, frame_w, frame_h,
                 threshold, blur_k, epsilon, min_pts, min_len):
    """
    HED interior edges via findContours on the thresholded HED map.

    We do NOT skeletonise here.  Skeletonisation of HED blobs produces ring
    artefacts ("bubbles") because the skeleton of a rounded/thick blob is
    itself a ring.  findContours traces the outer boundary of each blob as a
    tight polygon; for thin edges that polygon is visually indistinguishable
    from a single line, and it preserves fine detail (lashes, wrinkles, etc.)
    that skeletonisation prunes away.
    """
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    hed_u8 = (np.clip(hed_blended, 0.0, 1.0) * 255).astype(np.uint8)

    # Median blur reduces HED salt-and-pepper speckle without smearing edges
    if blur_k > 1:
        k = blur_k | 1
        hed_u8 = cv2.medianBlur(hed_u8, k)

    binary  = (hed_u8 > int(threshold * 255)).astype(np.uint8) * 255
    binary  = cv2.bitwise_and(binary, inner)

    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon,
                              closed=False)


def interior_depth(mask_bool, depth_blended, frame_w, frame_h,
                   threshold, epsilon, min_pts):
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []
    inner_f = inner.astype(np.float32) / 255.0
    masked  = depth_blended * inner_f
    gx = cv2.Sobel(masked, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(masked, cv2.CV_32F, 0, 1, ksize=3)
    mag    = cv2.magnitude(gx, gy)
    binary = (mag > threshold).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, inner)
    dil_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, dil_k, iterations=1)
    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon, closed=False)


def skeleton_to_polylines(skel_u8, frame_w, frame_h, epsilon, min_pts, min_len=0):
    """Convert a 1-pixel-wide skeleton image to ordered open polylines."""
    ys, xs = np.where(skel_u8 > 0)
    if len(xs) == 0:
        return []

    pixel_set = set(zip(xs.tolist(), ys.tolist()))
    _OFF = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _nb(x, y, src):
        return [(x+dx, y+dy) for dx, dy in _OFF if (x+dx, y+dy) in src]

    junctions = {p for p in pixel_set if len(_nb(p[0], p[1], pixel_set)) >= 3}
    working   = pixel_set - junctions

    visited    = set()
    components = []
    for seed in working:
        if seed in visited:
            continue
        comp  = []
        stack = [seed]
        visited.add(seed)
        while stack:
            p = stack.pop()
            comp.append(p)
            for nb in _nb(p[0], p[1], working):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)

    polylines = []
    for comp in components:
        if len(comp) < min_pts:
            continue
        comp_set  = set(comp)
        endpoints = [p for p in comp if len(_nb(p[0], p[1], comp_set)) <= 1]
        start     = endpoints[0] if endpoints else comp[0]

        # A component with no endpoints is a closed loop (every pixel has
        # exactly 2 neighbours).  Trace it fully and mark closed=True so it
        # is rendered as an actual ring, not a broken arc.
        is_loop = len(endpoints) == 0

        path = [start]
        seen = {start}
        curr = start
        while True:
            nbs = [nb for nb in _nb(curr[0], curr[1], comp_set) if nb not in seen]
            if not nbs:
                break
            curr = nbs[0]
            path.append(curr)
            seen.add(curr)

        if len(path) < min_pts:
            continue

        # Prune short branches by pixel arc-length
        if min_len > 0:
            arr = np.array(path, dtype=np.float32)
            arc_len = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            if arc_len < min_len:
                continue

        cnt = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        if epsilon > 0:
            approx = cv2.approxPolyDP(cnt, epsilon, is_loop)
            pts    = approx.reshape(-1, 2)
        else:
            pts = cnt.reshape(-1, 2)

        if len(pts) < min_pts:
            continue

        normed = [
            [ (float(pt[0]) / frame_w) * 2.0 - 1.0,
             -(float(pt[1]) / frame_h) * 2.0 + 1.0 ]
            for pt in pts
        ]
        polylines.append({'pts': normed, 'closed': is_loop})

    return polylines


def interior_thin(mask_bool, gray_blended, frame_w, frame_h,
                  thresh, blur_k, epsilon, min_pts):
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    img = np.clip(gray_blended, 0, 255).astype(np.uint8)
    if blur_k > 1:
        k = blur_k | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = cv2.bitwise_and(img, inner)

    img_f = img.astype(np.float32)
    gx    = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag   = cv2.magnitude(gx, gy)
    peak  = mag.max()
    if peak < 1e-6:
        return []
    binary = (mag / peak > thresh).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, inner)

    try:
        from skimage.morphology import skeletonize as ski_skeletonize
        skel = ski_skeletonize(binary > 0).astype(np.uint8) * 255
    except ImportError:
        try:
            skel = cv2.ximgproc.thinning(binary,
                       thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            skel = binary

    return skeleton_to_polylines(skel, frame_w, frame_h, epsilon, min_pts)


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
# Polyline ordering  (nearest-neighbour travel salesman — greedy)
# =============================================================================

def order_polylines(polys):
    """
    Re-order polylines so the blank-jump distance between consecutive strokes
    is minimised.  Each polyline may also be reversed if that reduces travel.
    Returns a new list in draw order.

    Implementation: vectorised nearest-neighbour greedy TSP.
    All distance computations are batched with numpy — O(N²) operations but
    ~20× faster than a pure Python loop for the N>100 case typical of hatch.
    All polyline metadata (outer, closed, …) is preserved through reversal.
    """
    if len(polys) <= 1:
        return polys

    N      = len(polys)
    starts = np.array([p['pts'][0]  for p in polys], dtype=np.float32)  # (N, 2)
    ends   = np.array([p['pts'][-1] for p in polys], dtype=np.float32)  # (N, 2)

    done    = np.zeros(N, dtype=np.bool_)
    ordered = []
    cur     = np.zeros(2, dtype=np.float32)   # laser pen starts at origin

    for _ in range(N):
        d_fwd          = np.sum((starts - cur) ** 2, axis=1)
        d_rev          = np.sum((ends   - cur) ** 2, axis=1)
        d_best         = np.where(d_fwd <= d_rev, d_fwd, d_rev)
        d_best[done]   = np.inf

        bi  = int(np.argmin(d_best))
        rev = bool(d_rev[bi] < d_fwd[bi])
        done[bi] = True

        poly = polys[bi]
        if rev and not poly.get('closed', False):
            # Copy all metadata fields; only reverse the point list.
            poly = {**poly, 'pts': list(reversed(poly['pts']))}

        ordered.append(poly)
        # After drawing, pen is at the polyline's last point.
        cur = starts[bi].copy() if rev else ends[bi].copy()

    return ordered


# =============================================================================
# Point budget enforcement
# =============================================================================

def apply_point_budget(polys, budget):
    """
    Reduce total point count to <= budget without destroying polyline shapes.

    Strategy:
      Phase 1 — gentle DP re-simplification at a small fixed epsilon sequence
                 (capped so we never flatten curves into straight lines).
                 Coords are normalized [-1,1]; 0.001 ≈ 0.5 px on 1080p.
      Phase 2 — if still over budget, drop the shortest polylines first
                 (they carry the least visual information).

    The outer contour (first polyline, closed=True) is always kept.
    """
    if budget <= 0 or not polys:
        return polys

    total = sum(len(p['pts']) for p in polys)
    if total <= budget:
        return polys

    import copy
    result = copy.deepcopy(polys)

    # Phase 1: fixed epsilon steps, hard-capped at 0.012 normalized units
    # (≈ 6 px on a 1080-wide frame — aggressive but not shape-destroying)
    for eps in [0.001, 0.002, 0.004, 0.007, 0.012]:
        if sum(len(p['pts']) for p in result) <= budget:
            break
        simplified = []
        for p in result:
            pts_arr = np.array(p['pts'], dtype=np.float32).reshape(-1, 1, 2)
            approx  = cv2.approxPolyDP(pts_arr, eps, p.get('closed', False))
            pts_out = approx.reshape(-1, 2).tolist()
            if len(pts_out) >= 2:
                simplified.append({'pts': pts_out, 'closed': p.get('closed', False)})
        if simplified:
            result = simplified

    # Phase 2: drop shortest interior polylines to meet budget
    # Separate outer contour (index 0, closed) from interior strokes
    if sum(len(p['pts']) for p in result) > budget:
        outer   = [p for p in result if p.get('closed', False)]
        interior = sorted(
            [p for p in result if not p.get('closed', False)],
            key=lambda p: len(p['pts']), reverse=True)   # longest first

        kept    = outer[:]
        running = sum(len(p['pts']) for p in kept)
        for p in interior:
            n = len(p['pts'])
            if running + n <= budget:
                kept.append(p)
                running += n
        result = kept

    return result


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
    """Apply Catmull-Rom spline fitting to every polyline in the list."""
    if n_per_seg <= 1:
        return polys
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


# =============================================================================
# Depth iso-contours  (topographic bands)
# =============================================================================

def interior_depth_iso(mask_bool, depth_blended, frame_w, frame_h,
                       n_levels, epsilon, min_pts):
    """
    Quantise the masked depth map into n_levels bands and extract the contour
    line between each adjacent pair of bands — a topographic / iso-depth map.
    Produces closed polylines, one ring per depth boundary per contiguous region.
    """
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    inner_bool  = inner > 0
    valid_depth = depth_blended[inner_bool]
    if len(valid_depth) == 0:
        return []

    d_min = float(valid_depth.min())
    d_max = float(valid_depth.max())
    if d_max - d_min < 1e-4:
        return []

    polys = []
    for i in range(1, n_levels):
        thresh = d_min + (d_max - d_min) * i / n_levels
        binary = ((depth_blended > thresh) & inner_bool).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polys  += _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon,
                                     closed=True)
    return polys


# =============================================================================
# Optical flow / motion
# =============================================================================

def interior_flow(mask_bool, gray_curr_raw, gray_prev_raw, frame_w, frame_h,
                  threshold, blur_k, epsilon, min_pts):
    """
    Compute dense Farneback optical flow between the previous and current raw
    (unblended) greyscale frames.  Threshold the magnitude map to find regions
    of significant motion, close small holes, then extract contours of those
    moving regions as closed polylines.

    Frame 0 always returns [] because there is no previous frame.
    """
    if gray_prev_raw is None:
        return []

    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    curr = np.clip(gray_curr_raw, 0, 255).astype(np.uint8)
    prev = np.clip(gray_prev_raw, 0, 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag = cv2.magnitude(flow[..., 0], flow[..., 1])

    if blur_k > 1:
        k   = blur_k | 1
        mag = cv2.GaussianBlur(mag, (k, k), 0)

    binary = (mag > threshold).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, inner)

    if cv2.countNonZero(binary) == 0:
        return []

    # Close gaps so that motion blobs are coherent shapes
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon, closed=True)


# =============================================================================
# Luminance iso-contours  (topographic map of brightness)
# =============================================================================

def interior_lum_iso(mask_bool, gray_blended, frame_w, frame_h,
                     n_levels, epsilon, min_pts, blur_k=5):
    """
    Quantise the masked luminance map into n_levels bands and extract the
    contour ring between each adjacent pair of bands.

    Produces closed polylines arranged like topographic contours — the visual
    effect is a sculptural elevation map of the subject's lighting.  Unlike
    depth_iso (which uses the depth model), lum_iso requires no ONNX model:
    it reads directly from the temporally-blended greyscale frame.

    A mild Gaussian pre-blur suppresses specular highlights and skin texture
    so the contours follow large-scale luminance forms (face shape, body
    curvature, drapery) rather than fine texture.

    Parameters
    ----------
    n_levels : number of iso-bands (4-10 typical).
    epsilon  : DP simplification tolerance (px).
    min_pts  : minimum points per contour.
    blur_k   : Gaussian blur kernel half-size applied before thresholding.
               Larger = smoother contours, fewer tiny artefacts.
    """
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    inner_bool = inner > 0

    lum = np.clip(gray_blended, 0.0, 255.0).astype(np.float32)

    # Gentle blur to remove texture noise before iso-contouring
    if blur_k > 0:
        k   = (blur_k * 2 + 1)
        lum = cv2.GaussianBlur(lum, (k, k), 0)

    valid_lum = lum[inner_bool]
    if len(valid_lum) == 0:
        return []

    l_min = float(valid_lum.min())
    l_max = float(valid_lum.max())
    if l_max - l_min < 1e-3:
        return []

    polys = []
    for i in range(1, n_levels):
        thresh = l_min + (l_max - l_min) * i / n_levels
        binary = ((lum > thresh) & inner_bool).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polys  += _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon,
                                     closed=True)
    return polys


# =============================================================================
# Toon preprocessing  (bilateral filter + optional k-means posterisation)
# =============================================================================

def toon_preprocess(frame_bgr, d=9, sigma_color=75.0, sigma_space=75.0, n_colors=0):
    """
    Two-pass bilateral filter + optional k-means colour quantisation.

    Why this improves edge detection
    ---------------------------------
    - Two passes of bilateral filtering remove texture noise (skin pores, fabric
      weave, foliage grain) while keeping sharp structural edges intact.
    - K-means posterisation (optional) eliminates residual fine colour gradients
      so that HED and Canny fire only at large-scale luminance boundaries.

    The result is fed to HED, Canny and Thin so they produce structural outlines
    rather than the texture halo that otherwise fills the interior of subjects.

    Parameters
    ----------
    d             : Bilateral filter neighbourhood diameter (px).  9-15 typical.
    sigma_color   : Bilateral colour sigma.  75-150.
    sigma_space   : Bilateral spatial sigma.  75-150.
    n_colors      : K-means cluster count (0 = skip quantisation).  8-16 good.
                    Adds ~0.3-0.8 s per frame on CPU.
    """
    out = cv2.bilateralFilter(frame_bgr, d, sigma_color, sigma_space)
    out = cv2.bilateralFilter(out,       d, sigma_color, sigma_space)

    if n_colors > 1:
        pixels = out.reshape(-1, 3).astype(np.float32)

        # Sample for k-means centroid estimation — full-image k-means is slow.
        n_sample = min(len(pixels), 15_000)
        rng      = np.random.default_rng(42)
        idx      = rng.integers(0, len(pixels), n_sample)
        sample   = pixels[idx]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(sample, n_colors, None, criteria, 3,
                                   cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)

        # Assign every pixel to its nearest centre.  Chunked to bound memory.
        CHUNK  = 100_000
        labels = np.empty(len(pixels), dtype=np.int32)
        c_sq   = np.sum(centers ** 2, axis=1)              # (k,)
        for i in range(0, len(pixels), CHUNK):
            chunk          = pixels[i:i + CHUNK]            # (N, 3)
            p_sq           = np.sum(chunk ** 2, axis=1, keepdims=True)   # (N, 1)
            cross          = chunk @ centers.T              # (N, k)
            dist2          = p_sq - 2.0 * cross + c_sq     # (N, k)
            labels[i:i + CHUNK] = np.argmin(dist2, axis=1)

        out = centers[labels].reshape(frame_bgr.shape).astype(np.uint8)

    return out


# =============================================================================
# Artistic hatching  (streamline-based engraving)
# =============================================================================

def interior_hatch(mask_bool, frame_bgr, frame_w, frame_h,
                   blur_sigma, n_tones, base_spacing, max_stroke_steps,
                   min_sep, epsilon, min_pts, point_budget=0):
    """
    Engraving-style stroke synthesis via Euler streamline integration.

    Strokes follow the tangent to the luminance gradient field — the direction
    perpendicular to the intensity gradient — so they naturally curve around the
    subject's form (cheekbones, muscle bellies, drapery folds).  Darker regions
    receive denser strokes; bright areas are left unhatched, mirroring the tonal
    logic of copperplate engraving.

    Algorithm
    ---------
    1.  Smooth the masked luminance with two passes of a heavy Gaussian
        (sigma = blur_sigma) to capture large-scale form rather than texture.
    2.  Compute Sobel gradients and derive the 90°-rotated unit tangent:
            T = (-∂L/∂y,  ∂L/∂x) / |∇L|
    3.  Divide the in-mask luminance range into n_tones percentile bands.
        Band 0 = darkest (densest seed grid).  Each successive band doubles
        the grid spacing and covers a progressively lighter tone range.
    4.  For each band, iterate over grid seeds:
          a.  Reject if outside mask, too bright, or inside sep zone.
          b.  Forward-trace max_stroke_steps Euler steps along +T.
          c.  Backward-trace the same along -T.
          d.  Concatenate → one curved stroke polyline.
          e.  Paint a separation bubble along the stroke in sep[].
    5.  DP simplification + normalise to laser [-1, 1] Y-up coordinates.

    Parameters
    ----------
    blur_sigma       : Gaussian sigma for luminance smoothing.  Large (8-20)
                       = broad sweeping strokes following overall form.
                       Small (2-6) = strokes following local surface detail.
    n_tones          : Number of luminance tone bands.
    base_spacing     : Seed grid spacing (px) for band 0 (darkest, densest).
                       Each successive band doubles the spacing.
    max_stroke_steps : Euler integration steps per direction (≈ 1 px/step).
    min_sep          : Sep-zone radius (px) kept clear around placed seeds.
    epsilon          : DP tolerance (px).
    min_pts          : Minimum points per output polyline.
    """
    H, W = mask_bool.shape
    if not mask_bool.any():
        return []

    # ------------------------------------------------------------------
    # 1.  Luminance gradient → tangent field
    # ------------------------------------------------------------------
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(1.0, float(blur_sigma))

    # Fill outside-mask pixels with max luminance before smoothing so the
    # gradient at the mask boundary points inward (no spurious boundary pull).
    smooth = gray.copy()
    smooth[~mask_bool] = 255.0
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=sigma)
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=sigma)  # second pass

    gx  = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy  = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx * gx + gy * gy)

    # Unit tangent = 90° CCW rotation of normalised gradient
    safe_mag = np.where(mag < 0.5, 1.0, mag)
    tx = -gy / safe_mag   # tangent x-component
    ty =  gx / safe_mag   # tangent y-component

    # ------------------------------------------------------------------
    # 2.  Luminance thresholds (percentile-based, inside-mask pixels only)
    # ------------------------------------------------------------------
    lum_in = gray[mask_bool]
    if len(lum_in) == 0:
        return []

    n_t = max(1, n_tones)
    # Evenly-spaced percentiles from 20th to 85th of the in-mask luminance.
    # Tone 0 = up to 20th percentile (deepest shadow, densest strokes).
    # Tone n-1 = up to 85th percentile (mid-lights, sparsest strokes).
    pct_lo, pct_hi = 20.0, 85.0
    tone_thresholds = [
        float(np.percentile(lum_in,
              pct_lo + (pct_hi - pct_lo) * i / max(1, n_t - 1)))
        for i in range(n_t)
    ]

    # ------------------------------------------------------------------
    # 3.  Separation map + seed grid scan
    # ------------------------------------------------------------------
    sep       = np.zeros((H, W), dtype=np.bool_)
    polylines = []
    n_steps   = int(max_stroke_steps)

    # Budget cap: stop adding strokes when total interior point count reaches
    # point_budget (0 = unlimited).  Darkest-tone strokes are added first, so
    # the most visually important lines survive a tight budget.
    inner_budget = max(0, point_budget - 120) if point_budget > 0 else 0
    running_pts  = 0

    for tone_idx in range(n_t):
        lum_thresh = tone_thresholds[tone_idx]

        # Each successive tone band doubles the seed grid spacing (sparser).
        spacing = max(4, int(round(base_spacing * (2.0 ** tone_idx))))

        # Stagger start offsets per tone so grid lines don't coincide.
        y_start = (spacing // 2) + (tone_idx * 2) % max(1, spacing // 4)
        x_start = (spacing // 2) + (tone_idx * 3) % max(1, spacing // 4)

        # Vectorised validity pre-filter: build arrays of candidate seeds so
        # the Python-level loop only runs over accepted seeds, not the full grid.
        gy_arr   = np.arange(y_start, H, spacing)
        gx_arr   = np.arange(x_start, W, spacing)
        gys, gxs = np.meshgrid(gy_arr, gx_arr, indexing='ij')
        gys      = gys.ravel();  gxs = gxs.ravel()
        valid    = (mask_bool[gys, gxs]
                    & (gray[gys, gxs] <= lum_thresh)
                    & ~sep[gys, gxs])
        valid_gy = gys[valid];  valid_gx = gxs[valid]

        for gy_s, gx_s in zip(valid_gy.tolist(), valid_gx.tolist()):
            # Re-check sep_map — it changes as strokes are placed.
            if sep[gy_s, gx_s]:
                continue

            # Budget cap (0 = unlimited)
            if inner_budget > 0 and running_pts >= inner_budget:
                break

                # ---- Forward Euler integration (+T direction) ----
                fwd             = []
                cx, cy          = float(gx_s), float(gy_s)
                prev_dx = prev_dy = 0.0
                for _ in range(n_steps):
                    ix = int(np.clip(round(cx), 0, W - 1))
                    iy = int(np.clip(round(cy), 0, H - 1))
                    if not mask_bool[iy, ix]:
                        break
                    dtx = float(tx[iy, ix])
                    dty = float(ty[iy, ix])
                    # Resolve 180° field ambiguity: keep direction consistent
                    # with the previous step so strokes don't U-turn.
                    if prev_dx * dtx + prev_dy * dty < -0.1:
                        dtx, dty = -dtx, -dty
                    prev_dx, prev_dy = dtx, dty
                    cx += dtx
                    cy += dty
                    fwd.append((cx, cy))

                # ---- Backward Euler integration (-T direction) ----
                bwd             = []
                cx, cy          = float(gx_s), float(gy_s)
                prev_dx = prev_dy = 0.0
                for _ in range(n_steps):
                    ix = int(np.clip(round(cx), 0, W - 1))
                    iy = int(np.clip(round(cy), 0, H - 1))
                    if not mask_bool[iy, ix]:
                        break
                    dtx = float(tx[iy, ix])
                    dty = float(ty[iy, ix])
                    # Go in the opposite direction relative to the field.
                    dtx, dty = -dtx, -dty
                    if prev_dx * dtx + prev_dy * dty < -0.1:
                        dtx, dty = -dtx, -dty
                    prev_dx, prev_dy = dtx, dty
                    cx += dtx
                    cy += dty
                    bwd.append((cx, cy))

                bwd.reverse()
                full_path = bwd + [(float(gx_s), float(gy_s))] + fwd

                if len(full_path) < min_pts:
                    continue

                # ---- Paint separation bubble along the stroke ----
                r = max(1, min_sep // 2)
                for si, (px, py) in enumerate(full_path):
                    if si % 3 != 0:
                        continue   # sample every 3rd point for speed
                    ix = int(np.clip(round(px), 0, W - 1))
                    iy = int(np.clip(round(py), 0, H - 1))
                    y1 = max(0, iy - r);  y2 = min(H, iy + r + 1)
                    x1 = max(0, ix - r);  x2 = min(W, ix + r + 1)
                    sep[y1:y2, x1:x2] = True

                # ---- DP simplification ----
                pts_f32 = np.array(full_path, dtype=np.float32).reshape(-1, 1, 2)
                approx  = cv2.approxPolyDP(pts_f32, epsilon, False)
                pts_s   = approx.reshape(-1, 2)
                if len(pts_s) < min_pts:
                    continue

                # ---- Normalise to laser coords [-1,1], Y-up ----
                norm = [
                    [(float(px) / frame_w) * 2.0 - 1.0,
                     -(float(py) / frame_h) * 2.0 + 1.0]
                    for px, py in pts_s
                ]
                polylines.append({'pts': norm, 'closed': False})
                running_pts += len(norm)

    return polylines


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

    print(f'[vectorize] Masks: {total_frames} frames  ({frame_w}x{frame_h})', flush=True)

    # -------------------------------------------------------------------------
    # Determine which methods to run
    # -------------------------------------------------------------------------
    methods        = (['canny', 'hed', 'depth', 'thin', 'depth_iso', 'flow',
                       'hatch', 'lum_iso']
                      if args.method == 'all' else [args.method])
    use_hed        = 'hed'       in methods
    use_depth      = 'depth'     in methods or 'depth_iso' in methods
    use_canny      = 'canny'     in methods
    use_thin       = 'thin'      in methods
    use_depth_iso  = 'depth_iso' in methods
    use_flow       = 'flow'      in methods
    use_hatch      = 'hatch'     in methods
    use_lum_iso    = 'lum_iso'   in methods

    # -------------------------------------------------------------------------
    # Load ONNX runners if needed
    # -------------------------------------------------------------------------
    hed_runner   = None
    depth_runner = None

    if use_hed:
        if not os.path.exists(args.hed_model):
            print(f'[vectorize] HED model not found: {args.hed_model} — skipping hed',
                  flush=True)
            use_hed = False
            methods = [m for m in methods if m != 'hed']
        else:
            try:
                hed_runner = HEDRunner(args.hed_model, args.device)
            except Exception as e:
                print(f'[vectorize] HED load failed: {e} — skipping hed', flush=True)
                use_hed = False
                methods = [m for m in methods if m != 'hed']

    if use_depth:
        if not os.path.exists(args.depth_model):
            print(f'[vectorize] Depth model not found: {args.depth_model} — skipping depth/depth_iso',
                  flush=True)
            use_depth = use_depth_iso = False
            methods = [m for m in methods if m not in ('depth', 'depth_iso')]
        else:
            try:
                depth_runner = DepthRunner(args.depth_model, args.device)
            except Exception as e:
                print(f'[vectorize] Depth load failed: {e} — skipping depth/depth_iso', flush=True)
                use_depth = use_depth_iso = False
                methods = [m for m in methods if m not in ('depth', 'depth_iso')]

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
    prev_gray     = None
    prev_gray_raw = None   # unblended — used by optical flow
    prev_hed      = None
    prev_depth    = None
    alpha         = args.frame_avg_alpha

    toon_enabled = args.toon and (use_canny or use_hed or use_thin)
    print(f'[vectorize] Running  method={args.method}  '
          f'frame_avg_alpha={alpha}  spline_samples={args.spline_samples}'
          + ('  toon=on' if toon_enabled else ''), flush=True)

    for frame_idx in range(total_frames):
        ret, bgr = cap.read()
        if not ret:
            print(f'[vectorize] Warning: ran out of video frames at {frame_idx}', flush=True)
            break

        mask = all_masks[frame_idx]   # bool [H, W]

        # --- Raw greyscale — always from the original frame (optical flow) ---
        gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- Toon-preprocessed frame for HED / Canny / Thin (if --toon) ---
        # Hatch uses raw bgr for its own luminance gradient; depth/flow are
        # unaffected by toon.
        if toon_enabled:
            frame_edge = toon_preprocess(
                bgr, args.toon_d,
                args.toon_sigma_color, args.toon_sigma_space,
                args.toon_colors)
            gray_edge_raw = cv2.cvtColor(frame_edge, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            frame_edge    = bgr
            gray_edge_raw = gray_raw

        # --- Temporally blended greyscale (used by Canny / Thin) ---
        gray = gray_edge_raw.copy()
        if prev_gray is not None and alpha < 1.0:
            gray = alpha * gray + (1.0 - alpha) * prev_gray
        prev_gray = gray

        hed_map = None
        if use_hed:
            hed_map = hed_runner.infer(frame_edge)   # toon frame if --toon
            if prev_hed is not None and alpha < 1.0:
                hed_map = alpha * hed_map + (1.0 - alpha) * prev_hed
            prev_hed = hed_map

        depth_map = None
        if use_depth:
            depth_map = depth_runner.infer(bgr)
            if prev_depth is not None and alpha < 1.0:
                depth_map = alpha * depth_map + (1.0 - alpha) * prev_depth
            prev_depth = depth_map

        # --- Outer contour (shared across all methods for this frame) ---
        outer = mask_outer_contours(mask, frame_w, frame_h, min_area_px,
                                    args.smooth_epsilon)

        # --- Interior edges per method ---
        for m in methods:
            interior = []
            if m == 'canny':
                interior = interior_canny(
                    mask, gray, frame_w, frame_h,
                    args.canny_low, args.canny_high, args.canny_blur,
                    args.canny_epsilon, args.canny_min_pts)
            elif m == 'hed' and hed_map is not None:
                interior = interior_hed(
                    mask, hed_map, frame_w, frame_h,
                    args.hed_threshold, args.hed_blur,
                    args.hed_epsilon, args.hed_min_pts, args.hed_min_len)
            elif m == 'depth' and depth_map is not None:
                interior = interior_depth(
                    mask, depth_map, frame_w, frame_h,
                    args.depth_threshold, args.depth_epsilon, args.depth_min_pts)
            elif m == 'thin':
                interior = interior_thin(
                    mask, gray, frame_w, frame_h,
                    args.thin_thresh, args.thin_blur,
                    args.thin_epsilon, args.thin_min_pts)
            elif m == 'depth_iso' and depth_map is not None:
                interior = interior_depth_iso(
                    mask, depth_map, frame_w, frame_h,
                    args.iso_levels, args.iso_epsilon, args.iso_min_pts)
            elif m == 'flow':
                interior = interior_flow(
                    mask, gray_raw, prev_gray_raw, frame_w, frame_h,
                    args.flow_threshold, args.flow_blur,
                    args.flow_epsilon, args.flow_min_pts)
            elif m == 'hatch':
                interior = interior_hatch(
                    mask, bgr, frame_w, frame_h,
                    args.hatch_blur, args.hatch_tones, args.hatch_spacing,
                    args.hatch_max_len, args.hatch_min_sep,
                    args.hatch_epsilon, args.hatch_min_pts,
                    point_budget=args.point_budget)
            elif m == 'lum_iso':
                interior = interior_lum_iso(
                    mask, gray, frame_w, frame_h,
                    args.lum_iso_levels, args.lum_iso_epsilon,
                    args.lum_iso_min_pts, args.lum_iso_blur)

            frame_polys = outer + interior
            frame_polys = order_polylines(frame_polys)
            frame_polys = apply_point_budget(frame_polys, args.point_budget)
            # Hatch strokes follow the smooth gradient field and need no further
            # spline fitting — it would only inflate point count unnecessarily.
            if m != 'hatch':
                frame_polys = apply_spline_fitting(frame_polys, args.spline_samples)
            frames_by_method[m][frame_idx] = frame_polys

        # Store raw gray AFTER dispatch so frame 0 gives flow an empty prev
        prev_gray_raw = gray_raw

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
            'point_budget': args.point_budget,
        }

        if args.method == 'all':
            out_path = _stem_path(args.output, f'_{m}')
        else:
            out_path = args.output

        _save_json(out_path, meta, frames_clean)


if __name__ == '__main__':
    main()
