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
    resources/polylines/{stem}_sam2-{model}_pidinet.json
    resources/polylines/{stem}_sam2-{model}_teed.json
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
      {"pts": [[x, y], ...], "closed": true},   <- outer SAM2 contour
      {"pts": [[x, y], ...], "closed": false}   <- interior edge path
    ],
    ...
  ]
}

Coordinates are normalized to [-1, 1] with Y-up (laser convention).
Color (always white) is added by encode.py.
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
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2

from edge_models import (HEDRunner,
                         PiDiNetRunner, TEEDRunner, EDTERRunner, DiffusionEdgeRunner)


# =============================================================================
# Args
# =============================================================================

_ALL_METHODS = ['canny', 'hed', 'pidinet', 'teed', 'edter', 'diffusion_edge']

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
    p.add_argument('--pidinet-model',        default='models/pidinet/model.onnx')
    p.add_argument('--teed-model',           default='models/teed/model.onnx')
    p.add_argument('--edter-model',          default='models/edter/EDTER-BSDS-VOC-StageI.pth')
    p.add_argument('--diffusion-edge-model',             default='models/diffusion_edge/bsds.pt')
    p.add_argument('--diffusion-edge-first-stage-model', default='models/diffusion_edge/first_stage_total_320.pt')
    p.add_argument('--diffusion-edge-config',            default='models/diffusion_edge/bsds_sample.yaml')
    p.add_argument('--device',               default='cuda' if __import__('torch').cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    # Outer contour
    p.add_argument('--min-area',        type=float, default=0.001,
                   help='Minimum mask area as fraction of frame area')
    p.add_argument('--smooth-epsilon',  type=float, default=2.0,
                   help='approxPolyDP epsilon for outer contours (pixels)')
    # Frame averaging
    p.add_argument('--frame-avg-alpha', type=float, default=0.7,
                   help='Signal temporal blend: 1.0=off, lower=more averaging')
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
    p.add_argument('--hed-min-len',     type=int,   default=20,
                   help='Minimum arc length in pixels. Strokes shorter than this '
                        'are discarded as speckle.')
    # PiDiNet
    p.add_argument('--pidinet-threshold', type=float, default=0.35)
    p.add_argument('--pidinet-blur',      type=int,   default=3)
    p.add_argument('--pidinet-epsilon',   type=float, default=1.5)
    p.add_argument('--pidinet-min-pts',   type=int,   default=3)
    p.add_argument('--pidinet-min-len',   type=int,   default=20)
    # TEED
    p.add_argument('--teed-threshold',   type=float, default=0.35)
    p.add_argument('--teed-blur',        type=int,   default=3)
    p.add_argument('--teed-epsilon',     type=float, default=1.5)
    p.add_argument('--teed-min-pts',     type=int,   default=3)
    p.add_argument('--teed-min-len',     type=int,   default=20)
    # EDTER
    p.add_argument('--edter-threshold',  type=float, default=0.35)
    p.add_argument('--edter-blur',       type=int,   default=3)
    p.add_argument('--edter-epsilon',    type=float, default=1.5)
    p.add_argument('--edter-min-pts',    type=int,   default=3)
    p.add_argument('--edter-min-len',    type=int,   default=20)
    # DiffusionEdge
    p.add_argument('--diffusion-edge-threshold', type=float, default=0.35)
    p.add_argument('--diffusion-edge-blur',      type=int,   default=3)
    p.add_argument('--diffusion-edge-epsilon',   type=float, default=1.5)
    p.add_argument('--diffusion-edge-min-pts',   type=int,   default=3)
    p.add_argument('--diffusion-edge-min-len',   type=int,   default=20)
    p.add_argument('--diffusion-edge-steps',     type=int,   default=5,
                   help='Denoising steps for DiffusionEdge (fewer = faster, less crisp)')
    # Spline fitting (applied to all methods)
    p.add_argument('--spline-samples',  type=int,   default=4,
                   help='Catmull-Rom samples per segment between DP vertices '
                        '(1=off, 4=smooth, 8=very smooth)')
    return p.parse_args()


# =============================================================================
# Shared helpers
# =============================================================================

_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def _erode_mask(mask_u8):
    inner = cv2.erode(mask_u8, _ERODE_KERNEL, iterations=1)
    return inner if cv2.countNonZero(inner) > 0 else None


def _contours_to_polys(contours, frame_w, frame_h, min_pts, epsilon, closed, min_len=0):
    """OpenCV contours → list of {'pts': [[x,y],...], 'closed': bool}.

    min_len: minimum arc length in pixels.  Contours shorter than this are
    discarded after approxPolyDP so that speckle fragments don't survive.
    """
    out = []
    for cnt in contours:
        if len(cnt) < min_pts:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed) if epsilon > 0 else cnt
        pts    = approx.reshape(-1, 2)
        if len(pts) < min_pts:
            continue
        if min_len > 0 and cv2.arcLength(approx, closed) < min_len:
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


def interior_edgemap(mask_bool, edge_map, frame_w, frame_h,
                     threshold, blur_k, epsilon, min_pts, min_len,
                     thin=False):
    """
    Interior edge extraction from a float [0,1] edge-probability map.
    Used by all neural edge detectors (HED, EDTER, DiffusionEdge, PiDiNet, TEED).

    thin=False (default):
        findContours traces the outer boundary of each thresholded blob.
        For thin blobs the boundary ≈ a single stroke.

    thin=True (EDTER):
        EDTER outputs wider probability blobs; the blob boundary produces
        double-outline rings on strong edges.  Morphological thinning
        (Zhang-Suen) collapses each blob to a 1-pixel skeleton before
        contour tracing, giving clean single strokes.
    """
    inner = _erode_mask(mask_bool.astype(np.uint8) * 255)
    if inner is None:
        return []

    edge_u8 = (np.clip(edge_map, 0.0, 1.0) * 255).astype(np.uint8)

    # Median blur reduces salt-and-pepper speckle without smearing edges
    if blur_k > 1:
        k = blur_k | 1
        edge_u8 = cv2.medianBlur(edge_u8, k)

    binary = (edge_u8 > int(threshold * 255)).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, inner)

    # Morphological thinning: collapse thick blobs to 1-pixel skeletons.
    # Prefer opencv-contrib (C++, fast); fall back to skimage.
    if thin:
        try:
            binary = cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            from skimage.morphology import skeletonize as _skeletonize
            skel   = _skeletonize(binary > 0)
            binary = skel.astype(np.uint8) * 255

    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polys(cnts, frame_w, frame_h, min_pts, epsilon,
                              closed=False, min_len=min_len)


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

    # All runners are now PyTorch-based — no onnxruntime anywhere.
    # HED auto-downloads weights from content.sniklaus.com on first run.
    edter_runner          = _try_load('edter',          args.edter_model,          EDTERRunner)
    diffusion_edge_runner = _try_load('diffusion_edge', args.diffusion_edge_model, DiffusionEdgeRunner,
                                      first_stage_path=args.diffusion_edge_first_stage_model,
                                      config_path=args.diffusion_edge_config,
                                      sampling_steps=args.diffusion_edge_steps)
    hed_runner            = _try_load('hed',     args.hed_model,     HEDRunner)
    pidinet_runner        = _try_load('pidinet', args.pidinet_model, PiDiNetRunner)
    teed_runner           = _try_load('teed',    args.teed_model,    TEEDRunner)

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
    prev_pidinet        = None
    prev_teed           = None
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

        # --- Greyscale, temporally blended (used by Canny) ---
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None and alpha < 1.0:
            gray = alpha * gray + (1.0 - alpha) * prev_gray
        prev_gray = gray

        def _run_and_blend(runner, bgr_frame, prev_map):
            """Run a runner and apply temporal blending. Returns (new_map, new_prev)."""
            if runner is None:
                return None, prev_map
            m = runner.infer(bgr_frame)
            if prev_map is not None and alpha < 1.0:
                m = alpha * m + (1.0 - alpha) * prev_map
            return m, m

        hed_map,            prev_hed            = _run_and_blend(hed_runner,            bgr, prev_hed)
        pidinet_map,        prev_pidinet        = _run_and_blend(pidinet_runner,        bgr, prev_pidinet)
        teed_map,           prev_teed           = _run_and_blend(teed_runner,           bgr, prev_teed)
        edter_map,          prev_edter          = _run_and_blend(edter_runner,          bgr, prev_edter)
        diffusion_edge_map, prev_diffusion_edge = _run_and_blend(diffusion_edge_runner, bgr, prev_diffusion_edge)

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
                interior = interior_edgemap(
                    mask, hed_map, frame_w, frame_h,
                    args.hed_threshold, args.hed_blur,
                    args.hed_epsilon, args.hed_min_pts, args.hed_min_len)
            elif m == 'pidinet' and pidinet_map is not None:
                interior = interior_edgemap(
                    mask, pidinet_map, frame_w, frame_h,
                    args.pidinet_threshold, args.pidinet_blur,
                    args.pidinet_epsilon, args.pidinet_min_pts, args.pidinet_min_len)
            elif m == 'teed' and teed_map is not None:
                interior = interior_edgemap(
                    mask, teed_map, frame_w, frame_h,
                    args.teed_threshold, args.teed_blur,
                    args.teed_epsilon, args.teed_min_pts, args.teed_min_len)
            elif m == 'edter' and edter_map is not None:
                interior = interior_edgemap(
                    mask, edter_map, frame_w, frame_h,
                    args.edter_threshold, args.edter_blur,
                    args.edter_epsilon, args.edter_min_pts, args.edter_min_len,
                    thin=True)
            elif m == 'diffusion_edge' and diffusion_edge_map is not None:
                interior = interior_edgemap(
                    mask, diffusion_edge_map, frame_w, frame_h,
                    args.diffusion_edge_threshold, args.diffusion_edge_blur,
                    args.diffusion_edge_epsilon, args.diffusion_edge_min_pts,
                    args.diffusion_edge_min_len)

            frame_polys = outer + interior
            frame_polys = apply_spline_fitting(frame_polys, args.spline_samples)
            frames_by_method[m][frame_idx] = frame_polys

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
