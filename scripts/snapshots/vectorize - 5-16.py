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
    p.add_argument('--edter-model',          default='models/edter/EDTER-BSDS-VOC-StageII.pth')
    p.add_argument('--diffusion-edge-model',             default='models/diffusion_edge/bsds.pt')
    p.add_argument('--diffusion-edge-first-stage-model', default='models/diffusion_edge/first_stage_total_320.pt')
    p.add_argument('--diffusion-edge-config',            default='models/diffusion_edge/bsds_sample.yaml')
    p.add_argument('--device',               default='cuda' if __import__('torch').cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    p.add_argument('--edge-video',          default='resources/polylines/',
                   help='Optional output MP4 path or prefix for model edge-map video(s). '
                        'If multiple methods are run, method suffixes are appended.')
    p.add_argument('--edge-video-fps',      type=float, default=None,
                   help='Frame rate for edge-map output video(s). Defaults to source video fps.')
    # Outer contour
    p.add_argument('--min-area',        type=float, default=0.001,
                   help='Minimum mask area as fraction of frame area')
    p.add_argument('--smooth-epsilon',  type=float, default=0.0004,
                   help='approxPolyDP epsilon for outer contours as fraction of '
                        'image diagonal.  Converted to pixels at runtime.')
    # Frame averaging
    p.add_argument('--frame-avg-alpha', type=float, default=1.0,
                   help='Signal temporal blend: 1.0=off, lower=more averaging')
    # Canny
    p.add_argument('--canny-low',       type=int,   default=40)
    p.add_argument('--canny-high',      type=int,   default=120)
    p.add_argument('--canny-blur',      type=int,   default=3)
    p.add_argument('--canny-epsilon',   type=float, default=0.0007,
                   help='approxPolyDP epsilon as fraction of image diagonal.')
    p.add_argument('--canny-min-pts',   type=int,   default=4)
    # HED  (gradient-mask → Steger ridge tracking → V-W simplify)
    p.add_argument('--hed-sigma',         type=float, default=1.0,
                   help='Gaussian smoothing σ for the Steger Hessian.  Should '
                        'be roughly the half-width of the ridges to detect.  '
                        '1.5 suits HED\'s ~1-2 pixel ridges; raise (2.0-3.0) '
                        'to fuse parallel close strands, lower (1.0) for '
                        'sharper isolated ridges.')
    p.add_argument('--hed-strength',      type=float, default=0.001,
                   help='Min Steger ridge strength (-λ_perpendicular).  Pixels '
                        'with weaker Hessian curvature perpendicular to the '
                        'ridge are dropped.  Lower = more ridges (incl. noise); '
                        'higher = only the most pronounced ridges.')
    p.add_argument('--hed-step-angle',    type=float, default=30.0,
                   help='Max angle (degrees) between consecutive ridge-pixel '
                        'tangents during Steger linking.  Higher = walks '
                        'tolerate sharper bends; lower = stricter straightness.')
    p.add_argument('--hed-mask-gain-high',type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is strong.  Sharpens confident edges.')
    p.add_argument('--hed-mask-gain-mid', type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is intermediate.')
    p.add_argument('--hed-mask-gain-low', type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is weak.  0 kills dim edge continuations '
                        '(breaks continuity); 0.5 preserves them at half weight; '
                        '1.0 disables the gradient mask entirely.')
    p.add_argument('--hed-link-kernel',   type=int,   default=10,
                   help='Max gap (pixels) for polyline-tip directional linking '
                        'after Steger.  1 = disabled.  Steger usually produces '
                        'continuous polylines on its own; this is a safety net '
                        'for occasional 1-3 px gaps where the Hessian was '
                        'unreliable.')
    p.add_argument('--hed-link-angle',    type=float, default=20.0,
                   help='Half-angle of tangent acceptance cone for polyline linking '
                        '(degrees).')
    p.add_argument('--hed-link-tangent-k',type=int,   default=12,
                   help='Number of tip pixels averaged for the polyline-tip tangent.')
    p.add_argument('--hed-link-soft-th',  type=float, default=0.05,
                   help='Min raw-HED probability required along a link path.')
    p.add_argument('--hed-link-co-circ',  type=float, default=16.0,
                   help='Max |α-β| in degrees for co-circular arc linking '
                        '(handles circular gaps where both tips deviate equally).')
    p.add_argument('--hed-simplify-area', type=float, default=1.0,
                   help='Visvalingam-Whyatt minimum effective triangle area '
                        '(pixels²) for polyline simplification.  Higher = fewer '
                        'points / coarser polylines.  0 = disabled.')
    p.add_argument('--hed-mask-erode',    type=int,   default=0,
                   help='SAM2 mask erosion kernel size (px) before bounding '
                        'the Steger search. 0 (default) disables erosion — '
                        'EDTER edges trace the silhouette precisely and any '
                        'erosion kills them. Raise (e.g. 5, 9) to suppress '
                        'edges near the mask boundary if needed.')
    p.add_argument('--hed-mask-dilate',   type=int,   default=9,
                   help='Kernel size (px) for dilating the SAM2 mask before '
                        'the Steger ridge search.  Gives Gaussian derivative '
                        'kernels breathing room at the subject boundary so the '
                        'Hessian is not distorted by hard zeros.  Post-detection '
                        'filter_bool removes any background specks.  Default 9.')
    p.add_argument('--hed-min-pts',     type=int,   default=5)
    p.add_argument('--hed-min-len',     type=int,   default=50,
                   help='Minimum polyline arc length in pixels.  Polylines '
                        'shorter than this are discarded as residual speckle.')
    p.add_argument('--hed-spread',      type=float, default=0.0,
                   help='Per-polyline intensity contrast: '
                        'new = clamp(mean + (raw - mean) * spread, 0, 1). '
                        '1.0=no change, >1=more contrast, <1=softer, 0=uniform.')
    # EDTER
    p.add_argument('--edter-stage',      type=int,   default=2, choices=[1, 2],
                   help='EDTER inference stage. 1 = global ViT-Large/16 only '
                        '(faster, coarser).  2 = full two-stage with local '
                        'ViT-Base/8 + SFT-FFM refinement (default; sharper '
                        'edges, ~1.6-2× slower per tile).  Auto-falls-back to '
                        'stage 1 if the supplied checkpoint lacks Stage II '
                        '(fuse_head.*) weights.')
    # EDTER Steger backend (mirror --hed-* args; EDTER soft maps are sharper
    # than HED's so strength_th is raised, link_kernel disabled by default,
    # mask_erode disabled by default since EDTER edges trace the silhouette
    # boundary precisely and erosion would kill them).
    p.add_argument('--edter-sigma',         type=float, default=4.0,
                   help='Gaussian smoothing σ for the Steger Hessian (px). '
                        'See --hed-sigma.')
    p.add_argument('--edter-strength',      type=float, default=0.005,
                   help='Min Steger ridge strength (-λ_perpendicular). '
                        'Higher than --hed-strength because EDTER ridges are '
                        'sharper and tolerate a stricter floor.')
    p.add_argument('--edter-step-angle',    type=float, default=90.0,
                   help='Max angle (degrees) between consecutive ridge-pixel '
                        'tangents during Steger linking. See --hed-step-angle.')
    p.add_argument('--edter-mask-gain-high',type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is strong.  Sharpens confident edges.')
    p.add_argument('--edter-mask-gain-mid', type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is intermediate.')
    p.add_argument('--edter-mask-gain-low', type=float, default=1.0,
                   help='Gradient-mask filter: probability multiplier where image '
                        'gradient is weak.  0 kills dim edge continuations '
                        '(breaks continuity); 0.5 preserves them at half weight; '
                        '1.0 disables the gradient mask entirely.')
    p.add_argument('--edter-link-kernel',   type=int,   default=1,
                   help='Polyline-tip directional linking gap (px). 1 = OFF. '
                        'EDTER + Steger usually produces continuous polylines '
                        'on its own; raise to enable gap bridging.')
    p.add_argument('--edter-link-angle',    type=float, default=30.0,
                   help='Half-angle (deg) of tangent acceptance cone for linking.')
    p.add_argument('--edter-link-tangent-k',type=int,   default=12,
                   help='Number of tip pixels averaged for the polyline tangent.')
    p.add_argument('--edter-link-soft-th',  type=float, default=0.05,
                   help='Min raw EDTER probability required along a link path.')
    p.add_argument('--edter-link-co-circ',  type=float, default=12.0,
                   help='Max |α-β| (deg) for co-circular arc linking.')
    p.add_argument('--edter-simplify-area', type=float, default=1.5,
                   help='Visvalingam-Whyatt min effective triangle area (px²) '
                        'for polyline simplification. 0 = disabled.')
    p.add_argument('--edter-mask-erode',    type=int,   default=0,
                   help='SAM2 mask erosion kernel size (px) before bounding '
                        'the Steger search. 0 (default) disables erosion — '
                        'EDTER edges trace the silhouette precisely and any '
                        'erosion kills them. Raise (e.g. 5, 9) to suppress '
                        'edges near the mask boundary if needed.')
    p.add_argument('--edter-mask-dilate',   type=int,   default=9,
                   help='Same as --hed-mask-dilate but for EDTER.  Default 9.')
    p.add_argument('--edter-min-pts',     type=int,   default=3)
    p.add_argument('--edter-min-len',     type=int,   default=40,
                   help='Minimum polyline arc length in pixels.')
    p.add_argument('--edter-spread',      type=float, default=0.5,
                   help='Per-polyline intensity contrast modulator (see --hed-spread).')
    p.add_argument('--edter-tile-blend',  choices=['gaussian', 'uniform'],
                   default='gaussian',
                   help='How overlapping 320×320 EDTER tiles are combined. '
                        'gaussian (default) = weighted mean with center-of-tile '
                        'dominance, removes 8×8 patch-grid artifacts at tile '
                        'seams. uniform = simple average (legacy).')


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

    # 1. Create a smoothed version ONLY for gradient calculation
    # This prevents 'jittery' directions without losing the raw peak values
    smoothed = cv2.GaussianBlur(prob_map, (5, 5), 0)
    smoothed = cv2.GaussianBlur(prob_map, (5, 5), 0)
    
    # Use a larger Sobel kernel: gradient direction at the *peak* of a soft
    # ridge is dominated by noise with ksize=3 because the magnitude is
    # near zero there.  ksize=5 averages over a wider neighbourhood and
    # gives a stable perpendicular-to-ridge direction even at ridge tops.
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
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


# -----------------------------------------------------------------------------
# HED pipeline helpers
# -----------------------------------------------------------------------------

def _refine_hed_map_with_gradient_mask(bgr_frame, edge_map,
                                       low_mask_th=0.05, high_mask_th=0.25,
                                       gain_high=1.0, gain_mid=1.0, gain_low=1.0):
    """Han & Zhong (2020) gradient-mask filter.  Multiplies the HED soft
    probability map by image-gradient-magnitude bins, sharpening genuine
    edges and suppressing smooth-region false positives.

    gain_low controls preservation of edge sections that cross
    low-image-gradient regions (e.g. a dim continuation in the middle of
    a long edge).  Historical default 0.0 killed those sections entirely
    and broke continuity; 0.5 preserves them at half weight; 1.0 disables
    the gradient mask.  Tunable via CLI (--hed-mask-gain-low).
    """
    gray   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag    = cv2.magnitude(grad_x, grad_y)
    mag    = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    high_mask = (mag >= high_mask_th).astype(np.float32)
    low_mask  = (mag <  low_mask_th).astype(np.float32)
    mid_mask  = ((mag >= low_mask_th) & (mag < high_mask_th)).astype(np.float32)

    refined = edge_map * (high_mask * gain_high
                          + mid_mask * gain_mid
                          + low_mask * gain_low)
    return np.clip(refined, 0.0, 1.0)


def _gaussian_derivative_kernels(sigma):
    """Return (gx, gy, gxx, gyy, gxy) kernels — analytical Gaussian derivatives.

    These are the exact kernels used by Steger (1998):
      G(x,y)    = exp(-(x²+y²)/(2σ²)) / (2πσ²)
      Gx(x,y)   = -x/σ² · G
      Gxx(x,y)  = (x²-σ²)/σ⁴ · G
      Gxy(x,y)  =  x·y/σ⁴  · G

    Using these avoids the Sobel-of-Sobel approximation, which produces
    an incorrect rxy term at diagonal orientations (≈45°) — exactly where
    circular arcs spend half their length.  Correct rxy → correct Hessian
    eigenvectors → correct tangent estimates → smooth circle walks.
    """
    r = max(3, int(np.ceil(3.0 * sigma)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()               # normalise so DC = 1 → unit-gain smoothing
    s2 = sigma ** 2
    s4 = sigma ** 4
    gx  = (-x / s2) * g
    gy  = (-y / s2) * g
    gxx = ((x ** 2 - s2) / s4) * g
    gyy = ((y ** 2 - s2) / s4) * g
    gxy = (x * y / s4) * g
    return (gx.astype(np.float32),  gy.astype(np.float32),
            gxx.astype(np.float32), gyy.astype(np.float32),
            gxy.astype(np.float32))


def _steger_ridge_points(soft_f32, sigma=1.5, strength_th=0.005):
    """Steger (1998) sub-pixel ridge detection on a soft probability map.

    For each pixel, computes the 2x2 Hessian of a Gaussian-smoothed copy
    of the input, then eigendecomposes it.  The eigenvector with the
    larger-magnitude eigenvalue points perpendicular to a ridge (n);
    the other eigenvector is the ridge tangent (t).  A pixel is a
    ridge point if the sub-pixel offset along n stays within the pixel
    AND the perpendicular eigenvalue is sufficiently negative
    (bright-ridge-on-dark assumption).  Sub-pixel position is recovered
    from the gradient projection along n.

    sigma        : Gaussian smoothing scale; should be ≈ ridge half-width.
                   For HED's ~1-2 pixel ridges, 1.5 is a good default.
    strength_th  : Minimum -λ_perpendicular to count as a ridge.  Larger
                   = fewer, more confident ridge points.

    Returns six 1-D arrays (one entry per detected ridge pixel):
      ys, xs       : int  pixel coordinates (top-left of the pixel)
      sub_y, sub_x : float sub-pixel ridge position
      tan_x, tan_y : float unit tangent vector
      strength     : float -λ_perpendicular (always > 0 for valid ridges)
    """
    eps = 1e-10

    # All five derivative maps from one set of analytical Gaussian kernels.
    # This is the correct Steger formulation — do NOT use Sobel-of-Sobel,
    # which gives an incorrect rxy at diagonal orientations (≈45°) and
    # therefore wrong eigenvectors for curved edges such as circles.
    gx_k, gy_k, gxx_k, gyy_k, gxy_k = _gaussian_derivative_kernels(sigma)
    rx  = cv2.filter2D(soft_f32, cv2.CV_32F, gx_k)
    ry  = cv2.filter2D(soft_f32, cv2.CV_32F, gy_k)
    rxx = cv2.filter2D(soft_f32, cv2.CV_32F, gxx_k)
    ryy = cv2.filter2D(soft_f32, cv2.CV_32F, gyy_k)
    rxy = cv2.filter2D(soft_f32, cv2.CV_32F, gxy_k)

    # Closed-form 2x2 symmetric eigendecomposition.
    trace = rxx + ryy
    det   = rxx * ryy - rxy * rxy
    disc  = np.sqrt(np.maximum(0.0, (trace * 0.5) ** 2 - det))
    lam_p = trace * 0.5 + disc        # algebraically larger eigenvalue
    lam_m = trace * 0.5 - disc        # algebraically smaller eigenvalue

    # Pick the eigenvalue with greater absolute magnitude — that one's
    # eigenvector is perpendicular to the ridge.  For bright ridges on
    # a dark background the chosen one is negative.
    pick_p = np.abs(lam_p) > np.abs(lam_m)
    lam_n  = np.where(pick_p, lam_p, lam_m)
    strength_full = -lam_n             # positive for bright ridges

    # Eigenvector for lam_n: any non-zero column of (H - lam_n·I).
    # Use (lam_n - ryy, rxy); fall back to (rxy, lam_n - rxx) where
    # the first is degenerate, then to canonical basis when both are.
    nx_a = lam_n - ryy
    ny_a = rxy
    norm_a = np.hypot(nx_a, ny_a)
    nx_b = rxy
    ny_b = lam_n - rxx
    norm_b = np.hypot(nx_b, ny_b)
    use_a = norm_a >= norm_b
    nx_raw = np.where(use_a, nx_a, nx_b)
    ny_raw = np.where(use_a, ny_a, ny_b)
    n_norm = np.hypot(nx_raw, ny_raw)
    # Where both candidates collapsed (diagonal Hessian), pick canonical.
    deg = n_norm < eps
    if np.any(deg):
        rxx_dom = np.abs(rxx) >= np.abs(ryy)
        nx_raw = np.where(deg, np.where(rxx_dom, 1.0, 0.0), nx_raw)
        ny_raw = np.where(deg, np.where(rxx_dom, 0.0, 1.0), ny_raw)
        n_norm = np.where(deg, 1.0, n_norm)
    nx = nx_raw / n_norm
    ny = ny_raw / n_norm
    # Tangent perpendicular to n
    tx = -ny
    ty =  nx

    # Sub-pixel ridge offset along n:
    #   p = -(rx·nx + ry·ny) / (rxx·nx² + 2·rxy·nx·ny + ryy·ny²)
    denom = rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny
    numer = -(rx * nx + ry * ny)
    # Safe divide (np.where evaluates both branches eagerly → divide-by-zero
    # warning even when masked out).  Replace tiny denominators with 1.0
    # before division and zero the result via the mask afterwards.
    safe_denom = np.where(np.abs(denom) > eps, denom, 1.0)
    p = np.where(np.abs(denom) > eps, numer / safe_denom, 0.0)
    off_x = p * nx
    off_y = p * ny

    # Ridge condition: sub-pixel offset stays inside the pixel AND
    # ridge strength clears the floor.
    is_ridge = (np.abs(off_x) <= 0.5) & (np.abs(off_y) <= 0.5) & \
               (strength_full >= strength_th)
    ys, xs = np.where(is_ridge)
    return (ys.astype(np.int32),
            xs.astype(np.int32),
            ys + off_y[ys, xs],
            xs + off_x[ys, xs],
            tx[ys, xs],
            ty[ys, xs],
            strength_full[ys, xs])


def _steger_link_polylines(ys, xs, sub_y, sub_x, tan_x, tan_y, strength,
                           img_h, img_w, max_step_angle_deg=60.0):
    """Link Steger ridge pixels into continuous polylines.

    For each ridge pixel (in descending strength order) start a walk in
    the +tangent direction: pick the unconsumed 8-neighbor ridge pixel
    whose displacement from the current pixel best aligns with the
    current tangent AND whose own tangent is approximately parallel.
    When the walk dead-ends, restart from the seed in the -tangent
    direction.  Concatenate to produce one polyline per seed.

    Each ridge pixel is consumed at most once → no Y-forks.

    Returns: list of dicts {'path': [(x_sub, y_sub), ...], 'closed': bool}.
    """
    n = len(ys)
    if n == 0:
        return []

    # O(1) (y, x) → ridge-index lookup via a sparse 2-D array.
    pix_idx = np.full((img_h, img_w), -1, dtype=np.int32)
    pix_idx[ys, xs] = np.arange(n, dtype=np.int32)

    cos_thr = math.cos(math.radians(max_step_angle_deg))
    consumed = np.zeros(n, dtype=bool)
    polylines = []

    # Strongest seed first
    order = np.argsort(-strength).tolist()

    # 8-neighbour offsets and their unit step direction
    offs = [(dy, dx, dy / math.hypot(dy, dx) if (dy or dx) else 0.0,
                     dx / math.hypot(dy, dx) if (dy or dx) else 0.0)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)]

    for seed in order:
        if consumed[seed]:
            continue
        consumed[seed] = True

        forward  = [(float(sub_x[seed]), float(sub_y[seed]))]
        backward = []

        for sign in (+1, -1):
            cur = seed
            cur_tx = sign * float(tan_x[cur])
            cur_ty = sign * float(tan_y[cur])
            target = forward if sign > 0 else backward

            while True:
                cy = int(ys[cur]); cx = int(xs[cur])
                best_i = -1; best_align = -2.0
                # Look at 8-connected ridge neighbours
                for dy, dx, sdy, sdx in offs:
                    ny2 = cy + dy; nx2 = cx + dx
                    if not (0 <= ny2 < img_h and 0 <= nx2 < img_w):
                        continue
                    ni = pix_idx[ny2, nx2]
                    if ni < 0 or consumed[ni]:
                        continue
                    # Step direction must be in the +tangent half-plane
                    align_step = sdx * cur_tx + sdy * cur_ty
                    if align_step <= 0.0:
                        continue
                    # Neighbour's tangent must be approximately parallel
                    ntx = float(tan_x[ni]); nty = float(tan_y[ni])
                    align_tan = abs(ntx * cur_tx + nty * cur_ty)
                    if align_tan < cos_thr:
                        continue
                    # Combined score: prefer well-aligned step direction,
                    # tie-broken by tangent agreement.
                    score = align_step + 0.1 * align_tan
                    if score > best_align:
                        best_align = score; best_i = ni

                if best_i < 0:
                    break

                consumed[best_i] = True
                # Carry tangent forward, flipping if needed to maintain heading.
                ntx = float(tan_x[best_i]); nty = float(tan_y[best_i])
                if ntx * cur_tx + nty * cur_ty < 0.0:
                    ntx = -ntx; nty = -nty
                cur = best_i
                cur_tx = ntx; cur_ty = nty
                target.append((float(sub_x[cur]), float(sub_y[cur])))

        path = list(reversed(backward)) + forward
        if len(path) >= 2:
            polylines.append({'path': path, 'closed': False})

    return polylines


def _link_polyline_endpoints(polylines, soft_f32, max_gap, max_angle_deg,
                              tangent_k=12, soft_th=0.05, co_circ_deg=12.0):
    """Polyline-tip directed gap closing.  Same co-circularity geometry as
    the (now-removed) pixel-level `_directed_endpoint_bridge`, but operating
    on polyline tips (~50–200 endpoints typical) rather than every skeleton
    endpoint (~thousands).  Tangent estimates average over `tangent_k`
    polyline pixels — far more reliable than a 3–5 pixel skeleton lookback.

    On accept: the two polylines are concatenated (with the bridge pixels
    inserted) and the originals are removed.  Each tip is consumed at most
    once → no Y-forks.
    """
    if max_gap <= 1 or len(polylines) < 2:
        return list(polylines)

    H, W       = soft_f32.shape
    cos_thr    = math.cos(math.radians(max_angle_deg))
    cos_2thr   = math.cos(math.radians(min(89.9, 2.0 * max_angle_deg)))
    co_circ_r  = math.radians(co_circ_deg)
    gap2       = float(max_gap * max_gap)

    # Build endpoint registry.  Each open polyline contributes 2 tips
    # (head=index 0, tail=index 1).  Closed polylines contribute none.
    tips = []   # list of {'pl': i, 'side': 'h'|'t', 'xy': (x,y), 'tan': (tx,ty)}
    for pi, pl in enumerate(polylines):
        if pl.get('closed', False) or len(pl['path']) < 2:
            continue
        path = pl['path']
        # Head tangent: points outward from path[0]
        x0, y0 = path[0]
        x1, y1 = path[min(len(path) - 1, tangent_k - 1)]
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6: continue
        tips.append({'pl': pi, 'side': 'h', 'xy': (x0, y0),
                     'tan': ((x0 - x1) / L, (y0 - y1) / L)})  # outward
        # Tail tangent: points outward from path[-1]
        x0, y0 = path[-1]
        x1, y1 = path[max(0, len(path) - tangent_k)]
        L = math.hypot(x0 - x1, y0 - y1)
        if L < 1e-6: continue
        tips.append({'pl': pi, 'side': 't', 'xy': (x0, y0),
                     'tan': ((x0 - x1) / L, (y0 - y1) / L)})  # outward

    N = len(tips)
    if N < 2:
        return list(polylines)

    xy_arr = np.array([t['xy'] for t in tips], dtype=np.float64)

    # Phase 1 — build candidate list
    candidates = []   # (score, i, j)
    for i in range(N):
        ax, ay = tips[i]['xy']; tax, tay = tips[i]['tan']
        diffs  = xy_arr - xy_arr[i]
        d2     = (diffs ** 2).sum(axis=1)
        near   = np.where((d2 > 0) & (d2 <= gap2))[0]
        for j in near:
            if j <= i: continue
            if tips[i]['pl'] == tips[j]['pl']: continue   # don't loop self
            bx, by = tips[j]['xy']; tbx, tby = tips[j]['tan']
            dx, dy = bx - ax, by - ay
            d = math.sqrt(dx * dx + dy * dy)
            ux, uy = dx / d, dy / d
            dot_a = max(-1.0, min(1.0, tax * ux + tay * uy))
            dot_b = max(-1.0, min(1.0, tbx * (-ux) + tby * (-uy)))
            alpha = math.acos(dot_a); beta = math.acos(dot_b)
            straight_ok = dot_a >= cos_thr and dot_b >= cos_thr
            arc_ok      = (abs(alpha - beta) <= co_circ_r
                           and dot_a >= cos_2thr and dot_b >= cos_2thr)
            if not (straight_ok or arc_ok):
                continue
            # Soft-map support along straight bridge path
            n_s = max(3, int(d) + 1); ok = True
            for ks in range(n_s):
                t  = ks / (n_s - 1)
                sx = int(round(ax + t * dx)); sy = int(round(ay + t * dy))
                if 0 <= sy < H and 0 <= sx < W:
                    if soft_f32[sy, sx] < soft_th:
                        ok = False; break
            if not ok: continue
            score = (dot_a + dot_b) * 0.5 - d / max_gap
            candidates.append((score, i, j))

    candidates.sort(reverse=True)

    # Phase 2 — exclusive greedy commit, each tip used at most once.
    consumed_tip = [False] * N
    consumed_pl  = [False] * len(polylines)
    # Map polyline index → working path (mutable copies)
    work = [list(pl['path']) for pl in polylines]
    closed_flag = [bool(pl.get('closed', False)) for pl in polylines]
    # Map tip index → which polyline + side it currently refers to.  After
    # a merge both tips point to the same merged polyline; we resolve at use.
    tip_pl   = [t['pl']    for t in tips]
    tip_side = [t['side']  for t in tips]
    bridge_pixels = []   # collect for debug; not currently used

    def _bridge_path(p1, p2):
        """Bresenham line between p1 and p2, exclusive of endpoints."""
        x0, y0 = p1; x1, y1 = p2
        pixels = []
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        while True:
            if (cx, cy) != p1 and (cx, cy) != p2:
                pixels.append((cx, cy))
            if cx == x1 and cy == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; cx += sx
            if e2 <= dx: err += dx; cy += sy
        return pixels

    for (_, i, j) in candidates:
        if consumed_tip[i] or consumed_tip[j]: continue
        pi, si = tip_pl[i], tip_side[i]
        pj, sj = tip_pl[j], tip_side[j]
        if pi == pj: continue                           # already merged
        if consumed_pl[pi] or consumed_pl[pj]: continue

        path_i = work[pi]; path_j = work[pj]
        # Orient path_i so its joining tip is at the END
        if si == 'h':  path_i = list(reversed(path_i))
        # Orient path_j so its joining tip is at the START
        if sj == 't':  path_j = list(reversed(path_j))

        bridge = _bridge_path(path_i[-1], path_j[0])
        merged = path_i + bridge + path_j
        bridge_pixels.extend(bridge)

        # Commit: pj's slot becomes the merged polyline; pi gets retired.
        # (We pick pj arbitrarily; could also pick pi.)
        work[pj] = merged
        consumed_pl[pi] = True

        # Update the two non-merging tips of pi/pj to now refer to pj as
        # their polyline.  After orientation: the surviving tips are the
        # OTHER sides of pi and pj.
        # The merged polyline's head is pi's far tip and its tail is pj's
        # far tip.
        for k in range(N):
            if consumed_tip[k]: continue
            if tip_pl[k] == pi:
                tip_pl[k]   = pj
                # pi was reversed iff si == 'h'.  Far tip of original pi:
                #   if si == 'h': original head is now the merged tail's far
                #                  predecessor → far tip is the original
                #                  TAIL of pi, which is now at merged head.
                # Simpler: after merge, tips of pi (other than the consumed
                # one) all map to merged HEAD ('h').
                tip_side[k] = 'h'
            elif tip_pl[k] == pj:
                # The OTHER tip of pj (not the consumed one) is the merged
                # TAIL ('t').
                tip_side[k] = 't'
        consumed_tip[i] = True
        consumed_tip[j] = True

    return [{'path': work[i], 'closed': closed_flag[i]}
            for i in range(len(work)) if not consumed_pl[i]]


def _visvalingam_whyatt(pts, area_threshold):
    """Iteratively remove the interior point whose triangle (with its two
    immediate neighbours) has the smallest area, until every remaining
    triangle has area >= area_threshold.  Naive O(N²); fine for ≤ a few
    hundred points per polyline.

    Preserves the first and last point.  pts is a list of (x, y); returns
    the same."""
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


def _paths_to_polys(polylines, frame_w, frame_h, min_pts, simplify_area,
                     min_len, edge_soft_f32=None, spread=1.0,
                     bgr_frame=None, mask=None,
                     color_step=10, color_patch=5, debug_prefix=None):
    """HED equivalent of `_contours_to_polys`.  Takes pixel-path polylines
    {'path': [(x,y),...], 'closed': bool}, applies V-W simplification,
    samples per-vertex intensity from the float soft map, samples per-vertex
    color via Option D, normalizes to [-1, 1] with y-up, returns the JSON
    dict format."""
    out = []
    if edge_soft_f32 is not None:
        H, W = edge_soft_f32.shape[:2]
    for pl in polylines:
        pts = pl['path']
        if simplify_area > 0 and len(pts) >= 3:
            pts = _visvalingam_whyatt(pts, simplify_area)
        if len(pts) < min_pts:
            continue
        if min_len > 0:
            arc = sum(math.hypot(pts[i+1][0] - pts[i][0],
                                 pts[i+1][1] - pts[i][1])
                      for i in range(len(pts) - 1))
            if arc < min_len:
                continue
        # Per-vertex intensity from the float soft map directly
        if edge_soft_f32 is None:
            intensities = [0.7] * len(pts)
        else:
            intensities = []
            for (x, y) in pts:
                xi = max(0, min(W - 1, int(x)))
                yi = max(0, min(H - 1, int(y)))
                intensities.append(float(edge_soft_f32[yi, xi]))
            if spread != 1.0:
                m = sum(intensities) / len(intensities)
                intensities = [max(0.0, min(1.0, m + (v - m) * spread))
                               for v in intensities]
        # Per-vertex color (Option D)
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
                    'closed': bool(pl.get('closed', False))})

    if debug_prefix:
        _render_polys_to_png(out, frame_w, frame_h,
                             f'{debug_prefix}_06_polys.png')
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
            intensities = [0.7] * len(pts)
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
                     bgr_frame=None, original_bgr=None,
                     color_step=10, color_patch=5,
                     debug_prefix=None):
    """
    Interior edge extraction from a float [0,1] edge-probability map.
    Used by EDTER and DiffusionEdge.  HED has its own dedicated pipeline
    in `interior_steger_graph()` (sub-pixel ridge tracking; not routed through here).
    EDTER also routes through `interior_steger_graph()` now; only DiffusionEdge
    still uses this function.

    Common skeleton: erode mask -> [SEAM: map -> binary] -> AND with mask
    -> findContours -> polylines.  Only the seam varies per model.

    preprocess='diffusion_edge':
        Hard threshold only.  DiffusionEdge already outputs near-binary
        single-pixel ridges, so no extra thinning is needed.

    preprocess='edter':
        NMS on the soft float map (before any threshold).  EDTER outputs
        moderately wide soft ridges; NMS collapses them to 1-pixel-wide
        skeletons while preserving the float intensity at the ridge centre.
        A single threshold then acts as a noise-floor kill.
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
    if preprocess == 'edter':
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

    # Dilate the binary edge map slightly to preserve edges near the mask boundary
    # that might otherwise be cut off by the bitwise_and with the eroded inner mask.
    #dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #binary = cv2.dilate(binary, dilate_kernel, iterations=1)

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


def interior_steger_graph(mask_bool, soft_f32, frame_w, frame_h,
                          sigma, strength_th, step_angle,
                          min_len, simplify_area, min_pts, spread,
                          link_kernel, link_angle, link_tangent_k,
                          link_soft_th, link_co_circ,
                          use_gradient_mask=True,
                          mask_erode_size=9,
                          mask_dilate_size=9,
                          mask_gain_high=1.5, mask_gain_mid=1.0, mask_gain_low=0.5,
                          bgr_frame=None, bgr_frame_for_mask=None,
                          color_step=10, color_patch=5,
                          debug_prefix=None):
    """Soft-edge-map vectorization via Steger sub-pixel ridge tracking.

    Generic backend — used by both HED (use_gradient_mask=True, default
    mask erosion) and EDTER (use_gradient_mask=False, no mask erosion).
    Was previously named interior_hed_graph().

    Pipeline:
      [A] Optional gradient-mask filter (Han & Zhong 2020) sharpens the
          soft probability map by multiplying it with bins of image-gradient
          magnitude.  Only used for HED — EDTER's soft map is already
          crisp and gets fed straight into Steger.
      [B] Steger (1998) ridge-point detection.  Per-pixel Hessian
          eigendecomposition; pixels containing a sub-pixel ridge crest
          (perpendicular to the larger-eigenvalue eigenvector) survive.
          Operates directly on the soft map — no binarisation, no NMS,
          no skeletonisation — so there are no thinning artefacts.
      [C] Tangent-following linker walks ridge pixels in the +tangent
          then -tangent direction, producing one continuous polyline
          per chain.
      [D] Optional polyline-tip directional linking for any remaining
          gaps (tangent-aligned only, never sideways).
      [E] Visvalingam-Whyatt simplification + min-length filter.
      [F] Per-vertex intensity (sampled from raw soft map) + color.

    sigma             : Gaussian smoothing σ for Steger Hessian (≈ ridge half-width).
    strength_th       : Min ridge strength (-λ_perpendicular) to count as a ridge.
    step_angle        : Max angle deviation (degrees) between consecutive
                        ridge tangents during Steger linking.
    use_gradient_mask : If True, apply Han & Zhong gradient-mask sharpening
                        before Steger (HED).  If False, skip stage [A] and
                        feed soft_f32 straight into Steger (EDTER).
    mask_erode_size   : Kernel size (px) for SAM2 mask erosion before it
                        bounds the Steger search.  When ≤ 1, skip erosion
                        entirely (use the raw mask).  EDTER traces silhouette-
                        boundary edges precisely; erosion kills them.
    bgr_frame_for_mask: Original BGR frame used by the gradient-mask filter
                        for image-gradient computation (HED only).

    Returns the same JSON polyline dict list as interior_edgemap().
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255

    # Steger input mask: dilated outward by 3σ (rounded to nearest odd int) so
    # the Gaussian derivative kernels at any true-subject-edge pixel see real HED
    # values rather than hard zeros.  Hard zeros inside the kernel support window
    # distort the Hessian eigenvectors and kill outer-silhouette ridge detections.
    steger_dil_k = mask_dilate_size if mask_dilate_size % 2 == 1 else mask_dilate_size + 1
    steger_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (steger_dil_k, steger_dil_k))
    steger_mask_u8 = cv2.dilate(mask_u8, steger_kernel, iterations=1)
    steger_bool = steger_mask_u8 > 0

    # Post-detection filter mask: original mask optionally eroded, applied AFTER
    # Steger to reject detections that landed in the dilated fringe / background.
    if mask_erode_size > 1:
        f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (mask_erode_size, mask_erode_size))
        filter_u8 = cv2.erode(mask_u8, f_kernel, iterations=1)
        if cv2.countNonZero(filter_u8) == 0:
            return []
        filter_bool = filter_u8 > 0
    else:
        filter_bool = mask_bool

    # [A] Optional gradient-mask filter — sharpen wide HED ridges (skipped for EDTER)
    if use_gradient_mask:
        source = bgr_frame_for_mask if bgr_frame_for_mask is not None else bgr_frame
        if source is None:
            source = np.stack((mask_bool.astype(np.uint8) * 255,) * 3, axis=-1)
        refined = _refine_hed_map_with_gradient_mask(
            source, soft_f32.astype(np.float32),
            gain_high=mask_gain_high, gain_mid=mask_gain_mid, gain_low=mask_gain_low)
        if debug_prefix:
            cv2.imwrite(f'{debug_prefix}_02_gradient_mask.png',
                        (refined * 255).astype(np.uint8))
    else:
        refined = soft_f32.astype(np.float32)

    # [B] Steger ridge-point detection — runs on the dilated mask so Hessian is clean
    H, W = refined.shape
    masked_soft = refined * steger_bool.astype(np.float32)
    ys, xs, sub_y, sub_x, tx, ty, strength = _steger_ridge_points(
        masked_soft, sigma=sigma, strength_th=strength_th)

    # [B2] Filter detections back to the original / lightly-eroded mask.
    # Removes background specks that landed in the dilated fringe.
    if len(ys) > 0:
        keep = filter_bool[ys, xs]
        ys       = ys[keep];     xs       = xs[keep]
        sub_y    = sub_y[keep];  sub_x    = sub_x[keep]
        tx       = tx[keep];     ty       = ty[keep]
        strength = strength[keep]

    if debug_prefix:
        canvas = np.zeros((H, W), dtype=np.uint8)
        if len(ys) > 0:
            # Render ridge strength (clipped) as the debug image
            s_max = float(strength.max()) if strength.size else 1.0
            v = np.clip(strength / max(s_max, 1e-6), 0.0, 1.0)
            canvas[ys, xs] = (v * 255).astype(np.uint8)
        cv2.imwrite(f'{debug_prefix}_03_steger_ridges.png', canvas)

    if len(ys) == 0:
        return []

    # [C] Steger linker — tangent-following walk
    polylines = _steger_link_polylines(ys, xs, sub_y, sub_x, tx, ty, strength,
                                       H, W, max_step_angle_deg=step_angle)
    if debug_prefix:
        canvas = np.zeros((H, W), dtype=np.uint8)
        for pl in polylines:
            if len(pl['path']) < 2: continue
            pts = np.array(pl['path'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, 255, 1)
        cv2.imwrite(f'{debug_prefix}_04_steger_polylines.png', canvas)

    # [D] Optional polyline-tip directional linking (gap-bridging safety net)
    if link_kernel > 1 and len(polylines) >= 2:
        # _link_polyline_endpoints expects integer pixel paths; round here.
        for pl in polylines:
            pl['path'] = [(int(round(x)), int(round(y))) for (x, y) in pl['path']]
        polylines = _link_polyline_endpoints(
            polylines, soft_f32, link_kernel, link_angle,
            tangent_k=link_tangent_k, soft_th=link_soft_th,
            co_circ_deg=link_co_circ)
    if debug_prefix:
        canvas = np.zeros((H, W), dtype=np.uint8)
        for pl in polylines:
            if len(pl['path']) < 2: continue
            pts = np.array(pl['path'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, 255, 1)
        cv2.imwrite(f'{debug_prefix}_05_linked.png', canvas)

    # [E + F] Simplify, min-len filter, sample intensity & color, normalize
    return _paths_to_polys(polylines, frame_w, frame_h, min_pts, simplify_area,
                            min_len, edge_soft_f32=soft_f32, spread=spread,
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


def _make_video_writer(path, width, height, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video writer: {path}')
    return writer


def _edge_map_to_bgr(edge_map):
    if edge_map is None:
        return None
    gray = np.clip(edge_map, 0.0, 1.0)
    gray_u8 = (gray * 255).astype(np.uint8)
    if gray_u8.ndim == 2:
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if gray_u8.shape[2] == 1:
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return gray_u8


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
    args.diffusion_edge_epsilon  = args.diffusion_edge_epsilon  * _diag

    print(f'[vectorize] Masks: {total_frames} frames  ({frame_w}x{frame_h})'
          f'  diagonal={_diag:.0f}px  '
          f'hed_simplify_area={args.hed_simplify_area:.2f}px²', flush=True)

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
    edter_runner          = _try_load('edter',          args.edter_model,          EDTERRunner,
                                      stage=args.edter_stage,
                                      tile_blend=args.edter_tile_blend)
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    edge_video_writers = {}
    if args.edge_video:
        base_path = args.edge_video
        as_dir = False
        if base_path.endswith(os.sep) or base_path.endswith('/') or os.path.isdir(base_path):
            as_dir = True
            os.makedirs(base_path, exist_ok=True)
        else:
            parent_dir = os.path.dirname(base_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        for m in methods:
            if as_dir:
                path = os.path.join(base_path, f'{args.video}_{m}_edgemap.mp4')
            else:
                base, ext = os.path.splitext(base_path)
                if ext.lower() == '.mp4' and len(methods) > 1:
                    path = base + f'_{m}_edgemap.mp4'
                elif ext.lower() == '.mp4':
                    path = base_path
                else:
                    path = base_path + f'_{m}_edgemap.mp4'
            try:
                edge_video_writers[m] = _make_video_writer(
                    path, frame_w, frame_h,
                    float(args.edge_video_fps) if args.edge_video_fps is not None else float(fps))
                print(f'[vectorize] Edge-map video enabled: {m} → {path}', flush=True)
            except Exception as e:
                print(f'[vectorize] Warning: failed to create edge-map writer for {m}: {e}', flush=True)
                edge_video_writers[m] = None

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

        if edge_video_writers:
            for method_name, writer in edge_video_writers.items():
                if writer is None:
                    continue
                if method_name == 'canny':
                    # Canny isn't run by a runner upstream — compute it inline,
                    # matching interior_canny()'s preprocessing exactly.
                    cimg = np.clip(gray, 0, 255).astype(np.uint8)
                    if args.canny_blur > 1:
                        ck = args.canny_blur | 1
                        cimg = cv2.GaussianBlur(cimg, (ck, ck), 0)
                    edge_map = cv2.Canny(cimg, args.canny_low,
                                         args.canny_high).astype(np.float32) / 255.0
                else:
                    edge_map = {
                        'hed': hed_map,
                        'edter': edter_map,
                        'diffusion_edge': diffusion_edge_map,
                    }.get(method_name)
                frame_img = _edge_map_to_bgr(edge_map)
                if frame_img is not None:
                    writer.write(frame_img)

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
                interior = interior_steger_graph(
                    mask, hed_map, frame_w, frame_h,
                    args.hed_sigma, args.hed_strength, args.hed_step_angle,
                    args.hed_min_len, args.hed_simplify_area,
                    args.hed_min_pts, args.hed_spread,
                    args.hed_link_kernel, args.hed_link_angle,
                    args.hed_link_tangent_k, args.hed_link_soft_th,
                    args.hed_link_co_circ,
                    use_gradient_mask=True,
                    mask_erode_size=args.hed_mask_erode,
                    mask_dilate_size=args.hed_mask_dilate,
                    mask_gain_high=args.hed_mask_gain_high,
                    mask_gain_mid=args.hed_mask_gain_mid,
                    mask_gain_low=args.hed_mask_gain_low,
                    bgr_frame=bgr_frame_arg,
                    bgr_frame_for_mask=bgr,
                    color_step=args.color_step,
                    color_patch=args.color_patch,
                    debug_prefix=dbg)
            elif m == 'edter' and edter_map is not None:
                interior = interior_steger_graph(
                    mask, edter_map, frame_w, frame_h,
                    args.edter_sigma, args.edter_strength, args.edter_step_angle,
                    args.edter_min_len, args.edter_simplify_area,
                    args.edter_min_pts, args.edter_spread,
                    args.edter_link_kernel, args.edter_link_angle,
                    args.edter_link_tangent_k, args.edter_link_soft_th,
                    args.edter_link_co_circ,
                    use_gradient_mask=True,
                    mask_erode_size=args.edter_mask_erode,
                    mask_dilate_size=args.edter_mask_dilate,
                    mask_gain_high=args.edter_mask_gain_high,
                    mask_gain_mid=args.edter_mask_gain_mid,
                    mask_gain_low=args.edter_mask_gain_low,
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
                    original_bgr=bgr,
                    color_step=args.color_step,
                    color_patch=args.color_patch,
                    debug_prefix=dbg)

            #frame_polys = outer + interior
            frame_polys = interior # Steger algorithm removes the need for SAM2 mask contour to be included in the final result
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
    for writer in edge_video_writers.values():
        if writer is not None:
            writer.release()

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
