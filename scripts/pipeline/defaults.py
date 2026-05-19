"""
Locked canonical defaults for every pipeline method.

The thesis-fairness constraint requires that NO stage be tuned per
experiment — every (Stage 0, Stage 1, Stage 2, Stage 3) combination uses
the same per-method parameters across all videos.  This module is the
single source of truth for those parameters.

Adding a new method
-------------------
Add a new top-level dict named *_DEFAULTS.  Don't fork values between
"production" and "experiment" presets — there is only one canonical
default per method.
"""

# =============================================================================
# Stage 0 — Preprocessing
# =============================================================================

PREPROCESS_NONE = {}

PREPROCESS_CLAHE_BILATERAL = {
    # CLAHE on LAB-L channel: enhances local contrast in 8x8 tiles, clipping
    # the histogram to prevent noise amplification (Pizer et al. 1987).
    'clahe_clip_limit':  2.0,
    'clahe_tile_grid':   (8, 8),
    # Bilateral filter: smooths homogeneous regions while preserving sharp
    # intensity discontinuities (Tomasi & Manduchi 1998).  d=9 is the kernel
    # diameter; sigma_color/space tune intensity vs spatial weighting.
    'bilateral_d':           9,
    'bilateral_sigma_color': 75.0,
    'bilateral_sigma_space': 75.0,
}


# =============================================================================
# Stage 1 — Edge detection
# =============================================================================

EDGE_CANNY_DEFAULTS = {
    'low':    40,
    'high':   120,
    'blur_k': 3,
}

EDGE_HED_DEFAULTS = {
    # No tunable runtime params — model is fully-conv, resizes internally.
    'model_path': 'bsds500',
}

EDGE_NBED_DEFAULTS = {
    'model_path': 'models/nbed/NBED-BIPED.pth',
}

EDGE_DIFFUSION_DEFAULTS = {
    'model_path':            'models/diffusion_edge/bsds.pt',
    'first_stage_path':      'models/diffusion_edge/first_stage_total_320.pt',
    'config_path':           'models/diffusion_edge/bsds_sample.yaml',
    'sampling_steps':        8,
}


# =============================================================================
# Stage 2 — Thinning
# =============================================================================

THIN_NONE_DEFAULTS = {
    # Pass-through for Canny: threshold its already-binary output at > 0,
    # build the chain graph directly.
    'threshold': 0.0,
}

THIN_NMS_DEFAULTS = {
    # Non-maximum suppression on a soft edge-probability map.  Smoothing
    # applied only for gradient direction computation, not for the suppression
    # comparison (preserves raw peak values).
    'smooth_sigma': 1.0,
    # After NMS, threshold to binary and skeletonize.
    'threshold':    0.05,
}

THIN_ZHANG_SUEN_DEFAULTS = {
    'threshold': 0.0,
    # Mask handling — Zhang-Suen operates on the eroded mask (default 9px
    # erosion via _erode_mask).
    'mask_erode_kernel': 9,
}

# Per-edge-model Zhang-Suen overrides (populated by calibrate_stage2.py).
# Each model's edge map has a different value distribution; the threshold
# must adapt to the noise floor and dynamic range of the incoming soft map.
THIN_ZS_HED_OVERRIDES   = {'threshold': 0.351}   # calibrated bike frame 15
THIN_ZS_NBED_OVERRIDES  = {'threshold': 0.437}   # calibrated bike frame 15
THIN_ZS_DE_OVERRIDES    = {}                      # DE calibration skipped (OOM at 1080p)
THIN_ZS_CANNY_OVERRIDES = {}                      # N/A — Canny normally pairs with thin=none

# Per-edge-model NMS overrides (same rationale as Zhang-Suen).
THIN_NMS_HED_OVERRIDES   = {'threshold': 0.175}  # calibrated bike frame 15
THIN_NMS_NBED_OVERRIDES  = {'threshold': 0.218}  # calibrated bike frame 15
THIN_NMS_DE_OVERRIDES    = {}                     # DE calibration skipped (OOM at 1080p)
THIN_NMS_CANNY_OVERRIDES = {}

THIN_STEGER_DEFAULTS = {
    # Gaussian smoothing σ for Steger Hessian (≈ ridge half-width).
    'sigma': 3.0,
    # Min ridge strength (-λ_perpendicular) to count as a ridge.
    'strength_th': 0.005,
    # Max angle (deg) between consecutive ridge-pixel tangents during linking.
    'step_angle_deg': 90.0,
    # SAM2 mask handling — dilate for clean Hessian, optional post-erode for
    # background-speck rejection.
    'mask_dilate_size':  9,
    'mask_erode_size':   0,
    # Optional Han & Zhong gradient-mask refinement (NBED's soft map is sharp
    # enough that this hurts; HED benefits).  Per-method toggle below.
    'use_gradient_mask': False,
    'mask_gain_high':    1.0,
    'mask_gain_mid':     1.0,
    'mask_gain_low':     1.0,
    # Optional polyline-tip directional gap linking (Steger usually gives
    # continuous ridges — 1 = disabled by default).
    'link_kernel':       1,
    'link_angle_deg':    30.0,
    'link_tangent_k':    12,
    'link_soft_th':      0.05,
    'link_co_circ_deg':  12.0,
}

# Per-edge-model Steger overrides (only fields that differ from the base).
# Stage 2 merges these on top of THIN_STEGER_DEFAULTS when the edge stage is
# known.  Keeps the dispatch generic but lets HED enable the gradient-mask
# filter while NBED keeps it off.
THIN_STEGER_HED_OVERRIDES = {
    'sigma':             4.0,    # calibrated bike frame 15: ridge half-width 6.43 px
    'use_gradient_mask': True,
    'mask_gain_high':    1.5,
    'mask_gain_mid':     1.0,
    'mask_gain_low':     1.0,
    'strength_th':       0.00351,  # calibrated bike frame 15
    'step_angle_deg':    30.0,
    'link_kernel':       7,
    'link_angle_deg':    20.0,
    'link_co_circ_deg':  16.0,
}
THIN_STEGER_NBED_OVERRIDES  = {'sigma': 1.91, 'strength_th': 0.00437}  # calibrated bike frame 15
THIN_STEGER_DE_OVERRIDES    = {}
THIN_STEGER_CANNY_OVERRIDES = {}


# =============================================================================
# Stage 3 — Vectorization
# =============================================================================

# Per-chain Douglas-Peucker / passthrough.  Despite the historical "DP" label,
# no DP step runs here — each chain is emitted as a polyline and Stage 4 (V-W)
# does all simplification uniformly across every vec method.  Kept as the
# "no graph traversal" baseline distinct from greedy_eulerian / iarussi.
VEC_DP_DEFAULTS = {
    # Visvalingam-Whyatt min effective triangle area (px²).  Applied in Stage 4.
    'simplify_area': 1.5,
    'min_pts':       3,
    'min_len':       40,
    # Per-polyline intensity contrast modulator.
    'spread':        0.5,
}

VEC_GREEDY_EULERIAN_DEFAULTS = {
    'junction_angle_deg': 30.0,
    'tangent_k':          5,
    'simplify_area':      1.5,
    'min_pts':            3,
    'min_len':            40,
    'spread':             0.5,
}

VEC_IARUSSI_DEFAULTS = {
    # Energy weights
    'w_K': 10.0,       # pen-up jumps (dominant)
    'w_T':  1.0,       # tangent smoothness at junctions
    'w_B':  0.1,       # length balance across colors
    # SA schedule
    'T0':            1.0,
    'T_min':         0.01,
    'cooling':       0.995,
    'n_proposals':   2000,
    'kick_threshold':  300,    # rejection-streak threshold for re-heat
    'kick_temperature': 0.3,
    'kick_duration':   200,
    'early_stop_rejections': 500,
    # Move mix (must sum to 1.0)
    'p_split':       0.4,
    'p_join':        0.4,
    'p_permute':     0.2,
    # Postprocess
    'simplify_area':      1.5,
    'min_pts':            3,
    'min_len':            40,
    'spread':             0.5,
    'tangent_k':          5,
}


# =============================================================================
# Stage 4 — Postprocess (shared by every pipeline)
# =============================================================================

POSTPROCESS_DEFAULTS = {
    'color_step':     10,
    'color_patch':    5,
    'spline_samples': 1,    # 1 = off; 4–8 for visible smoothing
    # Outer SAM2 contour
    'outer_min_area_frac': 0.001,
    'outer_smooth_epsilon_frac': 0.0004,
}
