"""
Stage 0 — Video preprocessing.

Standardizes input frames so downstream edge detection is not biased by
per-clip lighting, contrast, or noise differences.  Per the thesis-fairness
constraint, the same preprocessing runs uniformly on every clip in the
24-pipeline benchmark — no per-video tuning.

Methods
-------
    none             — identity; for the "preprocessing-off" experimental arm.
    clahe_bilateral  — CLAHE (Pizer 1987) on LAB-L for local contrast,
                       then bilateral filter (Tomasi & Manduchi 1998) for
                       edge-preserving denoise.

All methods take and return a BGR uint8 frame of identical shape.
"""

import cv2
import numpy as np

from .defaults import PREPROCESS_CLAHE_BILATERAL


def preprocess_none(frame: np.ndarray, cfg: dict = None) -> np.ndarray:
    """Identity passthrough.  Returns the frame unchanged."""
    return frame


def preprocess_clahe_bilateral(frame: np.ndarray, cfg: dict = None) -> np.ndarray:
    """CLAHE on LAB-L + bilateral filter.

    LAB color space isolates luminance into a single channel (L) so contrast
    enhancement does not introduce hue shifts.  Bilateral filter is applied
    in BGR to smooth Bayer-pattern sensor noise while preserving the edges
    we are about to detect.
    """
    cfg = cfg or PREPROCESS_CLAHE_BILATERAL

    # CLAHE on the L channel of LAB.
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(
        clipLimit=cfg['clahe_clip_limit'],
        tileGridSize=tuple(cfg['clahe_tile_grid']),
    )
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Edge-preserving denoise.
    out = cv2.bilateralFilter(
        out,
        d=cfg['bilateral_d'],
        sigmaColor=cfg['bilateral_sigma_color'],
        sigmaSpace=cfg['bilateral_sigma_space'],
    )
    return out
