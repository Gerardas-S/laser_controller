"""
Stage 1 — Edge detection dispatcher.

Wraps the model runners in `edge_models.py` behind a uniform
`run_<method>(frame, mask, runner_cache, **kwargs) -> EdgeMap` interface so
the pipeline driver can dispatch over method name without per-method special
cases.

Runner loading
--------------
Model weights are heavy (DiffusionEdge ≈ 1.2 GB, NBED ≈ 360 MB), so runners
are cached in a per-process dict that the driver passes in.  First call loads;
subsequent calls reuse.

Edgemap caching
---------------
Inference output for a (video, method) pair is cached as an MP4 in
`resources/cache/{stem}_{method}_edgemap.mp4` so re-running the pipeline
with different Stage 2/3 choices is instant.  Caching is generic and lives
here (not duplicated per model).
"""

import os
import sys
import cv2
import numpy as np

# Make sure the parent `scripts/` is importable so we can pull in edge_models
# when this module is loaded as a sub-package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# =============================================================================
# Per-method runner factories
# =============================================================================
# Each factory constructs the runner lazily so missing model files / CUDA
# failures only affect the method that needs them.  The driver wraps them in
# a try/except and drops the method on failure (mirrors the legacy behavior).

def _make_canny(cfg, device):
    from edge_models import CannyRunner
    return CannyRunner(
        low    = cfg.get('low',    40),
        high   = cfg.get('high',   120),
        blur_k = cfg.get('blur_k', 3),
    )

def _make_hed(cfg, device):
    from edge_models import HEDRunner
    return HEDRunner(cfg['model_path'], device=device)

def _make_nbed(cfg, device):
    from edge_models import NBEDRunner
    return NBEDRunner(cfg['model_path'], device=device)

def _make_diffusion(cfg, device):
    from edge_models import DiffusionEdgeRunner
    return DiffusionEdgeRunner(
        cfg['model_path'],
        device=device,
        first_stage_path=cfg['first_stage_path'],
        config_path=cfg['config_path'],
        sampling_steps=cfg.get('sampling_steps'),
    )

_RUNNER_FACTORIES = {
    'canny': _make_canny,
    'hed':   _make_hed,
    'nbed':  _make_nbed,
    'de':    _make_diffusion,
}


def get_runner(method: str, cfg: dict, device: str, runner_cache: dict):
    """Lazy-load and cache the runner for `method`.

    Returns the runner, or raises whatever the constructor raises.  The driver
    is responsible for catching and removing the method from the live set.
    """
    if method in runner_cache:
        return runner_cache[method]
    runner = _RUNNER_FACTORIES[method](cfg, device)
    runner_cache[method] = runner
    return runner


# =============================================================================
# Per-method run_* entry points (registered in registry.EDGE)
# =============================================================================

def _run_generic(method: str, frame, mask, cfg, device, runner_cache):
    runner = get_runner(method, cfg, device, runner_cache)
    return runner.infer(frame, mask=mask)

def run_canny(frame, mask, cfg, device, runner_cache):
    return _run_generic('canny', frame, mask, cfg, device, runner_cache)

def run_hed(frame, mask, cfg, device, runner_cache):
    return _run_generic('hed', frame, mask, cfg, device, runner_cache)

def run_nbed(frame, mask, cfg, device, runner_cache):
    return _run_generic('nbed', frame, mask, cfg, device, runner_cache)

def run_diffusion_edge(frame, mask, cfg, device, runner_cache):
    return _run_generic('de', frame, mask, cfg, device, runner_cache)


# =============================================================================
# Edgemap cache helpers
# =============================================================================

def edgemap_to_bgr(edge_map: np.ndarray) -> np.ndarray:
    """Convert a float32 [H, W] in [0, 1] edge map to a uint8 BGR frame for
    VideoWriter consumption (writes grayscale-as-BGR)."""
    gray_u8 = np.clip(edge_map, 0.0, 1.0)
    gray_u8 = (gray_u8 * 255).astype(np.uint8)
    if gray_u8.ndim == 2:
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return gray_u8


def bgr_to_edgemap(bgr: np.ndarray) -> np.ndarray:
    """Inverse of edgemap_to_bgr: extract the soft map from a cached MP4 frame."""
    return bgr[:, :, 0].astype(np.float32) / 255.0


def cache_path_for(video_stem: str, method: str, edgemap_dir: str = 'resources/cache',
                   stage0: str = 'none') -> str:
    return os.path.join(edgemap_dir, f'{video_stem}_{stage0}_{method}_edgemap.mp4')


def open_cache_reader(cache_path: str):
    """Open a cached edgemap MP4 for reading.  Returns VideoCapture or None
    on miss / read failure."""
    if not os.path.exists(cache_path):
        return None
    cap = cv2.VideoCapture(cache_path)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def open_cache_writer(cache_path: str, w: int, h: int, fps: float):
    """Open a VideoWriter for caching a new edgemap.  Returns None on failure."""
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(cache_path, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        return None
    return writer
