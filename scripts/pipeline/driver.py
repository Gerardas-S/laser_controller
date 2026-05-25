"""
Pipeline driver — orchestrates the 4-stage pipeline for a single video.

This is the engine: given a video, a SAM2 mask set, and a 4-tuple of method
names (stage0, edge, thin, vec), produce a list of per-frame polyline lists
ready for vectorize.py to serialise.

Stage 0 ─ Frame  → Frame      (CLAHE+bilateral, or none)
Stage 1 ─ Frame  → EdgeMap    (4 methods)
Stage 2 ─ EdgeMap → Graph     (4 methods)
Stage 3 ─ Graph  → Polylines  (3 methods)
Stage 4 ─ Polylines → JSON    (V-W, intensity/color sample, splines)
"""

import os
import math
import time
import numpy as np
import cv2

from . import edge_detect, registry, postprocess
from .defaults import (
    PREPROCESS_CLAHE_BILATERAL,
    EDGE_CANNY_DEFAULTS, EDGE_HED_DEFAULTS,
    EDGE_NBED_DEFAULTS,  EDGE_DIFFUSION_DEFAULTS,
    THIN_NONE_DEFAULTS, THIN_NMS_DEFAULTS,
    THIN_ZHANG_SUEN_DEFAULTS, THIN_STEGER_DEFAULTS,
    VEC_DP_DEFAULTS, VEC_GREEDY_EULERIAN_DEFAULTS, VEC_IARUSSI_DEFAULTS,
    POSTPROCESS_DEFAULTS,
)


# =============================================================================
# Resolved configs per method name (keys match registry.* dispatch tables)
# =============================================================================

_EDGE_CFG = {
    'canny': EDGE_CANNY_DEFAULTS,
    'hed':   EDGE_HED_DEFAULTS,
    'nbed':  EDGE_NBED_DEFAULTS,
    'de':    EDGE_DIFFUSION_DEFAULTS,
}
_THIN_CFG = {
    'none':   THIN_NONE_DEFAULTS,
    'nms':    THIN_NMS_DEFAULTS,
    'zs':     THIN_ZHANG_SUEN_DEFAULTS,
    'steger': THIN_STEGER_DEFAULTS,
}
_VEC_CFG = {
    'dp':      VEC_DP_DEFAULTS,
    'ge':      VEC_GREEDY_EULERIAN_DEFAULTS,
    'iarussi': VEC_IARUSSI_DEFAULTS,
}


# =============================================================================
# Per-frame pipeline
# =============================================================================

def process_frame(bgr_raw: np.ndarray,
                   mask: np.ndarray,
                   stage0: str, edge: str, thin: str, vec: str,
                   *,
                   runner_cache: dict,
                   device: str,
                   frame_w: int, frame_h: int,
                   use_color: bool = True,
                   spline_samples: int = 1,
                   debug_prefix: str = None,
                   pre_computed_edge: np.ndarray = None,
                   postprocess_overrides: dict = None) -> list:
    """Run the 4-stage pipeline on one frame.  Returns the JSON-ready
    polyline list (outer SAM2 contour + interior polylines), spline-filtered."""

    # ---- Stage 0 ----
    _t0 = time.perf_counter()
    stage0_fn = registry.PREPROCESS[stage0]
    frame = stage0_fn(bgr_raw, PREPROCESS_CLAHE_BILATERAL if stage0 == 'cb' else None)
    _t1 = time.perf_counter()

    # ---- Stage 1 ----
    if pre_computed_edge is not None:
        edge_map = pre_computed_edge
        _t2 = _t1          # cache hit — no inference time
    else:
        edge_fn = registry.EDGE[edge]
        edge_map = edge_fn(frame, mask, _EDGE_CFG[edge], device, runner_cache)
        _t2 = time.perf_counter()
        # Stash the freshly-computed edge map so the cache-writing block in
        # run_pipeline() can write it without a second model inference.
        runner_cache['_last_edge_map'] = edge_map

    # ---- Stage 2 ----
    thin_fn = registry.THIN[thin]
    graph = thin_fn(edge_map, mask, _THIN_CFG[thin],
                    edge_method=edge, bgr_frame_for_mask=bgr_raw)
    _t3 = time.perf_counter()

    if debug_prefix:
        _save_thinning_debug(graph, frame_w, frame_h, edge_map, debug_prefix)

    # ---- Stage 3 ----
    vec_fn = registry.VEC[vec]
    internal_polylines = vec_fn(graph, _VEC_CFG[vec])
    _t4 = time.perf_counter()

    # ---- Stage 4 (interior) ----
    pp = dict(POSTPROCESS_DEFAULTS)
    # Pull per-method postprocess params (simplify_area, min_pts, min_len, spread)
    vc = _VEC_CFG[vec]
    pp_kwargs = {
        'simplify_area': vc.get('simplify_area', 0.0),
        'min_pts':       vc.get('min_pts', 2),
        'min_len':       vc.get('min_len', 0.0),
        'spread':        vc.get('spread', 1.0),
        'color_step':    pp['color_step'],
        'color_patch':   pp['color_patch'],
        'use_white':     not use_color,
    }
    if postprocess_overrides:
        pp_kwargs.update(postprocess_overrides)

    interior = postprocess.apply_postprocess(
        internal_polylines, frame_w, frame_h,
        soft_map=edge_map,
        bgr_frame=(None if not use_color else bgr_raw),
        mask=mask,
        **pp_kwargs,
    )

    # Outer SAM2 silhouette
    _diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    outer = postprocess.mask_outer_contours(
        mask, frame_w, frame_h,
        min_area_px=pp['outer_min_area_frac'] * frame_w * frame_h,
        epsilon=pp['outer_smooth_epsilon_frac'] * _diag,
        bgr_frame=(None if not use_color else bgr_raw),
        mask_for_color=mask,
        color_step=pp['color_step'],
        color_patch=pp['color_patch'],
        use_white=not use_color,
    )

    frame_polys = outer + interior
    if spline_samples > 1:
        frame_polys = postprocess.apply_spline_fitting(frame_polys, spline_samples)
    _t5 = time.perf_counter()

    if debug_prefix:
        postprocess.render_polys_to_png(frame_polys, frame_w, frame_h,
                                          f'{debug_prefix}_07_splined.png')

    # Stash per-stage timings (ms) for the progress print in run_pipeline().
    runner_cache['_last_frame_timings'] = {
        's0': (_t1 - _t0) * 1e3,
        's1': (_t2 - _t1) * 1e3,
        's2': (_t3 - _t2) * 1e3,
        's3': (_t4 - _t3) * 1e3,
        's4': (_t5 - _t4) * 1e3,
    }

    return frame_polys


# =============================================================================
# Full-video pipeline with edgemap caching
# =============================================================================

def run_pipeline(video_path: str,
                  masks_path: str,
                  stage0: str, edge: str, thin: str, vec: str,
                  *,
                  device: str = 'cpu',
                  use_color: bool = True,
                  spline_samples: int = 1,
                  edge_video_fps: float = None,
                  edgemap_dir: str = 'resources/cache',
                  debug_dir: str = None) -> dict:
    """Run the pipeline on a full video and return a result dict containing
    'meta' and 'frames' (list of per-frame polyline lists)."""
    registry.validate(stage0, edge, thin, vec)

    # Load masks
    data = np.load(masks_path)
    all_masks = data['masks']
    frame_w = int(data['frame_w'])
    frame_h = int(data['frame_h'])
    total_frames = all_masks.shape[0]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Edgemap cache
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = edge_detect.cache_path_for(video_stem, edge, edgemap_dir, stage0=stage0)
    cap_reader = edge_detect.open_cache_reader(cache_path)
    writer = None
    if cap_reader is None:
        writer = edge_detect.open_cache_writer(
            cache_path, frame_w, frame_h,
            float(edge_video_fps) if edge_video_fps is not None else float(fps))
        if writer is None:
            print(f'[pipeline] WARN: cannot open cache writer at {cache_path}', flush=True)

    runner_cache: dict = {}

    print(f'[pipeline] {video_stem}  pipeline=({stage0}, {edge}, {thin}, {vec})  '
          f'frames={total_frames}  size={frame_w}x{frame_h}  device={device}', flush=True)

    frames_out: list = []
    t_start = time.perf_counter()

    for fi in range(total_frames):
        ret, bgr = cap.read()
        if not ret:
            print(f'[pipeline] WARN: ran out of video frames at {fi}', flush=True)
            break
        mask = all_masks[fi]

        # Pull cached edgemap if available
        pre_edge = None
        if cap_reader is not None:
            r, cached = cap_reader.read()
            if r:
                pre_edge = edge_detect.bgr_to_edgemap(cached)

        dbg = None
        if debug_dir and fi == 0:
            os.makedirs(debug_dir, exist_ok=True)
            dbg = os.path.join(debug_dir, f'frame00_{stage0}_{edge}_{thin}_{vec}')

        polys = process_frame(
            bgr, mask,
            stage0, edge, thin, vec,
            runner_cache=runner_cache,
            device=device,
            frame_w=frame_w, frame_h=frame_h,
            use_color=use_color,
            spline_samples=spline_samples,
            debug_prefix=dbg,
            pre_computed_edge=pre_edge,
        )

        # If we computed (rather than read) the edge map, write it to the
        # edgemap cache.  process_frame() stashes the result in
        # runner_cache['_last_edge_map'] so we never re-run model inference.
        if writer is not None and pre_edge is None:
            em = runner_cache.pop('_last_edge_map', None)
            if em is not None:
                writer.write(edge_detect.edgemap_to_bgr(em))

        tm = runner_cache.pop('_last_frame_timings', {})
        frames_out.append(polys)

        if (fi + 1) % 10 == 0 or fi == total_frames - 1:
            n_poly  = len(polys)
            n_pts   = sum(len(p['pts']) for p in polys)
            elapsed = time.perf_counter() - t_start
            frame_ms = sum(tm.values())
            print(
                f'[pipeline] {fi+1}/{total_frames}  polys={n_poly}  pts={n_pts}  '
                f'elapsed={elapsed:.1f}s  '
                f'last={frame_ms:.0f}ms'
                f'  (s0={tm.get("s0",0):.0f}'
                f'  s1={tm.get("s1",0):.0f}'
                f'  s2={tm.get("s2",0):.0f}'
                f'  s3={tm.get("s3",0):.0f}'
                f'  s4={tm.get("s4",0):.0f})',
                flush=True,
            )

    cap.release()
    if cap_reader is not None:
        cap_reader.release()
    if writer is not None:
        writer.release()

    meta = {
        'video':        os.path.basename(video_path),
        'masks':        os.path.basename(masks_path),
        'pipeline':     {'stage0': stage0, 'edge': edge, 'thin': thin, 'vec': vec},
        'frame_w':      frame_w,
        'frame_h':      frame_h,
        'total_frames': len(frames_out),
    }
    return {'meta': meta, 'frames': frames_out}


# =============================================================================
# Debug rendering of Stage 2 graph output
# =============================================================================

def _save_thinning_debug(graph, frame_w, frame_h, edge_map, prefix):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    # raw edge map
    cv2.imwrite(f'{prefix}_01_edge.png',
                (np.clip(edge_map, 0, 1) * 255).astype(np.uint8))
    # chains rasterized
    canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for c in graph['chains']:
        pts = np.array(c['pixels'], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], bool(c.get('closed', False)), 255, 1)
    cv2.imwrite(f'{prefix}_02_thin_chains.png', canvas)
    # nodes overlay
    overlay = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for nid, (x, y) in graph.get('nodes', {}).items():
        cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imwrite(f'{prefix}_03_nodes.png', overlay)
