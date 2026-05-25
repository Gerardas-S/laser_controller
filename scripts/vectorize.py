#!/usr/bin/env python3
"""
Stage 2 — Vectorization (4-stage modular pipeline)
==================================================
Driver script that runs ONE pipeline specification (Stage 0 × Stage 1 ×
Stage 2 × Stage 3) on a video + SAM2 mask file and writes the result as
JSON polylines.

For sweeping the full thesis matrix (24 pipelines × 10 videos), use
`experiment_runner.py`.

Pipeline architecture
---------------------
    Stage 0 — preprocess  (clahe_bilateral | none)
    Stage 1 — edge        (canny | hed | nbed | diffusion_edge)
    Stage 2 — thinning    (none | nms | zhang_suen | steger)
    Stage 3 — vectorize   (dp_vw | greedy_eulerian | iarussi)
    Stage 4 — postprocess (always: V-W simplify, intensity sample,
                            color sample, [-1,1] normalize, optional spline)

All per-method parameters are LOCKED to canonical defaults in
`pipeline/defaults.py` — the thesis-fairness constraint requires that no
stage be tuned per-experiment.  Only pipeline selection, I/O, and the
always-shared knobs (device, color on/off, spline samples) are CLI-exposed.

Reads
-----
    --video   original video file
    --masks   resources/masks/{stem}_sam2-{model}.npz  (from segment.py)

Writes
------
    --output  .json  (see below for schema)

JSON schema
-----------
    {
      "meta": {
        "video":    "clip.mp4",
        "masks":    "clip_sam2-tiny.npz",
        "pipeline": {"stage0": "...", "edge": "...", "thin": "...", "vec": "..."},
        "frame_w":  1920,
        "frame_h":  1080,
        "total_frames": 300
      },
      "frames": [
        [
          {"pts": [[x, y], ...],
           "intensities": [...],
           "colors": [[r, g, b], ...],
           "closed": true,
           "outer": true},
          ...
        ],
        ...
      ]
    }

Coordinates are normalized to [-1, 1] with Y-up (laser convention).
Per-vertex intensities ∈ [0, 1] sampled from the soft edge map.
Per-vertex colors ∈ [0, 1] sampled via Option D (step perpendicular into the
SAM2 mask).  When --no-color is set, every vertex is white.
"""

import os, sys, platform

# Windows c10 DLL preload (mirror of edge_models.py logic — must run BEFORE
# any torch import).
if platform.system() == 'Windows':
    import ctypes
    from importlib.util import find_spec
    try:
        spec = find_spec('torch')
        if spec and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), 'lib', 'c10.dll')
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch

from pipeline import run_pipeline


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='4-stage modular vectorization pipeline driver')

    # I/O
    p.add_argument('--video',  required=True, help='Original video file')
    p.add_argument('--masks',  required=True, help='Input .npz mask file (from segment.py)')
    p.add_argument('--output', required=True, help='Output .json path')

    # Pipeline selection (short tokens — see pipeline/registry.py for mapping)
    p.add_argument('--stage0', default='cb',
                   choices=['none', 'cb'],
                   help='Stage 0 — preprocessing.  cb = CLAHE + bilateral.')
    p.add_argument('--edge',   default='nbed',
                   choices=['canny', 'hed', 'nbed', 'de'],
                   help='Stage 1 — edge detection.  de = DiffusionEdge.')
    p.add_argument('--thin',   default='steger',
                   choices=['none', 'nms', 'zs', 'steger'],
                   help='Stage 2 — thinning.  zs = Zhang-Suen.')
    p.add_argument('--vec',    default='ge',
                   choices=['dp', 'ge', 'iarussi'],
                   help='Stage 3 — vectorization.  dp = per-chain (no traversal); '
                        'ge = greedy Eulerian; iarussi = SOTA but slow.')

    # Always-shared knobs
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   choices=['cpu', 'cuda'])
    p.add_argument('--no-color',        action='store_true',
                   help='Emit white vertices instead of sampling per-vertex BGR colors.')
    p.add_argument('--spline-samples',  type=int, default=1,
                   help='Catmull-Rom samples per segment (1=off, 4=smooth, 8=very smooth)')
    p.add_argument('--edge-video-fps',  type=float, default=None,
                   help='Frame rate for cached edgemap MP4s; defaults to source fps.')
    p.add_argument('--debug-dir',       default=None,
                   help='Write Stage 2/3 debug PNGs for frame 0 into this dir.')

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f'[vectorize] Video not found: {args.video}', flush=True)
        sys.exit(1)
    if not os.path.exists(args.masks):
        print(f'[vectorize] Masks not found: {args.masks}', flush=True)
        sys.exit(1)

    debug_dir = args.debug_dir
    if debug_dir is None:
        # Default debug dir: resources/debug/{output_stem}/
        # Keeps debug artefacts consolidated under one umbrella for easy
        # gitignore and quick visual inspection of frame 0 outputs.
        out_stem = os.path.splitext(os.path.basename(args.output))[0]
        debug_dir = os.path.join('resources', 'debug', out_stem)

    result = run_pipeline(
        args.video, args.masks,
        args.stage0, args.edge, args.thin, args.vec,
        device=args.device,
        use_color=not args.no_color,
        spline_samples=args.spline_samples,
        edge_video_fps=args.edge_video_fps,
        debug_dir=debug_dir,
    )

    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, separators=(',', ':'))
    size_kb = os.path.getsize(args.output) / 1024
    print(f'[vectorize] Saved: {args.output}  ({size_kb:.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
