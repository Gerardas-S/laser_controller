#!/usr/bin/env python3
"""
Experiment runner — full thesis matrix sweep + metrics aggregation.

Sweeps every cell of the 25-pipeline matrix across every clip in the
benchmark corpus, computes Chamfer / wIoU / Blanking Ratio metrics per
(pipeline, clip), and writes a CSV summary suitable for thesis §5.3.

Reads
-----
    --config benchmark_config.json   (clip paths, mask paths, GT paths)

Writes
------
    results/{pipeline_id}/{clip_stem}.json         — polyline output
    results/summary.csv                            — flat metrics table
    results/by_pipeline.csv                        — mean ± SD per pipeline
    results/by_clip.csv                            — mean ± SD per clip

Matrix
------
Stage 0 is fixed at 'clahe_bilateral' (per thesis-fairness §5.1.1).
For ablation, run with --include-stage0-none to add the `none` arm.

Usage
-----
    python experiment_runner.py --config benchmark_config.json \
        --output-dir results/ --device cuda

    python experiment_runner.py --config benchmark_config.json \
        --output-dir results/ --skip-iarussi    # skip the slow 55s/frame method

    python experiment_runner.py --config benchmark_config.json \
        --output-dir results/ --max-frames 30   # subclip mode for fast iteration
"""

import os, sys, platform, json, argparse, time, csv, traceback

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

import numpy as np
import cv2
import torch

from pipeline import run_pipeline
from metrics  import chamfer_distance, warped_iou, blanking_ratio


# =============================================================================
# Pipeline matrix
# =============================================================================

# Per the architectural rework plan: Stage 0 fixed at 'cb' (CLAHE+bilateral)
# for the headline benchmark.  Per-cell compatibility table excludes only
# cells that are either nonsensical (algorithm assumes input it cannot have)
# or astronomical (single-frame runtime measured in minutes).
#
# Exclusions (kept narrow per user policy):
#   - NMS + iarussi  — ASTRONOMICAL.  NMS produces 2000–5000 fragmented chains
#                      per 1080p frame; Iarussi's O(N²) energy hangs for
#                      10+ minutes/frame.  Empirically verified May 2026.
#   - Canny + nms / Canny + steger — NONSENSICAL.  Canny output is already a
#                      binary 1-px map; both methods assume a soft probability
#                      input.  Excluded structurally: `canny` only pairs with
#                      thin='none' in COMPATIBLE_THIN below.
#
# Everything else is kept, including some redundant cells (e.g. steger + dp
# and steger + iarussi reduce to steger + ge because Steger's graph has no
# junctions).  Redundancy is acceptable; the cells run fast and confirm the
# theoretical prediction empirically.

COMPATIBLE_THIN = {
    'canny': ['none'],
    'hed':   ['nms', 'zs', 'steger'],
    'nbed':  ['nms', 'zs', 'steger'],
    'de':    ['nms', 'zs', 'steger'],
}

# Per-thin allowed vectorization methods.  `dp` is the no-graph-traversal
# baseline; `ge` is greedy graph traversal; `iarussi` is SA-optimized global.
COMPATIBLE_VEC = {
    'none':   ['dp'],
    'nms':    ['dp', 'ge'],            # iarussi EXCLUDED — astronomical runtime
    'zs':     ['dp', 'ge', 'iarussi'],
    'steger': ['dp', 'ge', 'iarussi'],
}


def enumerate_pipelines(include_stage0_none: bool = False,
                        skip_iarussi: bool = False,
                        skip_diffusion: bool = False):
    """Yield (stage0, edge, thin, vec) tuples."""
    stage0_options = ['cb']
    if include_stage0_none:
        stage0_options.append('none')

    for stage0 in stage0_options:
        for edge, thins in COMPATIBLE_THIN.items():
            if skip_diffusion and edge == 'de':
                continue
            for thin in thins:
                for vec in COMPATIBLE_VEC[thin]:
                    if skip_iarussi and vec == 'iarussi':
                        continue
                    yield (stage0, edge, thin, vec)


def pipeline_id(stage0, edge, thin, vec) -> str:
    return f'{stage0[:4]}_{edge}_{thin}_{vec}'


# =============================================================================
# Benchmark config
# =============================================================================

def load_config(path: str) -> dict:
    """Load a benchmark configuration JSON.

    Schema:
    {
      "clips": [
        {"name": "ballet",   "video": "resources/ballet.mp4",
         "masks": "resources/masks/ballet_sam2-tiny.npz",
         "gt":    "resources/gt/ballet_edges.npz"     // optional, for Chamfer
        },
        ...
      ]
    }
    """
    with open(path) as f:
        return json.load(f)


def auto_config_from_resources(resources_dir: str = 'resources') -> dict:
    """Auto-build a config by scanning resources/ for matching video+mask pairs.
    Useful for quick iteration when no explicit config exists yet."""
    masks_dir = os.path.join(resources_dir, 'masks')
    clips = []
    for fn in os.listdir(masks_dir):
        if not fn.endswith('.npz'):
            continue
        stem = fn[:-len('.npz')]
        video_path = os.path.join(resources_dir, f'{stem}.mp4')
        if not os.path.exists(video_path):
            continue
        clips.append({
            'name':  stem,
            'video': video_path,
            'masks': os.path.join(masks_dir, fn),
            'gt':    None,
        })
    return {'clips': sorted(clips, key=lambda c: c['name'])}


# =============================================================================
# Per-cell evaluation
# =============================================================================

def load_video_gray(video_path: str, max_frames: int = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames) if frames else np.empty((0, 0, 0), dtype=np.uint8)


def load_gt(gt_path: str, max_frames: int = None) -> np.ndarray:
    """Load ground-truth edge masks from .npz.  None gracefully returns None."""
    if not gt_path or not os.path.exists(gt_path):
        return None
    data = np.load(gt_path)
    masks = data['masks'] if 'masks' in data else data[list(data.keys())[0]]
    if max_frames:
        masks = masks[:max_frames]
    return masks


def evaluate_cell(stage0, edge, thin, vec,
                   clip: dict, output_dir: str,
                   device: str, max_frames: int = None) -> dict:
    """Run one pipeline on one clip and compute metrics.  Returns a row dict
    ready for CSV append."""
    pid = pipeline_id(stage0, edge, thin, vec)
    out_json = os.path.join(output_dir, pid, f'{clip["name"]}.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    t0 = time.perf_counter()
    try:
        result = run_pipeline(
            clip['video'], clip['masks'],
            stage0, edge, thin, vec,
            device=device,
            use_color=False,        # metrics don't care about color
            spline_samples=1,
            debug_dir=None,
        )
    except Exception as e:
        return {
            'pipeline':   pid,
            'clip':       clip['name'],
            'status':     'ERROR',
            'error':      str(e).replace('\n', ' ')[:200],
            'elapsed_s':  time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0

    # Truncate to max_frames if requested
    frames = result['frames']
    if max_frames:
        frames = frames[:max_frames]
    fw, fh = result['meta']['frame_w'], result['meta']['frame_h']

    # Save JSON output
    with open(out_json, 'w') as f:
        json.dump({'meta': result['meta'], 'frames': frames},
                  f, separators=(',', ':'))

    # Compute metrics
    row = {
        'pipeline':  pid,
        'clip':      clip['name'],
        'status':    'OK',
        'elapsed_s': elapsed,
        'n_frames':  len(frames),
    }

    # Blanking ratio (no external deps)
    br = blanking_ratio(frames)
    row.update({
        'BR_mean':  br['mean'],
        'BR_std':   br['std'],
        'K_mean':   br['mean_K'],
        'K_std':    br['std_K'],
    })

    # wIoU (requires video frames)
    try:
        gray = load_video_gray(clip['video'], max_frames=len(frames))
        if len(gray) >= 2:
            wiou = warped_iou(frames, gray, fw, fh)
            row['wIoU_mean'] = wiou['mean']
            row['wIoU_std']  = wiou['std']
        else:
            row['wIoU_mean'] = float('nan')
            row['wIoU_std']  = 0.0
    except Exception as e:
        row['wIoU_mean'] = float('nan')
        row['wIoU_std']  = 0.0
        row['wIoU_err']  = str(e)[:100]

    # Chamfer (requires ground truth)
    gt = load_gt(clip.get('gt'), max_frames=len(frames))
    if gt is not None:
        try:
            cd = chamfer_distance(frames, gt, fw, fh)
            row['CD_mean'] = cd['mean']
            row['CD_std']  = cd['std']
        except Exception as e:
            row['CD_mean'] = float('nan')
            row['CD_std']  = 0.0
            row['CD_err']  = str(e)[:100]
    else:
        row['CD_mean'] = float('nan')
        row['CD_std']  = 0.0

    return row


# =============================================================================
# Aggregation
# =============================================================================

def write_summary_csv(rows: list, path: str):
    if not rows:
        return
    # Stable column order
    base_cols = ['pipeline', 'clip', 'status', 'n_frames', 'elapsed_s',
                  'BR_mean', 'BR_std', 'K_mean', 'K_std',
                  'wIoU_mean', 'wIoU_std',
                  'CD_mean', 'CD_std', 'error']
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    cols = base_cols + sorted(all_keys - set(base_cols))

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_by(rows: list, key: str, out_path: str):
    """Group rows by `key` and emit mean ± std for numeric metrics."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r.get('status') != 'OK':
            continue
        groups[r[key]].append(r)

    metric_cols = ['BR_mean', 'K_mean', 'wIoU_mean', 'CD_mean', 'elapsed_s']
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = [key] + [f'{c}_avg' for c in metric_cols] + [f'{c}_std' for c in metric_cols] + ['n']
        w.writerow(header)
        for k in sorted(groups):
            grp = groups[k]
            avgs = []
            stds = []
            for c in metric_cols:
                vals = np.array([r.get(c, np.nan) for r in grp], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    avgs.append(float(vals.mean()))
                    stds.append(float(vals.std()))
                else:
                    avgs.append(float('nan'))
                    stds.append(0.0)
            w.writerow([k] + avgs + stds + [len(grp)])


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Thesis-matrix experiment runner')
    p.add_argument('--config',     default=None,
                   help='Benchmark config JSON. If omitted, auto-built from resources/.')
    p.add_argument('--output-dir', default='results',
                   help='Where to write per-pipeline JSONs and CSV summaries.')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu',
                   choices=['cpu', 'cuda'])
    p.add_argument('--max-frames', type=int, default=None,
                   help='Limit frames per clip (default: full clip).  '
                        'Use --max-frames 30 for fast iteration.')
    p.add_argument('--include-stage0-none', action='store_true',
                   help='Add Stage 0 = none arm for ablation.')
    p.add_argument('--skip-iarussi',     action='store_true',
                   help='Skip Iarussi cells (saves ~55s/frame).')
    p.add_argument('--skip-diffusion',   action='store_true',
                   help='Skip DiffusionEdge cells (saves ~5s/frame).')
    p.add_argument('--dry-run',          action='store_true',
                   help='Print matrix size without running anything.')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config) if args.config else auto_config_from_resources()
    print(f'[exp] {len(cfg["clips"])} clips, '
          f'pipelines={sum(1 for _ in enumerate_pipelines(args.include_stage0_none, args.skip_iarussi, args.skip_diffusion))}',
          flush=True)

    if args.dry_run:
        print('Dry run — pipelines that WOULD be evaluated:')
        for tup in enumerate_pipelines(args.include_stage0_none, args.skip_iarussi, args.skip_diffusion):
            print(f'  {pipeline_id(*tup)}')
        print('Clips:')
        for c in cfg['clips']:
            print(f'  {c["name"]}  ({c["video"]})')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    pipelines = list(enumerate_pipelines(args.include_stage0_none, args.skip_iarussi, args.skip_diffusion))
    total_cells = len(pipelines) * len(cfg['clips'])
    cell = 0
    t_all_start = time.perf_counter()

    for tup in pipelines:
        for clip in cfg['clips']:
            cell += 1
            pid = pipeline_id(*tup)
            print(f'\n[exp] [{cell}/{total_cells}] {pid}  on  {clip["name"]}', flush=True)
            row = evaluate_cell(*tup, clip=clip,
                                 output_dir=args.output_dir,
                                 device=args.device,
                                 max_frames=args.max_frames)
            rows.append(row)
            # Write incrementally so a crash doesn't lose results
            write_summary_csv(rows, os.path.join(args.output_dir, 'summary.csv'))
            print(f'[exp]   status={row.get("status")}  '
                  f'K={row.get("K_mean", float("nan")):.1f}±{row.get("K_std", 0):.1f}  '
                  f'BR={row.get("BR_mean", float("nan")):.3f}  '
                  f'wIoU={row.get("wIoU_mean", float("nan")):.3f}  '
                  f'CD={row.get("CD_mean", float("nan")):.1f}  '
                  f'({row.get("elapsed_s", 0):.0f}s)',
                  flush=True)

    # Final aggregation
    aggregate_by(rows, 'pipeline', os.path.join(args.output_dir, 'by_pipeline.csv'))
    aggregate_by(rows, 'clip',     os.path.join(args.output_dir, 'by_clip.csv'))

    t_total = time.perf_counter() - t_all_start
    n_ok = sum(1 for r in rows if r.get('status') == 'OK')
    n_err = sum(1 for r in rows if r.get('status') == 'ERROR')
    print(f'\n[exp] Done.  {n_ok} OK, {n_err} ERROR.  Total time: {t_total/60:.1f} min', flush=True)
    print(f'[exp] Wrote summary to {args.output_dir}/summary.csv', flush=True)
    print(f'[exp] Wrote aggregates to {args.output_dir}/by_pipeline.csv, {args.output_dir}/by_clip.csv', flush=True)


if __name__ == '__main__':
    main()
