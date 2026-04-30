#!/usr/bin/env python3
"""
Stage 3 — Temporal filter + ILDA encode
========================================
Apply temporal persistence filtering to polylines and write an ILDA file.

Reads
-----
  --polylines  resources/polylines/{stem}_sam2-{model}_{method}.json  (from vectorize.py)

Writes
------
  resources/animations/{stem}_sam2-{model}_{method}.ild

The ILDA file is immediately playable by laser_controller.

Usage
-----
  python encode.py --polylines <path.json> --output <path.ild>
                  [--persist-frames 2]
                  [--persist-dist 15]
"""

import argparse
import json
import os
import struct
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Polylines → ILDA encoder — Stage 3')
    p.add_argument('--polylines',      required=True,
                   help='Input .json polylines file (from vectorize.py)')
    p.add_argument('--output',         required=True,
                   help='Output .ild file path')
    p.add_argument('--persist-frames', type=int,   default=2,
                   help='Min consecutive frames a polyline must survive (1=off)')
    p.add_argument('--persist-dist',   type=int,   default=40,
                   help='Max centroid distance in pixels for frame-to-frame matching. '
                        'Outer contours are always kept regardless of this value.')
    return p.parse_args()


# =============================================================================
# ILDA Format 5 writer  (matches C++ ILDAFile.cpp exactly)
# =============================================================================

def _ilda_header(frame_idx, total_frames, num_points):
    hdr  = b'ILDA'
    hdr += b'\x00\x00\x00'
    hdr += struct.pack('B', 5)
    hdr += b'frame   '
    hdr += b'LZRCTRL '
    hdr += struct.pack('>HHH', num_points, frame_idx, total_frames)
    hdr += b'\x00\x00'
    return hdr


def _ilda_point(x, y, r, g, b, blank=False):
    xi = max(-32767, min(32767, int(x * 32767)))
    yi = max(-32767, min(32767, int(y * 32767)))
    return struct.pack('>hh', xi, yi) + struct.pack('BBBB',
                                                     0x40 if blank else 0x00,
                                                     b, g, r)


def write_ilda(path, animation):
    """
    animation: list of frames.
    Each frame is a list of polylines.
    Each polyline is a dict: {'pts': [[x, y], ...], 'closed': bool}
    Color is always white (255, 255, 255).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    total = len(animation)
    with open(path, 'wb') as f:
        for fi, frame_polys in enumerate(animation):
            pts = []
            for poly in frame_polys:
                points = poly['pts']
                closed = poly.get('closed', False)
                if len(points) < 2:
                    continue
                # For closed polylines, repeat the first point at the end
                draw_pts = points + [points[0]] if closed else points
                # Blank jump to start of polyline
                x, y = draw_pts[0]
                pts.append(_ilda_point(x, y, 255, 255, 255, blank=True))
                for x, y in draw_pts[1:]:
                    pts.append(_ilda_point(x, y, 255, 255, 255, blank=False))
            if not pts:
                continue
            f.write(_ilda_header(fi, total, len(pts)))
            for pt in pts:
                f.write(pt)
        f.write(_ilda_header(0, total, 0))   # EOF frame


# =============================================================================
# Temporal persistence filter
# =============================================================================

def _poly_centroid_px(pts, frame_w, frame_h):
    """Centroid in pixel coords from normalized [[x, y], ...] points."""
    xs = [(p[0] + 1.0) * 0.5 * frame_w for p in pts]
    ys = [(1.0 - p[1]) * 0.5 * frame_h for p in pts]
    return float(np.mean(xs)), float(np.mean(ys))


def temporal_persistence_filter(frames, frame_w, frame_h,
                                 persist_frames, persist_dist):
    """
    Keep a polyline only if a spatially close counterpart existed in each of
    the previous (persist_frames - 1) frames.

    persist_frames=1  → no filtering (everything passes)
    persist_frames=2  → must match previous frame
    persist_frames=3  → must match both previous two frames
    """
    if persist_frames <= 1:
        return frames

    n     = len(frames)
    dist2 = persist_dist ** 2

    # Precompute centroids for every polyline in every frame
    centroids = []
    for frame_polys in frames:
        centroids.append([
            _poly_centroid_px(poly['pts'], frame_w, frame_h)
            for poly in frame_polys
        ])

    result = [[] for _ in range(n)]
    for fi in range(n):
        required = persist_frames - 1
        for pi, poly in enumerate(frames[fi]):
            # Outer contours (subject silhouette) are always kept — their
            # centroid moves with the subject so the distance check would
            # incorrectly drop them on fast-moving frames.
            if poly.get('outer', False):
                result[fi].append(poly)
                continue

            cx, cy  = centroids[fi][pi]
            matched = 0
            for back in range(1, persist_frames):
                prev_fi = fi - back
                if prev_fi < 0:
                    break
                for pcx, pcy in centroids[prev_fi]:
                    if (pcx - cx) ** 2 + (pcy - cy) ** 2 <= dist2:
                        matched += 1
                        break
            if matched >= required:
                result[fi].append(poly)

    before = sum(len(f) for f in frames)
    after  = sum(len(f) for f in result)
    print(f'[encode] Persistence filter: {before} → {after} polylines '
          f'(removed {before - after})', flush=True)
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if not os.path.exists(args.polylines):
        print(f'[encode] Polylines file not found: {args.polylines}', flush=True)
        sys.exit(1)

    print(f'[encode] Loading {args.polylines} ...', flush=True)
    with open(args.polylines, 'r') as f:
        doc = json.load(f)

    meta   = doc['meta']
    frames = doc['frames']

    frame_w = meta['frame_w']
    frame_h = meta['frame_h']
    method  = meta.get('method', '?')

    print(f'[encode] {len(frames)} frames  method={method}  '
          f'({frame_w}x{frame_h})', flush=True)

    # Temporal persistence filter
    filtered = temporal_persistence_filter(
        frames, frame_w, frame_h,
        args.persist_frames, args.persist_dist)

    total_polys = sum(len(f) for f in filtered)
    print(f'[encode] Writing ILDA: {len(filtered)} frames  '
          f'{total_polys} total polylines', flush=True)

    write_ilda(args.output, filtered)

    size_kb = os.path.getsize(args.output) / 1024
    print(f'[encode] Saved: {args.output}  ({size_kb:.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
