#!/usr/bin/env python3
"""
merge_masks.py — OR two or more mask NPZ files into one
========================================================
All inputs must have the same frame count and resolution.

Usage
-----
    python scripts/merge_masks.py --inputs a.npz b.npz --output merged.npz
    python scripts/merge_masks.py --inputs a.npz b.npz c.npz --output merged.npz
"""

import argparse
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Merge SAM2 mask NPZ files by OR')
    p.add_argument('--inputs',  nargs='+', required=True, help='Two or more input .npz mask files')
    p.add_argument('--output',  required=True,            help='Output .npz mask file')
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.inputs) < 2:
        print('[merge_masks] Need at least two input files.', flush=True)
        sys.exit(1)

    merged = None
    frame_w = None
    frame_h = None

    for path in args.inputs:
        if not os.path.exists(path):
            print(f'[merge_masks] Not found: {path}', flush=True)
            sys.exit(1)

        data = np.load(path)
        masks = data['masks']          # bool [N, H, W]
        w     = int(data['frame_w'])
        h     = int(data['frame_h'])

        if merged is None:
            merged  = masks.copy()
            frame_w = w
            frame_h = h
            print(f'[merge_masks] Base: {os.path.basename(path)}  '
                  f'{masks.shape[0]} frames  {w}x{h}', flush=True)
        else:
            if masks.shape != merged.shape:
                print(f'[merge_masks] Shape mismatch: {masks.shape} vs {merged.shape} '
                      f'in {path}', flush=True)
                sys.exit(1)
            if w != frame_w or h != frame_h:
                print(f'[merge_masks] Resolution mismatch in {path}', flush=True)
                sys.exit(1)
            merged |= masks
            coverage = merged.any(axis=(1, 2)).sum()
            print(f'[merge_masks] + {os.path.basename(path)}  '
                  f'→ {coverage}/{merged.shape[0]} frames with mask', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        masks=merged,
        frame_w=np.int32(frame_w),
        frame_h=np.int32(frame_h),
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'[merge_masks] Saved: {args.output}  ({size_mb:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
