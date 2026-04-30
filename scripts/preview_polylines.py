#!/usr/bin/env python3
"""
Preview polyline output from vectorize.py.

Renders normalized laser polylines on a black canvas so you can inspect the
paths before encoding to ILDA.  Outer contours (closed=True) are drawn in
white; interior paths (closed=False) are drawn in cyan so you can distinguish
the two types at a glance.

Controls
--------
  Space       play / pause
  Right arrow next frame
  Left arrow  previous frame
  o           toggle outer contours (closed)
  i           toggle interior paths (open)
  q / Esc     quit

Usage
-----
  python preview_polylines.py --polylines resources/polylines/clip_sam2-tiny_thin.json
                               --width 1280
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Polyline preview — Stage 2 output')
    p.add_argument('--polylines', required=True,
                   help='Path to .json polylines file (from vectorize.py)')
    p.add_argument('--width',  type=int, default=1280,
                   help='Canvas width in pixels (height auto-computed from video AR)')
    p.add_argument('--fps',    type=float, default=30,
                   help='Playback FPS')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.polylines):
        print(f'[preview] File not found: {args.polylines}', flush=True)
        sys.exit(1)

    print(f'[preview] Loading {args.polylines} ...', flush=True)
    with open(args.polylines, 'r') as f:
        doc = json.load(f)

    meta   = doc['meta']
    frames = doc['frames']

    frame_w = meta['frame_w']
    frame_h = meta['frame_h']
    method  = meta.get('method', '?')
    n_frames = len(frames)

    # Canvas size — preserve video aspect ratio
    canvas_w = args.width
    canvas_h = int(canvas_w * frame_h / frame_w)

    print(f'[preview] {n_frames} frames  method={method}  '
          f'canvas={canvas_w}x{canvas_h}', flush=True)

    delay_ms = max(1, int(1000.0 / args.fps))

    # -------------------------------------------------------------------------
    # Coordinate helpers
    # -------------------------------------------------------------------------
    def norm_to_px(x, y):
        """Laser normalized [-1,1] → canvas pixel coords."""
        px = int((x + 1.0) * 0.5 * canvas_w)
        py = int((1.0 - y) * 0.5 * canvas_h)   # Y-flip (laser Y-up → image Y-down)
        return px, py

    # -------------------------------------------------------------------------
    # Render one frame
    # -------------------------------------------------------------------------
    COLOR_OUTER    = (255, 255, 255)   # white  — closed outer contours
    COLOR_INTERIOR = (0,   220, 255)   # cyan   — open interior paths

    def render(fi, show_outer, show_interior):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        frame_polys = frames[fi] if fi < len(frames) else []
        n_outer = n_inner = 0

        for poly in frame_polys:
            pts    = poly['pts']
            closed = poly.get('closed', False)

            if closed and not show_outer:
                continue
            if not closed and not show_interior:
                continue

            color = COLOR_OUTER if closed else COLOR_INTERIOR
            px_pts = [norm_to_px(p[0], p[1]) for p in pts]

            if closed:
                n_outer += 1
                loop_pts = px_pts + [px_pts[0]]
                for a, b in zip(loop_pts, loop_pts[1:]):
                    cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
            else:
                n_inner += 1
                for a, b in zip(px_pts, px_pts[1:]):
                    cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)

            # Mark start point (blank jump destination)
            if px_pts:
                cv2.circle(canvas, px_pts[0], 3, (80, 80, 80), -1)

        # HUD
        outer_label = f'[o] outer={n_outer}' if show_outer  else '[o] outer=off'
        inner_label = f'[i] inner={n_inner}' if show_interior else '[i] inner=off'
        cv2.rectangle(canvas, (0, 0), (canvas_w, 36), (20, 20, 20), -1)
        cv2.putText(canvas,
                    f'Frame {fi + 1}/{n_frames}  {method}  '
                    f'{outer_label}  {inner_label}  [space]=play  [q]=quit',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)
        return canvas

    # -------------------------------------------------------------------------
    # Interactive loop
    # -------------------------------------------------------------------------
    window = f'Polyline Preview — {method}  [q=quit]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, canvas_w, canvas_h)

    fi             = 0
    playing        = False
    show_outer     = True
    show_interior  = True

    while True:
        cv2.imshow(window, render(fi, show_outer, show_interior))
        key = cv2.waitKey(delay_ms if playing else 50) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 83 or key == ord('d'):        # right / d
            fi = min(fi + 1, n_frames - 1)
            playing = False
        elif key == 81 or key == ord('a'):        # left / a
            fi = max(fi - 1, 0)
            playing = False
        elif key == ord('o'):
            show_outer = not show_outer
        elif key == ord('i'):
            show_interior = not show_interior
        elif cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

        if playing:
            fi += 1
            if fi >= n_frames:
                fi = 0

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
