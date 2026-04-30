#!/usr/bin/env python3
"""
Preview mask output from segment.py.

Overlays the SAM2 binary mask on the original video frames so you can verify
what the segmenter is actually tracking before spending time on vectorization.

Controls
--------
  Space       play / pause
  Right arrow next frame
  Left arrow  previous frame
  q / Esc     quit

Usage
-----
  python preview_masks.py --masks resources/masks/clip_sam2-tiny.npz
                          --video  path/to/clip.mp4          (optional overlay)
                          --fps    24                        (playback speed, default=source)
"""

import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Mask preview — Stage 1 output')
    p.add_argument('--masks', required=True,
                   help='Path to .npz mask file (from segment.py)')
    p.add_argument('--video', default=None,
                   help='Original video for background overlay (optional)')
    p.add_argument('--fps',   type=float, default=0,
                   help='Playback FPS (0 = use video FPS or 30)')
    return p.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load masks
    # -------------------------------------------------------------------------
    if not os.path.exists(args.masks):
        print(f'[preview] Masks file not found: {args.masks}', flush=True)
        sys.exit(1)

    data      = np.load(args.masks)
    masks     = data['masks']          # bool [N, H, W]
    frame_w   = int(data['frame_w'])
    frame_h   = int(data['frame_h'])
    n_frames  = masks.shape[0]
    print(f'[preview] {n_frames} frames  ({frame_w}x{frame_h})', flush=True)

    # -------------------------------------------------------------------------
    # Load video frames (optional)
    # -------------------------------------------------------------------------
    video_frames = []
    source_fps   = 30.0
    if args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while len(video_frames) < n_frames:
            ret, frm = cap.read()
            if not ret:
                break
            video_frames.append(frm)
        cap.release()
        print(f'[preview] Loaded {len(video_frames)} video frames '
              f'({source_fps:.1f} fps)', flush=True)
    else:
        if args.video:
            print(f'[preview] Video not found: {args.video} — showing mask only',
                  flush=True)

    fps = args.fps if args.fps > 0 else source_fps
    delay_ms = max(1, int(1000.0 / fps))

    # -------------------------------------------------------------------------
    # Render one frame → BGR image
    # -------------------------------------------------------------------------
    # Overlay colour: bright green
    MASK_COLOR = np.array([0, 220, 80], dtype=np.uint8)   # BGR
    ALPHA      = 0.45

    def render(fi):
        mask = masks[fi]   # bool [H, W]

        if fi < len(video_frames):
            bg = video_frames[fi].copy()
            bg = cv2.resize(bg, (frame_w, frame_h))
        else:
            bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Coloured mask overlay
        overlay       = bg.copy()
        overlay[mask] = (overlay[mask].astype(np.float32) * (1 - ALPHA)
                         + MASK_COLOR.astype(np.float32) * ALPHA).astype(np.uint8)

        # Mask outline
        mask_u8   = mask.astype(np.uint8) * 255
        cnts, _   = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 80), 2)

        # HUD
        coverage = mask.mean() * 100.0
        cv2.rectangle(overlay, (0, 0), (frame_w, 36), (0, 0, 0), -1)
        cv2.putText(overlay,
                    f'Frame {fi + 1}/{n_frames}   coverage={coverage:.1f}%   '
                    f'[space]=play  [<>]=step  [q]=quit',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
        return overlay

    # -------------------------------------------------------------------------
    # Interactive loop
    # -------------------------------------------------------------------------
    window = 'Mask Preview  [q=quit]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(frame_w, 1280), min(frame_h, 720))

    fi      = 0
    playing = False

    while True:
        cv2.imshow(window, render(fi))
        key = cv2.waitKey(delay_ms if playing else 50) & 0xFF

        if key == ord('q') or key == 27:          # q / Esc
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 83 or key == ord('d'):        # right arrow / d
            fi = min(fi + 1, n_frames - 1)
            playing = False
        elif key == 81 or key == ord('a'):        # left arrow / a
            fi = max(fi - 1, 0)
            playing = False
        elif cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

        if playing:
            fi += 1
            if fi >= n_frames:
                fi = 0   # loop

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
