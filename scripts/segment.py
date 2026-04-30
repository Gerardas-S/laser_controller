#!/usr/bin/env python3
"""
Stage 1 — Segmentation
=======================
Run SAM2 on a video and save per-frame binary masks to disk.

Prompting
---------
  Without --prompt   Five fixed grid points (center + quadrant centers).
                     Fast but blind — works best when the subject fills the frame.

  With --prompt      Grounding DINO localises the subject on frame 0 from a text
                     query, returning a bounding box.  SAM2 receives the box as its
                     initial prompt instead of grid points.  Much more reliable on
                     real-world footage.  Requires the GroundingDINO checkpoint
                     (see --gdino-model).

Output
------
  resources/masks/{video_stem}_sam2-{model}.npz
    masks   : bool  [N_frames, H, W]  — union of all tracked objects per frame
    frame_w : int32
    frame_h : int32

Usage
-----
  # Blind (no text prompt)
  python segment.py --video clip.mp4 --output masks/clip_sam2-tiny.npz

  # Text-grounded via Grounding DINO
  python segment.py --video clip.mp4 --output masks/clip_sam2-tiny.npz
                    --prompt "dancer"
                    --gdino-model models/gdino/groundingdino_swint_ogc.pth
"""

import argparse
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import tempfile
import shutil

import numpy as np
import cv2


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='SAM2 video segmentation — Stage 1')
    # I/O
    p.add_argument('--video',           required=True,
                   help='Input video file')
    p.add_argument('--output',          required=True,
                   help='Output .npz mask file path')
    # SAM2
    p.add_argument('--model',           default='tiny',
                   choices=['tiny', 'small', 'base', 'large'],
                   help='SAM2 model size')
    p.add_argument('--checkpoint-dir',  default='models/sam2',
                   help='Directory containing SAM2 checkpoint files')
    p.add_argument('--device',          default='cpu',
                   choices=['cpu', 'cuda'])
    # Grounding DINO (optional)
    p.add_argument('--prompt',          default=None,
                   help='Text query for Grounding DINO, e.g. "person" or "dancer". '
                        'If omitted, falls back to blind 5-point grid prompting.')
    p.add_argument('--gdino-model',     default='models/gdino/groundingdino_swint_ogc.pth',
                   help='Path to GroundingDINO checkpoint (.pth)')
    p.add_argument('--gdino-config',    default=None,
                   help='Path to GroundingDINO config (.py).  Auto-detected if omitted.')
    p.add_argument('--box-threshold',   type=float, default=0.30,
                   help='GroundingDINO box confidence threshold [0,1]')
    p.add_argument('--text-threshold',  type=float, default=0.25,
                   help='GroundingDINO text confidence threshold [0,1]')
    return p.parse_args()


# =============================================================================
# Grounding DINO — run on a single BGR frame, return best bounding box
# =============================================================================

def gdino_locate(frame_bgr, prompt, gdino_model_path, gdino_config_path,
                 box_threshold, text_threshold, device):
    """
    Run Grounding DINO on one frame.

    Returns
    -------
    box : np.ndarray shape [4]  — [x1, y1, x2, y2] in pixel coords, or None
    confidence : float
    """
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino
        import torch
        # GDINO runs on one frame — CPU is fine and avoids CUDA version conflicts
        device = 'cpu'
    except ImportError:
        print('[segment] GroundingDINO not installed.', flush=True)
        print('[segment] pip install git+https://github.com/IDEA-Research/GroundingDINO.git',
              flush=True)
        return None, 0.0

    # Auto-detect config if not provided
    if gdino_config_path is None:
        pkg_dir = os.path.dirname(groundingdino.__file__)
        gdino_config_path = os.path.join(
            pkg_dir, 'config', 'GroundingDINO_SwinT_OGC.py')
        if not os.path.exists(gdino_config_path):
            # Fallback: look next to the checkpoint
            gdino_config_path = os.path.join(
                os.path.dirname(gdino_model_path),
                'GroundingDINO_SwinT_OGC.py')

    if not os.path.exists(gdino_model_path):
        print(f'[segment] GroundingDINO checkpoint not found: {gdino_model_path}',
              flush=True)
        print('[segment] Download from:  '
              'https://github.com/IDEA-Research/GroundingDINO/releases', flush=True)
        return None, 0.0

    if not os.path.exists(gdino_config_path):
        print(f'[segment] GroundingDINO config not found: {gdino_config_path}',
              flush=True)
        return None, 0.0

    print(f'[segment] Loading GroundingDINO from {gdino_model_path} ...', flush=True)
    model = load_model(gdino_config_path, gdino_model_path, device=device)

    h, w = frame_bgr.shape[:2]
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # groundingdino.predict expects a PIL image or torch tensor
    from PIL import Image as PILImage
    from groundingdino.util.inference import predict
    from groundingdino.util import box_ops
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_img   = PILImage.fromarray(rgb)
    img_tensor = transform(pil_img).to(device)   # 3D (C,H,W) — predict() adds batch dim internally

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=img_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    if boxes is None or len(boxes) == 0:
        print(f'[segment] GroundingDINO: no boxes found for "{prompt}"', flush=True)
        return None, 0.0

    # boxes are cx,cy,w,h normalised [0,1] — convert to pixel xyxy
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * \
                 torch.tensor([w, h, w, h], dtype=torch.float32)
    scores     = logits.squeeze(-1) if logits.dim() > 1 else logits

    best_idx   = scores.argmax().item()
    best_box   = boxes_xyxy[best_idx].cpu().numpy()   # [x1,y1,x2,y2] pixels
    best_score = float(scores[best_idx])

    print(f'[segment] GroundingDINO: "{prompt}"  '
          f'box=[{best_box[0]:.0f},{best_box[1]:.0f},'
          f'{best_box[2]:.0f},{best_box[3]:.0f}]  '
          f'conf={best_score:.3f}  '
          f'({len(boxes)} candidate(s))', flush=True)

    return best_box.astype(np.float32), best_score


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f'[segment] Error: video not found: {args.video}', flush=True)
        sys.exit(1)

    try:
        from sam2.build_sam import build_sam2_video_predictor
        import torch
    except ImportError:
        print('[segment] sam2 not installed.', flush=True)
        print('[segment] pip install git+https://github.com/facebookresearch/sam2.git',
              flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # SAM2 checkpoint
    # -------------------------------------------------------------------------
    configs = {
        'tiny':  ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml'),
        'small': ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml'),
        'base':  ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
        'large': ('sam2.1_hiera_large.pt',     'configs/sam2.1/sam2.1_hiera_l.yaml'),
    }
    ckpt_file, cfg_file = configs[args.model]
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)

    if not os.path.exists(ckpt_path):
        print(f'[segment] SAM2 checkpoint not found: {ckpt_path}', flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Extract frames
    # -------------------------------------------------------------------------
    print(f'[segment] Extracting frames from {args.video} ...', flush=True)
    tmp_dir = tempfile.mkdtemp(prefix='sam2_frames_')

    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f'[segment] Cannot open video: {args.video}', flush=True)
            sys.exit(1)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame0_bgr = None
        fi = 0
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            if fi == 0:
                frame0_bgr = frm.copy()
            cv2.imwrite(os.path.join(tmp_dir, f'{fi:05d}.jpg'), frm,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            fi += 1
        cap.release()
        total_frames = fi
        print(f'[segment] {total_frames} frames  ({frame_w}x{frame_h})', flush=True)

        if total_frames == 0:
            print('[segment] Error: no frames extracted', flush=True)
            sys.exit(1)

        # -------------------------------------------------------------------------
        # Determine SAM2 prompt
        # -------------------------------------------------------------------------
        use_box    = False
        prompt_box = None   # [x1, y1, x2, y2] pixels

        if args.prompt:
            print(f'[segment] Text prompt: "{args.prompt}"  '
                  f'— running Grounding DINO on frame 0 ...', flush=True)
            box, conf = gdino_locate(
                frame0_bgr, args.prompt,
                args.gdino_model, args.gdino_config,
                args.box_threshold, args.text_threshold,
                args.device)
            if box is not None:
                use_box    = True
                prompt_box = box
            else:
                print('[segment] Grounding DINO returned no box — '
                      'falling back to 5-point grid prompt.', flush=True)

        if not use_box:
            cx, cy = frame_w // 2, frame_h // 2
            prompt_points = np.array([
                [cx,               cy              ],
                [frame_w // 4,     frame_h // 4    ],
                [3 * frame_w // 4, frame_h // 4    ],
                [frame_w // 4,     3 * frame_h // 4],
                [3 * frame_w // 4, 3 * frame_h // 4],
            ], dtype=np.float32)
            prompt_labels = np.ones(len(prompt_points), dtype=np.int32)
            print(f'[segment] Using 5-point grid prompt (blind)', flush=True)

        # -------------------------------------------------------------------------
        # SAM2 propagation
        # -------------------------------------------------------------------------
        print(f'[segment] Loading SAM2-{args.model} on {args.device} ...', flush=True)
        predictor = build_sam2_video_predictor(cfg_file, ckpt_path, device=args.device)

        all_masks = np.zeros((total_frames, frame_h, frame_w), dtype=np.bool_)

        import torch
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)

            if use_box:
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    box=prompt_box)
            else:
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    points=prompt_points, labels=prompt_labels)

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks  = (mask_logits > 0.0).squeeze(1).cpu().numpy()
                merged = masks.any(axis=0)
                all_masks[frame_idx] = merged

                if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
                    pct = merged.mean() * 100.0
                    print(f'[segment] {frame_idx + 1}/{total_frames}  '
                          f'coverage={pct:.1f}%', flush=True)

        # -------------------------------------------------------------------------
        # Save
        # -------------------------------------------------------------------------
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez_compressed(
            args.output,
            masks=all_masks,
            frame_w=np.int32(frame_w),
            frame_h=np.int32(frame_h),
        )
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f'[segment] Saved: {args.output}  ({size_mb:.1f} MB)', flush=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
