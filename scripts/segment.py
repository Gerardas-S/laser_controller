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
    p.add_argument('--gdino-select',    default='union',
                   choices=['best', 'largest', 'union'],
                   help='How to select from multiple GDINO candidates: '
                        'best=highest confidence (old default), '
                        'largest=biggest bounding box area, '
                        'union=merge all candidates into one box (default).')
    # Interactive point selection (overrides --prompt)
    p.add_argument('--interactive',     action='store_true',
                   help='Show frame 0 in a window and collect SAM2 point prompts '
                        'by clicking.  Left=subject (green), right=background '
                        '(red), U=undo, Enter=confirm, Esc=cancel.  Takes '
                        'priority over --prompt.')
    return p.parse_args()


# =============================================================================
# Interactive point selection — show frame 0, collect SAM2 point prompts
# =============================================================================

def interactive_point_select(frame_bgr):
    """
    Open an OpenCV window on frame 0 and collect SAM2 point prompts.

    Controls
    --------
      Left  click : add subject  point (label 1, green)
      Right click : add background point (label 0, red)
      U / u       : undo last point
      Enter       : confirm and proceed
      Esc         : cancel (caller falls back to blind grid)

    Returns
    -------
    (points, labels) : (np.float32 [N, 2], np.int32 [N]) on success
    (None, None)     : on cancel or no positive points
    """
    points = []   # list of (x, y, label)  label: 1=subject  0=background
    win    = ('SAM2 point prompt - L:subject  R:background  '
              'U:undo  Enter:confirm  Esc:cancel')

    def _redraw():
        img = frame_bgr.copy()
        # Black instruction bar at the top
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(img,
                    'L=subject  R=background  U=undo  Enter=confirm  Esc=cancel',
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Bottom-left point counter
        n_pos = sum(1 for _, _, l in points if l == 1)
        n_neg = sum(1 for _, _, l in points if l == 0)
        cv2.putText(img, f'+{n_pos}  -{n_neg}',
                    (6, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        for (x, y, lbl) in points:
            color = (0, 220, 0) if lbl == 1 else (0, 0, 220)
            cv2.circle(img, (x, y), 7, color, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
        cv2.imshow(win, img)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y, 1))
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y, 0))
            _redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, _on_mouse)
    _redraw()

    cancelled = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):           # Enter / LF
            break
        if key == 27:                 # Esc
            cancelled = True
            break
        if key in (ord('u'), ord('U')):
            if points:
                points.pop()
                _redraw()

    cv2.destroyWindow(win)
    cv2.waitKey(1)   # let the window actually close on some platforms

    if cancelled or not points or not any(l == 1 for _, _, l in points):
        return None, None

    pts_arr = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
    lbl_arr = np.array([l for _, _, l in points], dtype=np.int32)
    return pts_arr, lbl_arr


# =============================================================================
# Grounding DINO — run on a single BGR frame, return best bounding box
# =============================================================================

def gdino_locate(frame_bgr, prompt, gdino_model_path, gdino_config_path,
                 box_threshold, text_threshold, device,
                 select='union', debug_path=None):
    """
    Run Grounding DINO on one frame.

    select     : 'best'    — highest-confidence candidate
                 'largest' — largest bounding-box area
                 'union'   — axis-aligned union of all candidates
    debug_path : if given, save an annotated JPEG showing all candidates.

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
    boxes_np   = boxes_xyxy.cpu().numpy()   # [N, 4]
    scores_np  = scores.cpu().numpy()

    # --- always print all candidates so user can see what was found ---
    print(f'[segment] GroundingDINO: {len(boxes_np)} candidate(s) for "{prompt}":',
          flush=True)
    for ci, (b, s) in enumerate(zip(boxes_np, scores_np)):
        print(f'  [{ci}]  box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]  '
              f'conf={float(s):.3f}  phrase="{phrases[ci] if ci < len(phrases) else "?"}"',
              flush=True)

    # --- box selection ---
    if select == 'union':
        chosen_box = np.array([
            boxes_np[:, 0].min(),
            boxes_np[:, 1].min(),
            boxes_np[:, 2].max(),
            boxes_np[:, 3].max(),
        ], dtype=np.float32)
        chosen_score = float(scores_np.max())
    elif select == 'largest':
        areas   = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        idx     = int(areas.argmax())
        chosen_box   = boxes_np[idx].astype(np.float32)
        chosen_score = float(scores_np[idx])
    else:  # 'best'
        idx          = int(scores_np.argmax())
        chosen_box   = boxes_np[idx].astype(np.float32)
        chosen_score = float(scores_np[idx])

    print(f'[segment] Selected ({select}): '
          f'box=[{chosen_box[0]:.0f},{chosen_box[1]:.0f},'
          f'{chosen_box[2]:.0f},{chosen_box[3]:.0f}]  '
          f'conf={chosen_score:.3f}', flush=True)

    # --- debug image: all candidates in blue, chosen box in green ---
    if debug_path is not None:
        dbg = frame_bgr.copy()
        for ci, (b, s) in enumerate(zip(boxes_np, scores_np)):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(dbg, f'{ci} {float(s):.2f}',
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 1,
                        cv2.LINE_AA)
        cx1, cy1 = int(chosen_box[0]), int(chosen_box[1])
        cx2, cy2 = int(chosen_box[2]), int(chosen_box[3])
        cv2.rectangle(dbg, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        cv2.putText(dbg, f'chosen ({select})',
                    (cx1, max(0, cy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                    cv2.LINE_AA)
        os.makedirs(os.path.dirname(os.path.abspath(debug_path)), exist_ok=True)
        cv2.imwrite(debug_path, dbg)
        print(f'[segment] GDINO debug image → {debug_path}', flush=True)

    return chosen_box, chosen_score


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
        use_box       = False
        use_points    = False
        prompt_box    = None   # [x1, y1, x2, y2] pixels
        prompt_points = None
        prompt_labels = None

        # Priority chain: --interactive > --prompt > blind 5-point grid
        if args.interactive:
            print('[segment] Interactive mode — click points on frame 0  '
                  '(L=subject  R=background  U=undo  Enter=confirm  Esc=cancel)',
                  flush=True)
            pts, lbls = interactive_point_select(frame0_bgr)
            if pts is not None:
                use_points    = True
                prompt_points = pts
                prompt_labels = lbls
                n_pos = int((lbls == 1).sum())
                n_neg = int((lbls == 0).sum())
                print(f'[segment] Interactive: collected {n_pos} positive + '
                      f'{n_neg} negative point(s)', flush=True)
            else:
                print('[segment] Interactive: cancelled or no positive points — '
                      'falling back to 5-point grid prompt.', flush=True)

        elif args.prompt:
            print(f'[segment] Text prompt: "{args.prompt}"  '
                  f'— running Grounding DINO on frame 0 ...', flush=True)
            debug_img = os.path.splitext(
                os.path.abspath(args.output))[0] + '_gdino_boxes.jpg'
            box, conf = gdino_locate(
                frame0_bgr, args.prompt,
                args.gdino_model, args.gdino_config,
                args.box_threshold, args.text_threshold,
                args.device,
                select=args.gdino_select,
                debug_path=debug_img)
            if box is not None:
                use_box    = True
                prompt_box = box
            else:
                print('[segment] Grounding DINO returned no box — '
                      'falling back to 5-point grid prompt.', flush=True)

        if not use_box and not use_points:
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
