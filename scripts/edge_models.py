"""
Edge detection model runners
============================

Each class wraps an edge-detection model.  They all share the same interface
so vectorize.py can swap them transparently:

    runner   = SomeRunner(model_path, device='cpu' | 'cuda')
    edge_map = runner.infer(bgr_frame)          # float32 [H, W] in [0, 1]

The returned edge map is a per-pixel edge-likelihood array at the input
frame's original resolution.  Downstream, vectorize.py thresholds and
skeletonises it to produce polylines (see interior_hed in vectorize.py).

Adding a new model
------------------
Create a new class with __init__(model_path, device) and infer(bgr_frame).
Wire it into vectorize.py's --method choices and runner-loading block.
That's it.

Models present
--------------
    HEDRunner            — Holistically-Nested Edge Detection (Xie & Tu, 2015)       ONNX
    DepthRunner          — Depth Anything V2 (Yang et al., 2024)   *not edges*        ONNX
    PiDiNetRunner        — Pixel Difference Networks (Su et al., ICCV 2021)           ONNX  [PLACEHOLDER]
    TEEDRunner           — Tiny and Efficient Edge Detector (Soria et al., 2023)      ONNX  [PLACEHOLDER]
    EDTERRunner          — Edge Detection with Transformer (Pu et al., CVPR 2022)     PyTorch
    DiffusionEdgeRunner  — Diffusion-based Edge Detection (Ye et al., AAAI 2024)      PyTorch
"""

import sys as _sys
import os as _os

# ---------------------------------------------------------------------------
# Windows DLL search path fix (Python 3.8+)
#
# Python 3.8 stopped searching the process PATH for DLL loading; only dirs
# registered via os.add_dll_directory() are searched.  torch registers its
# own lib/ folder, but cudnn_cnn64_9.dll and friends depend on DLLs that
# live in the CUDA/cuDNN system directories (zlibwapi.dll being the most
# common missing piece).
#
# When Python is spawned as a subprocess from the C++ app, those system
# directories may not be in PATH at all (app launched from VS, shortcut,
# etc.) and our earlier PATH-scanning fix found nothing.
#
# Two-pronged fix:
#   1. Add ALL PATH entries (cheap, handles the case where they are in PATH).
#   2. Scan known NVIDIA/CUDA installation directories on the filesystem
#      directly, independent of PATH.  Covers the common case where the app
#      was launched without the CUDA environment variables set.
# ---------------------------------------------------------------------------
# os.add_dll_directory() returns a handle object whose __del__ calls
# RemoveDllDirectory — so if the return value is discarded the directory is
# removed immediately by CPython's reference-counting GC.
# Keep every handle in this module-level list so they stay alive for the
# lifetime of the process.
_dll_dir_handles: list = []

if _os.name == 'nt' and hasattr(_os, 'add_dll_directory'):
    import glob as _glob

    def _safe_add(d):
        try:
            _dll_dir_handles.append(_os.add_dll_directory(d))
        except OSError:
            pass

    # 1. Every directory currently in PATH
    for _d in _os.environ.get('PATH', '').split(';'):
        _d = _d.strip().strip('"')
        if _d and _os.path.isdir(_d):
            _safe_add(_d)

    # 2. Well-known NVIDIA / CUDA install locations on the filesystem
    for _pat in [
        # CUDA Toolkit (contains cudart64_12.dll, zlibwapi.dll, …)
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\libnvvp',
        # Standalone cuDNN installer (v8/v9 layout)
        r'C:\Program Files\NVIDIA\CUDNN\v*\bin\*',
        r'C:\Program Files\NVIDIA\CUDNN\v*\bin',
        # Older cuDNN layout (placed directly in CUDA bin)
        r'C:\tools\cuda\bin',
    ]:
        for _d in sorted(_glob.glob(_pat), reverse=True):  # newest version first
            if _os.path.isdir(_d):
                _safe_add(_d)

    # 3. torch\lib — bundles cudart, cuDNN *and* zlibwapi.dll which
    #    onnxruntime_providers_cuda.dll needs on Windows but that is not
    #    shipped with the CUDA Toolkit or the standalone cuDNN installer.
    #    Use find_spec to locate the wheel without actually importing torch.
    import importlib.util as _ilu
    _tspec = _ilu.find_spec('torch')
    _tlib  = (_os.path.join(_os.path.dirname(_tspec.origin), 'lib')
              if (_tspec and _tspec.origin) else None)
    if _tlib and _os.path.isdir(_tlib):
        _safe_add(_tlib)
    del _ilu, _tspec, _tlib

    # 4. nvidia pip wheels (nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12, …)
    #    When cuDNN is installed via pip rather than the system installer, its
    #    DLLs land in  site-packages\nvidia\<pkg>\bin\.
    for _sp_root in _sys.path:
        _nvidia_dir = _os.path.join(_sp_root, 'nvidia')
        if _os.path.isdir(_nvidia_dir):
            for _d in _glob.glob(_os.path.join(_nvidia_dir, '*', 'bin')):
                if _os.path.isdir(_d):
                    _safe_add(_d)
    del _sp_root, _nvidia_dir

    del _glob, _safe_add, _d, _pat

# Root of the cloned DiffusionEdge repo — injected into sys.path on first use.
_DIFF_EDGE_ROOT = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'libs', 'diffusion_edge')
)

import cv2
import numpy as np
import torch


# =============================================================================
# HED — Holistically-Nested Edge Detection (Xie & Tu, ICCV 2015)
# Native PyTorch implementation.
# Architecture: github.com/sniklaus/pytorch-hed
# Weights: http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch
# =============================================================================

_HED_WEIGHTS_URL = 'http://content.sniklaus.com/github/pytorch-hed/network-{name}.pytorch'


class _HEDNetwork(torch.nn.Module):
    """VGG16-based HED network.  Preprocessing (×255 − BGR means) is baked into
    forward() so callers only need to pass float32 [0, 1] BGR tensors."""

    def __init__(self):
        super().__init__()
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(3,   64,  3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(64,  64,  3, 1, 1), torch.nn.ReLU(inplace=False),
        )
        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64,  128, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU(inplace=False),
        )
        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(inplace=False),
        )
        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
        )
        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
        )
        self.netScoreOne = torch.nn.Conv2d(64,  1, 1, 1, 0)
        self.netScoreTwo = torch.nn.Conv2d(128, 1, 1, 1, 0)
        self.netScoreThr = torch.nn.Conv2d(256, 1, 1, 1, 0)
        self.netScoreFou = torch.nn.Conv2d(512, 1, 1, 1, 0)
        self.netScoreFiv = torch.nn.Conv2d(512, 1, 1, 1, 0)
        self.netCombine  = torch.nn.Sequential(
            torch.nn.Conv2d(5, 1, 1, 1, 0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # x: float32 [B, 3, H, W] in [0, 1], BGR channel order
        x = x * 255.0 - torch.tensor(
            [104.00698793, 116.66876762, 122.67891434],
            dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        v1 = self.netVggOne(x)
        v2 = self.netVggTwo(v1)
        v3 = self.netVggThr(v2)
        v4 = self.netVggFou(v3)
        v5 = self.netVggFiv(v4)
        hw = x.shape[2:]
        s1 = torch.nn.functional.interpolate(self.netScoreOne(v1), size=hw, mode='bilinear', align_corners=False)
        s2 = torch.nn.functional.interpolate(self.netScoreTwo(v2), size=hw, mode='bilinear', align_corners=False)
        s3 = torch.nn.functional.interpolate(self.netScoreThr(v3), size=hw, mode='bilinear', align_corners=False)
        s4 = torch.nn.functional.interpolate(self.netScoreFou(v4), size=hw, mode='bilinear', align_corners=False)
        s5 = torch.nn.functional.interpolate(self.netScoreFiv(v5), size=hw, mode='bilinear', align_corners=False)
        return self.netCombine(torch.cat([s1, s2, s3, s4, s5], dim=1))


class HEDRunner:
    INPUT_H, INPUT_W = 480, 480   # resize before inference; model is fully-conv

    def __init__(self, model_path='bsds500', device='cpu'):
        self.model = _HEDNetwork()
        # model_path: path to a local .pytorch weights file, or a model name
        # (e.g. 'bsds500') which is auto-downloaded once and cached by torch.hub.
        if _os.path.isfile(model_path):
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        else:
            state = torch.hub.load_state_dict_from_url(
                url=_HED_WEIGHTS_URL.format(name=model_path),
                file_name=f'hed-{model_path}',
                map_location='cpu')
        # Weights trained with DataParallel have a 'module.' prefix; remap to 'net'.
        state = {k.replace('module', 'net'): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        print(f'[hed] Loaded  model={model_path}  device={self.device}', flush=True)

    def infer(self, bgr_frame):
        h0, w0 = bgr_frame.shape[:2]
        rsz  = cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H))
        blob = torch.from_numpy(
            rsz.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(blob)          # [1, 1, H, W] in [0, 1]
        edge = out.squeeze().cpu().numpy()  # [H, W]
        return cv2.resize(edge, (w0, h0), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# PiDiNet — Pixel Difference Networks (Su et al., ICCV 2021)             [TODO]
# https://github.com/hellozhuo/pidinet
#
# Lightweight CNN (~0.7M params) using "pixel difference convolutions" that
# directly compute local gradient differences.  Matches HED quality at 1/20
# the parameter count and 2-3× the inference speed.  Three model sizes
# available: PiDiNet, PiDiNet-small, PiDiNet-tiny.
#
# To finish this runner:
#   1. Export the chosen variant to ONNX from the official repo and place
#      the file at  models/pidinet/model.onnx  (or wherever --pidinet-model
#      points).
#   2. Verify the input format used by that specific export — most PiDiNet
#      exports take BGR with HED-style mean subtraction at 512×512, but
#      some forks use ImageNet RGB normalisation.  Adjust __init__ and
#      infer() accordingly.
#   3. The output is typically a sigmoid edge probability at input
#      resolution; resize back to (w0, h0) as HED does.
# =============================================================================

class PiDiNetRunner:
    INPUT_H, INPUT_W = 512, 512
    # Default to HED-style BGR mean subtraction; switch to ImageNet RGB if
    # the chosen export expects that instead.
    MEAN_B, MEAN_G, MEAN_R = 104.00698793, 116.66876762, 122.67891434

    def __init__(self, model_path, device='cpu'):
        raise NotImplementedError(
            'PiDiNet runner not yet implemented — no PyTorch model available')

    def infer(self, bgr_frame):
        raise NotImplementedError('PiDiNet runner not yet implemented')


# =============================================================================
# TEED — Tiny and Efficient Edge Detector (Soria et al., WACV 2023)      [TODO]
# https://github.com/xavysp/TEED
#
# Extremely lightweight (~58K params).  Runs at 200+ FPS with ODS slightly
# better than HED on BSDS500.  Designed for sparse, clean edge maps —
# excellent fit for laser projection where the point budget is tight.
#
# To finish this runner:
#   1. Use the official ONNX export from the TEED repo (release page) or
#      run their export script.  Place at  models/teed/model.onnx.
#   2. TEED training resolution is 352×352 with ImageNet normalisation
#      (RGB, mean/std normalisation in [0,1]).  Verify against the export.
#   3. The model has multiple side outputs; the official inference script
#      uses the fused output (last in the list).  Mirror that here.
# =============================================================================

class TEEDRunner:
    INPUT_H, INPUT_W = 352, 352
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, device='cpu'):
        raise NotImplementedError(
            'TEED runner not yet implemented — no PyTorch model available')

    def infer(self, bgr_frame):
        raise NotImplementedError('TEED runner not yet implemented')


# =============================================================================
# EDTER — Edge Detection with Transformer (Pu et al., CVPR 2022)
# https://github.com/MengyangPu/EDTER
#
# Stage I: ViT-Large/16 backbone with Bidirectional Multi-Layer Aggregation
# (BIMLA) at transformer layers 5, 11, 17, 23.  Sliding-window inference at
# 320×320 (stride 280).  ODS 0.820 on BSDS500.
#
# Stage II adds a local ViT-Base/8 branch (160×160 crops) fused via SFT.
# When a Stage II checkpoint is supplied, this runner extracts the Stage I
# global_model sub-weights and runs them at Stage I quality.  Full Stage II
# two-branch inference is not implemented.
#
# Pure PyTorch — no mmcv dependency.  Uses models/edter/EDTER-BSDS-VOC-StageI.pth.
# =============================================================================

# ---------------------------------------------------------------------------
# Lazy builder: creates all EDTER nn.Module classes on first use so that
# torch/nn are not imported at module load time (matches the rest of this file).
# ---------------------------------------------------------------------------

_edter_stage1_cls = None  # cached once built


def _build_edter_stage1():
    """Instantiate _EDTER_Stage1; imports torch/nn on first call."""
    global _edter_stage1_cls
    if _edter_stage1_cls is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # ── ViT primitive blocks ──────────────────────────────────────────

        class _Attn(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim  = dim // num_heads
                self.scale     = self.head_dim ** -0.5
                self.qkv  = nn.Linear(dim, dim * 3)
                self.proj = nn.Linear(dim, dim)

            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                           self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                return self.proj(x)

        class _Mlp(nn.Module):
            def __init__(self, dim, ratio=4.):
                super().__init__()
                hid = int(dim * ratio)
                self.fc1 = nn.Linear(dim, hid)
                self.fc2 = nn.Linear(hid, dim)

            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x)))

        class _Block(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn  = _Attn(dim, num_heads)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp   = _Mlp(dim)

            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x

        # ── Patch embedding ───────────────────────────────────────────────

        class _PatchEmbed(nn.Module):
            """Key: patch_embed.proj.*"""
            def __init__(self, patch_size=16, in_ch=3, embed_dim=1024):
                super().__init__()
                self.proj = nn.Conv2d(in_ch, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

            def forward(self, x):
                return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, E]

        # ── BIMLA (Bidirectional Multi-Layer Aggregation) ─────────────────
        # Keys: mla_p{2..5}_1x1.{0,1}  mla_p{2..5}.{0,1}  mla_b{2..5}.{0,1}
        # Each entry is nn.Sequential(Conv2d, BN2d).

        class _MLA(nn.Module):
            def __init__(self, in_ch=1024, mla_ch=256):
                super().__init__()
                def _seq1x1():
                    return nn.Sequential(
                        nn.Conv2d(in_ch, mla_ch, 1, bias=False),
                        nn.BatchNorm2d(mla_ch),
                    )
                def _seq3x3():
                    return nn.Sequential(
                        nn.Conv2d(mla_ch, mla_ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(mla_ch),
                    )
                # 1×1 projections
                self.mla_p2_1x1 = _seq1x1()
                self.mla_p3_1x1 = _seq1x1()
                self.mla_p4_1x1 = _seq1x1()
                self.mla_p5_1x1 = _seq1x1()
                # Forward (top-down) path
                self.mla_p2 = _seq3x3()
                self.mla_p3 = _seq3x3()
                self.mla_p4 = _seq3x3()
                self.mla_p5 = _seq3x3()
                # Backward (bottom-up) path
                self.mla_b2 = _seq3x3()
                self.mla_b3 = _seq3x3()
                self.mla_b4 = _seq3x3()
                self.mla_b5 = _seq3x3()

            def forward(self, feats):
                # feats = [f2, f3, f4, f5] each [B, in_ch, H, W]
                # where f2=shallowest (layer 5), f5=deepest (layer 23)
                e2 = self.mla_p2_1x1(feats[0])
                e3 = self.mla_p3_1x1(feats[1])
                e4 = self.mla_p4_1x1(feats[2])
                e5 = self.mla_p5_1x1(feats[3])
                # Top-down forward path (p5 → p2)
                tp5 = self.mla_p5(e5)
                tp4 = self.mla_p4(e4 + tp5)
                tp3 = self.mla_p3(e3 + tp4)
                tp2 = self.mla_p2(e2 + tp3)
                # Bottom-up backward path (p2 → p5)
                bp2 = self.mla_b2(e2)
                bp3 = self.mla_b3(e3 + bp2)
                bp4 = self.mla_b4(e4 + bp3)
                bp5 = self.mla_b5(e5 + bp4)
                # Return 8 feature maps: forward [tp2..tp5] + backward [bp2..bp5]
                return [tp2, tp3, tp4, tp5, bp2, bp3, bp4, bp5]

        # ── VIT-BIMLA backbone ────────────────────────────────────────────
        # Keys: cls_token, pos_embed, patch_embed.*, blocks.N.*, norm_{0..3}.*, mla.*

        class _VIT_BIMLA(nn.Module):
            # ViT-Large/16 with BIMLA, tapped at transformer layers 5,11,17,23
            MLA_IDX = (5, 11, 17, 23)

            def __init__(self, img_size=320, patch_size=16, embed_dim=1024,
                         num_heads=16, depth=24, mla_ch=256):
                super().__init__()
                n_patches = (img_size // patch_size) ** 2
                self.patch_embed = _PatchEmbed(patch_size, 3, embed_dim)
                self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed   = nn.Parameter(
                    torch.zeros(1, n_patches + 1, embed_dim))
                self.blocks = nn.ModuleList(
                    [_Block(embed_dim, num_heads) for _ in range(depth)])
                for i in range(4):
                    setattr(self, f'norm_{i}', nn.LayerNorm(embed_dim))
                self.mla = _MLA(embed_dim, mla_ch)

            def forward(self, x):
                B, _, H, W = x.shape
                x = self.patch_embed(x)
                cls = self.cls_token.expand(B, -1, -1)
                x   = torch.cat([cls, x], dim=1)
                x   = x + self.pos_embed

                feats = []
                ni = 0
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    if i in self.MLA_IDX:
                        norm = getattr(self, f'norm_{ni}')
                        feats.append(norm(x))
                        ni += 1

                # Remove cls token and reshape to [B, embed_dim, ph, pw]
                ph, pw = H // 16, W // 16
                spatial = []
                for f in feats:
                    s = f[:, 1:, :].permute(0, 2, 1).reshape(B, -1, ph, pw)
                    spatial.append(s)
                return self.mla(spatial)  # list of 8 × [B, 256, ph, pw]

        # ── VIT-BIMLAHead decode head ─────────────────────────────────────
        # Keys: mlahead.head{2..5}{,_1}.{0,1,3,4}, global_features.*, edge.*, conv_seg.*

        def _mla_branch(in_ch=256, out_ch=128):
            # Two-step transposed conv: 20→80 (×4) then 80→320 (×4)
            # k=4, s=4, p=0 for step1; k=16, s=4, p=6 for step2
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch,  out_ch, 4,  stride=4, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_ch, out_ch, 16, stride=4, padding=6, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        class _MLAHead(nn.Module):
            def __init__(self, mla_ch=256, out_ch=128):
                super().__init__()
                # forward path (top-down): head2..head5
                self.head2   = _mla_branch(mla_ch, out_ch)
                self.head3   = _mla_branch(mla_ch, out_ch)
                self.head4   = _mla_branch(mla_ch, out_ch)
                self.head5   = _mla_branch(mla_ch, out_ch)
                # backward path (bottom-up): head2_1..head5_1
                self.head2_1 = _mla_branch(mla_ch, out_ch)
                self.head3_1 = _mla_branch(mla_ch, out_ch)
                self.head4_1 = _mla_branch(mla_ch, out_ch)
                self.head5_1 = _mla_branch(mla_ch, out_ch)

            def forward(self, mla_feats):
                # mla_feats from MLA: [tp2, tp3, tp4, tp5, bp2, bp3, bp4, bp5]
                tp2, tp3, tp4, tp5, bp2, bp3, bp4, bp5 = mla_feats
                return [
                    self.head2(tp2),   self.head3(tp3),
                    self.head4(tp4),   self.head5(tp5),
                    self.head2_1(bp2), self.head3_1(bp3),
                    self.head4_1(bp4), self.head5_1(bp5),
                ]

        class _VIT_BIMLAHead(nn.Module):
            def __init__(self, mla_ch=256, out_ch=128):
                super().__init__()
                self.mlahead = _MLAHead(mla_ch, out_ch)
                # Fusion convs: 8 × out_ch → out_ch → 1
                # Indices 0,1,[2],3,4,[5],6,7,[8],9,10
                # (ReLU at 2,5,8 have no params)
                self.global_features = nn.Sequential(
                    nn.Conv2d(out_ch * 8, out_ch, 3, padding=1, bias=True),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 1, bias=True),
                    nn.BatchNorm2d(out_ch),
                )
                self.edge    = nn.Conv2d(out_ch,     1, 1)
                # Training-only auxiliary head (conv_seg takes 4 forward heads × out_ch)
                self.conv_seg = nn.Conv2d(out_ch * 4, 1, 1)

            def forward(self, mla_feats):
                heads = self.mlahead(mla_feats)      # 8 × [B, 128, H, W]
                fused = torch.cat(heads, dim=1)      # [B, 1024, H, W]
                gf    = self.global_features(fused)  # [B, 128, H, W]
                gf    = F.relu(gf, inplace=True)
                return torch.sigmoid(self.edge(gf))  # [B, 1, H, W]

        # ── Top-level Stage I model ───────────────────────────────────────

        class _EDTER_Stage1(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone    = _VIT_BIMLA()
                self.decode_head = _VIT_BIMLAHead()

            def forward(self, x):
                return self.decode_head(self.backbone(x))

        _edter_stage1_cls = _EDTER_Stage1

    return _edter_stage1_cls()


class EDTERRunner:
    """
    EDTER Stage I — pure PyTorch, no mmcv.

    Accepts either an EDTER-BSDS-VOC-StageI.pth or StageII.pth checkpoint.
    When a Stage II file is supplied, the global_model sub-weights are
    extracted and run at Stage I quality (full Stage II two-branch inference
    is not implemented).

    Inference: sliding window 320×320, stride 280, ImageNet normalisation.
    """
    CROP   = 320
    STRIDE = 280
    MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, device='cpu'):
        import torch

        def _build(dev):
            model = _build_edter_stage1()

            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('state_dict', ckpt)

            # Stage II checkpoint: weights live under 'global_model.*'
            if any(k.startswith('global_model.') for k in sd):
                sd = {k[len('global_model.'):]: v
                      for k, v in sd.items() if k.startswith('global_model.')}
                print('[edter] Stage II checkpoint — using global_model (Stage I quality)',
                      flush=True)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            # Filter to non-auxiliary missing keys (auxiliary_head is training-only)
            real_missing = [k for k in missing
                            if not k.startswith('auxiliary_head')
                            and not k.startswith('decode_head.conv_seg')]
            if real_missing:
                print(f'[edter] Warning: {len(real_missing)} unexpected missing keys: '
                      f'{real_missing[:5]}', flush=True)

            model.eval()
            model.to(torch.device(dev))
            # The decode_head BatchNorm running stats in this checkpoint are
            # accumulated on a distribution that does not match single-crop
            # inference.  Keeping decode_head BN in train mode lets it use
            # actual batch statistics from each 320×320 crop, which produces
            # correct edge probabilities.  The backbone BN running stats are
            # fine in eval mode.  torch.no_grad() in infer() prevents any
            # weight updates; only the BN running stats will drift slightly
            # toward the inference distribution, which is harmless.
            for m in model.decode_head.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.train()
            return model, torch.device(dev)

        def _is_cuda_err(e):
            s = str(e).lower()
            return '127' in str(e) or 'cuda' in s or 'cudnn' in s

        try:
            self._model, self._device = _build(device)
        except Exception as e:
            if device != 'cpu' and _is_cuda_err(e):
                print(f'[edter] CUDA failed ({e}), retrying on CPU', flush=True)
                self._model, self._device = _build('cpu')
            else:
                raise

        self._torch = torch
        print(f'[edter] Loaded {model_path}  device={self._device}', flush=True)

    def infer(self, bgr_frame):
        import torch.nn.functional as F
        import time as _time
        torch  = self._torch
        h0, w0 = bgr_frame.shape[:2]

        if not getattr(self, '_infer_diag_done', False):
            param_dev = next(self._model.parameters()).device
            print(f'[edter] infer diag: self._device={self._device}  '
                  f'model param device={param_dev}', flush=True)
            self._infer_diag_done = True
        _t0 = _time.perf_counter()

        # BGR → RGB, normalise to ImageNet
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD        # mean/std are broadcast over HxWx3

        # Pad if smaller than one crop
        ph = max(0, self.CROP - h0)
        pw = max(0, self.CROP - w0)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        H, W = rgb.shape[:2]

        inp = torch.from_numpy(
            rgb.transpose(2, 0, 1)).unsqueeze(0).to(self._device)   # [1, 3, H, W]

        h_grids = max(H - self.CROP + self.STRIDE - 1, 0) // self.STRIDE + 1
        w_grids = max(W - self.CROP + self.STRIDE - 1, 0) // self.STRIDE + 1
        preds   = inp.new_zeros((1, 1, H, W))
        cnt_mat = inp.new_zeros((1, 1, H, W))
        print(f'[edter] tiles={h_grids}×{w_grids}={h_grids*w_grids}  '
              f'frame={w0}×{h0}', flush=True)

        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * self.STRIDE;        x1 = wi * self.STRIDE
                y2 = min(y1 + self.CROP, H);  x2 = min(x1 + self.CROP, W)
                y1 = max(y2 - self.CROP, 0);  x1 = max(x2 - self.CROP, 0)
                crop = inp[:, :, y1:y2, x1:x2]          # always [1, 3, 320, 320]
                _tc = _time.perf_counter()
                with torch.no_grad():
                    out = self._model(crop)              # [1, 1, 320, 320]
                if hi == 0 and wi == 0:
                    print(f'[edter] first tile: {(_time.perf_counter()-_tc)*1000:.0f} ms', flush=True)
                preds  [:, :, y1:y2, x1:x2] += out
                cnt_mat[:, :, y1:y2, x1:x2] += 1.0

        edge = (preds / cnt_mat.clamp(min=1e-6)).squeeze().cpu().numpy()
        elapsed = _time.perf_counter() - _t0
        print(f'[edter] frame done: {elapsed*1000:.0f} ms total', flush=True)
        return cv2.resize(edge[:h0, :w0], (w0, h0), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# DiffusionEdge — Diffusion-Based Edge Detection (Ye et al., AAAI 2024)
# https://github.com/GuHuangAI/DiffusionEdge
#
# Two-stage latent diffusion: AutoencoderKL VAE (first_stage_total_320.pt)
# + conditional UNet (bsds.pt).  Runs a continuous-time SDE sampling loop
# (default 5 steps) and decodes the latent into a 1-channel edge map.
#
# Requires libs/diffusion_edge (git clone of the repo) on the Python path
# and the following extra packages: einops, timm==0.6.12, fvcore, ema-pytorch.
# =============================================================================

class DiffusionEdgeRunner:
    CROP   = 320
    STRIDE = 240

    def __init__(self, model_path, device='cpu', *,
                 first_stage_path, config_path, sampling_steps=None):
        import torch
        import yaml
        from fvcore.common.config import CfgNode

        if _DIFF_EDGE_ROOT not in _sys.path:
            _sys.path.insert(0, _DIFF_EDGE_ROOT)

        from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion

        def _build(dev):
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            cfg = CfgNode(raw)
            mc  = cfg.model
            fsc = mc.first_stage

            vae = AutoencoderKL(
                ddconfig  = fsc.ddconfig,
                lossconfig= fsc.lossconfig,
                embed_dim = fsc.embed_dim,
                ckpt_path = first_stage_path,
            )

            uc   = mc.unet
            unet = Unet(
                dim            = uc.dim,
                channels       = uc.channels,
                dim_mults      = uc.dim_mults,
                learned_variance = uc.get('learned_variance', False),
                out_mul        = uc.out_mul,
                cond_in_dim    = uc.cond_in_dim,
                cond_dim       = uc.cond_dim,
                cond_dim_mults = uc.cond_dim_mults,
                window_sizes1  = uc.window_sizes1,
                window_sizes2  = uc.window_sizes2,
                fourier_scale  = uc.fourier_scale,
                cfg            = uc,
            )

            steps = sampling_steps or mc.get('sampling_timesteps', 5)
            ldm   = LatentDiffusion(
                model              = unet,
                auto_encoder       = vae,
                train_sample       = mc.train_sample,
                image_size         = mc.image_size,
                timesteps          = mc.timesteps,
                sampling_timesteps = steps,
                loss_type          = mc.loss_type,
                objective          = mc.objective,
                scale_factor       = mc.scale_factor,
                scale_by_std       = mc.scale_by_std,
                scale_by_softsign  = mc.scale_by_softsign,
                default_scale      = mc.get('default_scale', False),
                input_keys         = mc.input_keys,
                start_dist         = mc.start_dist,
                perceptual_weight  = mc.perceptual_weight,
                use_l1             = mc.get('use_l1', True),
                cfg                = mc,
            )
            ldm.init_from_ckpt(model_path, use_ema=True)
            ldm.eval().to(torch.device(dev))
            return ldm, steps, torch.device(dev)

        def _is_cuda_error(e):
            s = str(e)
            return '127' in s or 'cudnn' in s.lower() or 'cuda' in s.lower()

        try:
            self.ldm, self.steps, self._device = _build(device)
        except Exception as e:
            if device != 'cpu' and _is_cuda_error(e):
                print(f'[diffusion_edge] CUDA failed ({e}), retrying on CPU', flush=True)
                self.ldm, self.steps, self._device = _build('cpu')
            else:
                raise

        self._torch = torch
        print(f'[diffusion_edge] Loaded {model_path}  device={self._device}  steps={self.steps}', flush=True)

    def infer(self, bgr_frame):
        import torch.nn.functional as F
        import time as _time
        torch  = self._torch
        h0, w0 = bgr_frame.shape[:2]

        if not getattr(self, '_infer_diag_done', False):
            param_dev = next(self.ldm.parameters()).device
            print(f'[diffusion_edge] infer diag: self._device={self._device}  '
                  f'model param device={param_dev}', flush=True)
            self._infer_diag_done = True
        _t0 = _time.perf_counter()

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5  # [-1, 1]

        ph = max(0, self.CROP - h0)
        pw = max(0, self.CROP - w0)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        H, W = rgb.shape[:2]

        inp = self._torch.from_numpy(
            rgb.transpose(2, 0, 1)).unsqueeze(0).to(self._device)  # [1, 3, H, W]

        h_grids = max(H - self.CROP + self.STRIDE - 1, 0) // self.STRIDE + 1
        w_grids = max(W - self.CROP + self.STRIDE - 1, 0) // self.STRIDE + 1
        preds   = inp.new_zeros((1, 1, H, W))
        cnt_mat = inp.new_zeros((1, 1, H, W))
        print(f'[diffusion_edge] tiles={h_grids}×{w_grids}={h_grids*w_grids}  '
              f'steps={self.steps}  frame={w0}×{h0}', flush=True)

        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * self.STRIDE;  x1 = wi * self.STRIDE
                y2 = min(y1 + self.CROP, H);  x2 = min(x1 + self.CROP, W)
                y1 = max(y2 - self.CROP, 0);  x1 = max(x2 - self.CROP, 0)
                crop = inp[:, :, y1:y2, x1:x2]
                _tc = _time.perf_counter()
                with torch.no_grad():
                    out = self.ldm.sample(batch_size=1, cond=crop, mask=None)
                if hi == 0 and wi == 0:
                    print(f'[diffusion_edge] first tile: {(_time.perf_counter()-_tc)*1000:.0f} ms', flush=True)
                preds += F.pad(out, (x1, W - x2, y1, H - y2))
                cnt_mat[:, :, y1:y2, x1:x2] += 1.0

        edge = (preds / cnt_mat.clamp(min=1e-6)).squeeze().cpu().numpy()
        elapsed = _time.perf_counter() - _t0
        print(f'[diffusion_edge] frame done: {elapsed*1000:.0f} ms total', flush=True)
        return cv2.resize(edge[:h0, :w0], (w0, h0), interpolation=cv2.INTER_LINEAR)
