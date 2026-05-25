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
    HEDRunner            — Holistically-Nested Edge Detection (Xie & Tu, 2015)       PyTorch
    NBEDRunner           — A New Baseline for Edge Detection (Ze et al., 2024)        PyTorch
    DiffusionEdgeRunner  — Diffusion-based Edge Detection (Ye et al., AAAI 2024)     PyTorch
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

    def infer(self, bgr_frame, mask=None):
        # mask is unused: HED resizes the whole frame to 480×480 in a single
        # forward pass — there is no tile loop to gate on coverage.  Accepted
        # only to keep the runner interface uniform with EDTER/DiffusionEdge.
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
# NBED — A New Baseline for Edge Detection (arXiv 2024)
# https://github.com/Bin-ze/NBED   (cloned at models/NBED-main)
#
# Single-pass full-frame CNN+Transformer (CAFormer-M36 with optional Dynamic
# Up-sampling Layer encoder, UNet++-style decoder, default conv head).
# No sliding window, no tiling, no fusion stages — runs in one forward.
# Pure PyTorch + timm; uses the official model code from models/NBED-main/
# verbatim (we don't reimplement classes).
# Weights: models/nbed/NBED-BIPED.pth (recommended for laser projection per
# the NBED README — BIPED-trained checkpoints emit cleaner long contours
# than BSDS-trained ones, which suits the downstream Steger thinner.)
# =============================================================================

_nbed_cls         = None    # cached Basemodel class after first build
_nbed_runtime_pkg = None    # cached `model.*` package once sys.path is wired


def _build_nbed_model():
    """Instantiate the official NBED Basemodel. Lazy: imports torch/timm/NBED
    on first call only. Returns (model_class, model_instance_factory)."""
    global _nbed_cls, _nbed_runtime_pkg
    if _nbed_cls is not None:
        return _nbed_cls

    import sys as _s, os as _o

    # NBED's source uses the legacy timm import path; alias to the modern one
    # before any NBED module imports.
    import timm.layers.helpers as _modern_helpers
    _s.modules.setdefault('timm.models.layers.helpers', _modern_helpers)

    nbed_root = _o.path.normpath(_o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), '..', 'models', 'NBED-main'))
    if nbed_root not in _s.path:
        _s.path.insert(0, nbed_root)

    # NBED's get_encoder() hard-codes pretrained=True, which triggers a load
    # of model/caformer_m36_384_in21ft1k.pth (the ImageNet pretrained file).
    # Our BIPED checkpoint already carries every encoder weight, so we
    # monkey-patch to flip pretrained=False — avoids requiring the extra
    # ImageNet file on disk and skips a ~360 MB load we'd overwrite anyway.
    from model import utils as _nbed_utils

    def _get_encoder_no_pretrained(nm, Dulbrn=16):
        nu = nm.upper()
        if nu == 'DUL-M36':
            from model.caformer import caformer_m36_384_in21ft1k
            return caformer_m36_384_in21ft1k(pretrained=False, Dulbrn=Dulbrn)
        if nu == 'CAFORMER-M36':
            from model.caformer import caformer_m36_384_in21ft1k
            return caformer_m36_384_in21ft1k(pretrained=False)
        if nu == 'DUL-S18':
            from model.caformer import caformer_s18_384_in21ft1k
            return caformer_s18_384_in21ft1k(pretrained=False, Dulbrn=Dulbrn)
        raise ValueError(f'Unsupported NBED encoder {nm!r}')

    _nbed_utils.get_encoder = _get_encoder_no_pretrained

    from model.basemodel import Basemodel
    _nbed_cls = Basemodel
    _nbed_runtime_pkg = _nbed_utils
    return _nbed_cls


class NBEDRunner:
    """
    NBED (Ze et al., 2024) — pure PyTorch, no mmcv, no tiling.

    Single forward pass per frame on the full image.  Input is normalised to
    [-1, 1] (NBED's own preprocessing).  Output is a sigmoid-activated edge
    probability map at the input resolution.

    Default architecture: DUL-M36 encoder, UNet++ decoder, default conv head.
    BIPED checkpoint is the laser-friendly choice (cleaner long contours).

    Compatible with the SAM2 mask gating used elsewhere in vectorize.py: the
    mask is ignored at inference time (NBED runs on the full frame in one
    pass and is cheap enough not to need tile skipping) but downstream
    consumers can still AND the output against the mask themselves.
    """

    # Strict-load sentinel set: one weight per top-level branch.
    # If any of these drift from the checkpoint value after load, refuse to run.
    _SENTINELS = (
        'encoder.conv1.0.weight',          # DUL local conv1
        'encoder.conv2.0.weight',          # DUL local conv2 (post-rename)
        'encoder.stages.0.0.norm1.weight', # CAFormer first block, first stage
        'decoder.convs.conv0_0.conv1.0.weight',  # UNet++ first decoder block
        'head.final.0.weight',             # head final conv (post-rename)
    )

    def __init__(self, model_path,
                 device='cuda',
                 encoder='DUL-M36',
                 decoder='UNETP',
                 head='default'):
        import torch
        self._torch = torch

        Basemodel = _build_nbed_model()
        model = Basemodel(encoder_name=encoder,
                          decoder_name=decoder,
                          head_name=head)

        # ── Load BIPED / BSDS checkpoint via the official inference.py recipe
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('state_dict', ckpt)

        # Strip DataParallel prefix if present
        if any(k.startswith('module.') for k in sd):
            sd = {(k[len('module.'):] if k.startswith('module.') else k): v
                  for k, v in sd.items()}

        # Conditional key renames — BIPED checkpoint uses different parameter
        # paths than our reconstructed model expects.  Per the official
        # inference.py, encoder.conv2.1.* moved to encoder.conv2.0.*, and we
        # additionally found decoder.final.0.* lives at head.final.0.*.
        for old, new in (
            ('encoder.conv2.1.weight',  'encoder.conv2.0.weight'),
            ('encoder.conv2.1.bias',    'encoder.conv2.0.bias'),
            ('decoder.final.0.weight',  'head.final.0.weight'),
            ('decoder.final.0.bias',    'head.final.0.bias'),
        ):
            if old in sd:
                sd[new] = sd.pop(old)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            from collections import Counter
            def _branch(k):
                p = k.split('.', 2)
                return '.'.join(p[:2]) if len(p) > 1 else p[0]
            print(f'[nbed] checkpoint load: {len(missing)} missing, '
                  f'{len(unexpected)} unexpected', flush=True)
            if missing:
                print(f'  missing by branch:    '
                      f'{dict(Counter(map(_branch, missing)))}', flush=True)
                print(f'  first 20 missing:     {missing[:20]}', flush=True)
            if unexpected:
                print(f'  unexpected by branch: '
                      f'{dict(Counter(map(_branch, unexpected)))}', flush=True)
                print(f'  first 20 unexpected:  {unexpected[:20]}', flush=True)
            raise RuntimeError(
                f'NBED checkpoint load incomplete: {len(missing)} missing, '
                f'{len(unexpected)} unexpected. Refusing to run on '
                f'partially-initialised weights — see log above.')

        # Sentinel parameter check — proves the load actually populated each
        # branch.  Catches the "load reported clean but values stayed random"
        # failure mode that bit us with EDTER.
        named = dict(model.named_parameters())
        for key in self._SENTINELS:
            if key not in named:
                raise RuntimeError(
                    f'[nbed] sentinel key {key!r} not present in model — '
                    f'NBED source layout drifted; update _SENTINELS.')
            if key not in sd:
                raise RuntimeError(
                    f'[nbed] sentinel key {key!r} not in checkpoint — '
                    f'wrong checkpoint variant? Expected BIPED/BSDS DUL-M36.')
            diff = (named[key].detach().cpu()
                    - sd[key].detach().cpu()).abs().max().item()
            print(f'  [nbed] sentinel {key}: max|model-ckpt| = {diff:.2e}',
                  flush=True)
            if diff > 1e-6:
                raise RuntimeError(
                    f'[nbed] sentinel mismatch on {key}: max diff {diff} '
                    f'— load silently failed.')

        print(f'[nbed] checkpoint verified, '
              f'{len(sd)} keys loaded, all {len(self._SENTINELS)} '
              f'sentinels match', flush=True)

        # ── Place on device.  Fall back to CPU on CUDA failure (matches the
        #    pattern used by HEDRunner / DiffusionEdgeRunner above).
        def _is_cuda_err(e):
            s = str(e).lower()
            return 'cuda' in s or 'cudnn' in s or '127' in str(e)

        try:
            model.eval().to(torch.device(device))
            self._device = torch.device(device)
        except Exception as e:
            if device != 'cpu' and _is_cuda_err(e):
                print(f'[nbed] CUDA failed ({e}), retrying on CPU', flush=True)
                model.eval().to(torch.device('cpu'))
                self._device = torch.device('cpu')
            else:
                raise

        self._model = model
        print(f'[nbed] Loaded {model_path}  device={self._device}  '
              f'encoder={encoder}/{decoder}/{head}', flush=True)

    def infer(self, bgr_frame, mask=None):
        """Run NBED on a single BGR uint8 frame, return float32 [H,W] in [0,1].

        `mask` parameter accepted for interface compatibility with other
        runners (HED/DiffusionEdge gate by it); NBED is fast enough to not
        need tile skipping and ignores it."""
        import time as _time
        _t0 = _time.perf_counter()
        torch = self._torch

        h0, w0 = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        x   = x * 2.0 - 1.0                                 # [-1, 1]
        x   = x.to(self._device, non_blocking=True)

        with torch.no_grad():
            edge = self._model(x)

        # Most NBED configs (incl. DUL-M36 + UNETP) emit at input resolution,
        # but interpolate defensively in case a future encoder downsamples.
        if edge.shape[-2:] != (h0, w0):
            edge = torch.nn.functional.interpolate(
                edge, size=(h0, w0), mode='bilinear', align_corners=False)

        edge_np = edge.squeeze().detach().cpu().numpy().astype(np.float32)
        edge_np = np.clip(edge_np, 0.0, 1.0)

        elapsed = _time.perf_counter() - _t0
        above_half = float((edge_np > 0.5).sum()) / edge_np.size * 100.0
        print(f'[nbed] frame: shape={edge_np.shape}  min={edge_np.min():.4f}  '
              f'max={edge_np.max():.4f}  mean={edge_np.mean():.4f}  '
              f'px>0.5: {above_half:.1f}%  {elapsed*1000:.0f} ms',
              flush=True)
        return edge_np


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
    """
    DiffusionEdge — sliding window 320×320, stride 240.

    Tile skipping: when an optional SAM2 mask is supplied to infer(), tiles
    whose mask coverage is below TILE_SKIP_THRESHOLD are skipped entirely.
    Each diffusion sample takes ~5 denoising steps so skipping background
    tiles is significantly cheaper than for EDTER.
    """
    CROP                = 320
    STRIDE              = 240
    TILE_SKIP_THRESHOLD = 0.05

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

    def infer(self, bgr_frame, mask=None):
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

        skipped = 0
        total   = h_grids * w_grids
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * self.STRIDE;  x1 = wi * self.STRIDE
                y2 = min(y1 + self.CROP, H);  x2 = min(x1 + self.CROP, W)
                y1 = max(y2 - self.CROP, 0);  x1 = max(x2 - self.CROP, 0)

                # Skip tiles with negligible mask coverage. Both axes are
                # clamped to (h0, w0) so reflect-padded regions never count
                # as foreground.
                if mask is not None:
                    my2 = min(y2, h0); mx2 = min(x2, w0)
                    if my2 <= y1 or mx2 <= x1 \
                       or mask[y1:my2, x1:mx2].mean() < self.TILE_SKIP_THRESHOLD:
                        skipped += 1
                        continue

                crop = inp[:, :, y1:y2, x1:x2]
                _tc = _time.perf_counter()
                with torch.no_grad():
                    out = self.ldm.sample(batch_size=1, cond=crop, mask=None)
                if not getattr(self, '_first_tile_logged', False):
                    print(f'[diffusion_edge] first tile: {(_time.perf_counter()-_tc)*1000:.0f} ms', flush=True)
                    self._first_tile_logged = True
                preds += F.pad(out, (x1, W - x2, y1, H - y2))
                cnt_mat[:, :, y1:y2, x1:x2] += 1.0

        print(f'[diffusion_edge] tiles={h_grids}x{w_grids}={total}  '
              f'skipped={skipped}  steps={self.steps}  frame={w0}x{h0}', flush=True)

        edge = (preds / cnt_mat.clamp(min=1e-6)).squeeze().cpu().numpy()
        elapsed = _time.perf_counter() - _t0
        print(f'[diffusion_edge] frame done: {elapsed*1000:.0f} ms total', flush=True)
        return cv2.resize(edge[:h0, :w0], (w0, h0), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# Canny — classical Sobel-of-Gaussian + hysteresis edge detector.
#
# Wrapped as a Runner so Stage 1 can dispatch over all four edge methods
# uniformly.  No model file; parameters baked in from defaults.py.
# =============================================================================

class CannyRunner:
    """OpenCV Canny edge detector, presented as a uniform Runner.

    Returns a float32 [H, W] edge map in [0, 1] — the binary cv2.Canny output
    cast to the soft-map contract so Stage 2 can thin it identically to HED /
    NBED / DiffusionEdge outputs.
    """

    def __init__(self, model_path=None, device='cpu', *,
                 low=40, high=120, blur_k=3):
        # model_path / device unused; accepted for interface compatibility
        # with the loader signature in vectorize.py.
        self.low    = int(low)
        self.high   = int(high)
        self.blur_k = int(blur_k)
        print(f'[canny] Loaded  low={self.low}  high={self.high}  '
              f'blur_k={self.blur_k}', flush=True)

    def infer(self, bgr_frame, mask=None):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if self.blur_k > 1:
            k = self.blur_k | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, self.low, self.high)
        return edges.astype(np.float32) / 255.0
