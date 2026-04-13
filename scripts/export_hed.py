"""
Export HED (Holistically-nested Edge Detection) to ONNX.

Architecture: VGG16 backbone + 5 side outputs + weighted fusion.
Pre-trained weights: sniklaus/pytorch-hed (BSD500 dataset).

Usage:
    pip install torch torchvision requests
    python scripts/export_hed.py

Output: models/hed/model.onnx

The exported model:
    Input  "data"  : [1, 3, H, W]  float32
                     BGR channel order, mean-subtracted
                     (subtract [104.007, 116.669, 122.679] before passing)
    Output "fused" : [1, 1, H, W]  float32  (sigmoid edge probability, 0..1)

H and W are dynamic — the model runs at whatever resolution you feed it.
For real-time use, resize frames to 480x480 before inference.
"""

import os
import struct
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition
# Matches the sniklaus/pytorch-hed weight layout exactly.
# ---------------------------------------------------------------------------

class HED(nn.Module):
    def __init__(self):
        super().__init__()

        # VGG16 feature blocks (no FC layers, no classifier)
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2, stride=2)

        # Side output projections (each block -> 1-channel edge score)
        self.side1 = nn.Conv2d(64,  1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)

        # Weighted fusion of all five side outputs
        self.fuse  = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        h1 = self.block1(x)
        h2 = self.block2(self.pool(h1))
        h3 = self.block3(self.pool(h2))
        h4 = self.block4(self.pool(h3))
        h5 = self.block5(self.pool(h4))

        def up(t):
            return F.interpolate(t, (H, W), mode='bilinear', align_corners=False)

        s1 = torch.sigmoid(up(self.side1(h1)))
        s2 = torch.sigmoid(up(self.side2(h2)))
        s3 = torch.sigmoid(up(self.side3(h3)))
        s4 = torch.sigmoid(up(self.side4(h4)))
        s5 = torch.sigmoid(up(self.side5(h5)))

        fused = torch.sigmoid(self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1)))
        return fused   # [1, 1, H, W]


# ---------------------------------------------------------------------------
# Weight loading — sniklaus/pytorch-hed checkpoint format
# ---------------------------------------------------------------------------

WEIGHTS_URL = (
    "http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch"
)
WEIGHTS_FILE = "scripts/output/network-bsds500.pytorch"


def download_weights():
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    if os.path.exists(WEIGHTS_FILE):
        print(f"Weights already downloaded: {WEIGHTS_FILE}")
        return
    print(f"Downloading pre-trained weights from sniklaus/pytorch-hed ...")
    print(f"  URL: {WEIGHTS_URL}")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
    print(f"  Saved to: {WEIGHTS_FILE}")


# Map from sniklaus checkpoint keys to our module names.
# The sniklaus repo uses moduleVggXx notation.
# sniklaus/pytorch-hed actual key names (moduleVgg*, moduleScore*, moduleCombine).
# Blocks 2-5 include MaxPool at index 0 of their Sequential, so conv weights
# start at index 1 (block1 has no leading pool, so its convs start at index 0).
SNIKLAUS_KEY_MAP = {
    # Block 1 — no leading pool
    "moduleVggOne.0.weight": "block1.0.weight",
    "moduleVggOne.0.bias":   "block1.0.bias",
    "moduleVggOne.2.weight": "block1.2.weight",
    "moduleVggOne.2.bias":   "block1.2.bias",

    # Block 2 — pool at index 0, convs at 1 and 3
    "moduleVggTwo.1.weight": "block2.0.weight",
    "moduleVggTwo.1.bias":   "block2.0.bias",
    "moduleVggTwo.3.weight": "block2.2.weight",
    "moduleVggTwo.3.bias":   "block2.2.bias",

    # Block 3 — pool at index 0, convs at 1, 3, 5
    "moduleVggThr.1.weight": "block3.0.weight",
    "moduleVggThr.1.bias":   "block3.0.bias",
    "moduleVggThr.3.weight": "block3.2.weight",
    "moduleVggThr.3.bias":   "block3.2.bias",
    "moduleVggThr.5.weight": "block3.4.weight",
    "moduleVggThr.5.bias":   "block3.4.bias",

    # Block 4 — pool at index 0, convs at 1, 3, 5
    "moduleVggFou.1.weight": "block4.0.weight",
    "moduleVggFou.1.bias":   "block4.0.bias",
    "moduleVggFou.3.weight": "block4.2.weight",
    "moduleVggFou.3.bias":   "block4.2.bias",
    "moduleVggFou.5.weight": "block4.4.weight",
    "moduleVggFou.5.bias":   "block4.4.bias",

    # Block 5 — pool at index 0, convs at 1, 3, 5
    "moduleVggFiv.1.weight": "block5.0.weight",
    "moduleVggFiv.1.bias":   "block5.0.bias",
    "moduleVggFiv.3.weight": "block5.2.weight",
    "moduleVggFiv.3.bias":   "block5.2.bias",
    "moduleVggFiv.5.weight": "block5.4.weight",
    "moduleVggFiv.5.bias":   "block5.4.bias",

    # Side output projections
    "moduleScoreOne.weight": "side1.weight",
    "moduleScoreOne.bias":   "side1.bias",
    "moduleScoreTwo.weight": "side2.weight",
    "moduleScoreTwo.bias":   "side2.bias",
    "moduleScoreThr.weight": "side3.weight",
    "moduleScoreThr.bias":   "side3.bias",
    "moduleScoreFou.weight": "side4.weight",
    "moduleScoreFou.bias":   "side4.bias",
    "moduleScoreFiv.weight": "side5.weight",
    "moduleScoreFiv.bias":   "side5.bias",

    # Fusion layer
    "moduleCombine.weight":  "fuse.weight",
    "moduleCombine.bias":    "fuse.bias",
}


def load_sniklaus_weights(model: HED):
    ckpt = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False)

    # The checkpoint may be wrapped under a "state_dict" key or be flat.
    state = ckpt.get("state_dict", ckpt)

    # Print actual keys if nothing maps, so we can diagnose quickly.
    matched = [k for k in SNIKLAUS_KEY_MAP if k in state]
    if not matched:
        print("  No keys matched. All checkpoint keys:")
        for k in state.keys():
            print(f"    {k}")
        raise RuntimeError("Key map mismatch — see keys above and update SNIKLAUS_KEY_MAP")

    mapped = {}
    for src_key, dst_key in SNIKLAUS_KEY_MAP.items():
        if src_key in state:
            mapped[dst_key] = state[src_key]
        else:
            print(f"  WARNING: key not found in checkpoint: {src_key}")

    missing, _ = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"  Missing keys after load: {missing}")
    print("  Weights loaded OK")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

OUTPUT_ONNX = "models/hed/model.onnx"


def export_onnx(model: HED):
    os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)
    model.eval()

    # Dummy input at 480x480.  H and W are exported as dynamic axes so the
    # model works at any resolution at runtime.
    dummy = torch.zeros(1, 3, 480, 480)

    torch.onnx.export(
        model,
        dummy,
        OUTPUT_ONNX,
        opset_version       = 12,   # widely supported by ONNX Runtime 1.17
        input_names         = ["data"],
        output_names        = ["fused"],
        dynamic_axes        = {
            "data":  {2: "height", 3: "width"},
            "fused": {2: "height", 3: "width"},
        },
        do_constant_folding = True,
        dynamo              = False,  # force legacy TorchScript exporter
    )
    print(f"ONNX model saved to: {OUTPUT_ONNX}")

    # Quick sanity check
    import onnx
    m = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(m)
    print(f"ONNX check passed.  Inputs: {[i.name for i in m.graph.input]}")
    print(f"                   Outputs: {[o.name for o in m.graph.output]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HED ONNX Export ===")

    download_weights()

    model = HED()
    print("Loading weights...")
    load_sniklaus_weights(model)

    print("Exporting to ONNX...")
    export_onnx(model)

    print("Done.  Place model at:  models/hed/model.onnx")
    print("C++ usage:")
    print("  hed.Initialize(HEDInference::Backend::CPU);")
    print("  hed.LoadModel(L\"models/hed/model.onnx\");")
    print("  auto frame = hed.ProcessFrame(bgrMat);")
