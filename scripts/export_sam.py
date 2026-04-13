"""
Export SAM (Segment Anything Model) to ONNX (encoder + decoder).

Model options (set MODEL_TYPE below):
    "vit_h"  — SAM ViT-H (best quality, ~2.4GB, slow on CPU)
    "vit_l"  — SAM ViT-L (good quality, ~1.2GB)
    "vit_b"  — SAM ViT-B (fastest SAM, ~375MB)

Since real-time performance is not required, vit_h is recommended.

Usage:
    pip install segment-anything onnx
    python scripts/export_sam.py

Weights are downloaded automatically.

Outputs:
    models/sam/encoder.onnx
    models/sam/decoder.onnx
"""

import os
import torch
import numpy as np

OUTPUT_DIR  = "models/sam"
MODEL_TYPE  = "vit_b"   # "vit_h" | "vit_l" | "vit_b"

WEIGHTS_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def export(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load SAM
    # -------------------------------------------------------------------------
    try:
        from segment_anything import sam_model_registry
        from segment_anything.utils.onnx import SamOnnxModel
    except ImportError:
        print("Install segment-anything first:  pip install segment-anything")
        raise

    weights_path = f"scripts/output/sam_{MODEL_TYPE}.pth"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    if not os.path.exists(weights_path):
        import urllib.request
        url = WEIGHTS_URLS[MODEL_TYPE]
        sizes = {'vit_h': '2.4GB', 'vit_l': '1.2GB', 'vit_b': '375MB'}
        print(f"Downloading SAM {MODEL_TYPE} weights (~{sizes[MODEL_TYPE]})...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, weights_path)
        print(f"  Saved to {weights_path}")

    print(f"Loading SAM {MODEL_TYPE}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=weights_path)
    sam.eval()
    print("SAM loaded OK")

    # -------------------------------------------------------------------------
    # Export encoder
    # -------------------------------------------------------------------------
    print("Exporting encoder (this may take several minutes for vit_h)...")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, sam):
            super().__init__()
            self.encoder = sam.image_encoder

        def forward(self, image):
            return self.encoder(image)

    encoder = EncoderWrapper(sam)
    dummy_image = torch.zeros(1, 3, 1024, 1024)

    encoder_path = os.path.join(output_dir, "encoder.onnx")
    print("  Running torch.onnx.export for encoder...")
    torch.onnx.export(
        encoder,
        dummy_image,
        encoder_path,
        opset_version    = 12,
        input_names      = ["image"],
        output_names     = ["image_embeddings"],
        dynamic_axes     = {},   # fixed 1024x1024 input
        do_constant_folding = True,
        dynamo           = False,
    )
    print(f"  Encoder saved: {encoder_path}")
    import sys; sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Export decoder
    # -------------------------------------------------------------------------
    print("Exporting decoder...")
    sys.stdout.flush()

    # SAM's built-in ONNX export for the mask decoder.
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    onnx_model.eval()

    # Dummy inputs matching the decoder interface.
    embed_dim   = sam.prompt_encoder.embed_dim          # 256
    embed_size  = sam.image_encoder.img_size // sam.image_encoder.patch_embed.proj.kernel_size[0]  # 64

    dummy_embeddings  = torch.zeros(1, embed_dim, embed_size, embed_size)
    dummy_coords      = torch.zeros(1, 1, 2,  dtype=torch.float)
    dummy_labels      = torch.zeros(1, 1,     dtype=torch.float)
    dummy_mask_input  = torch.zeros(1, 1, 256, 256)
    dummy_has_mask    = torch.zeros(1,          dtype=torch.float)
    dummy_orig_size   = torch.tensor([1024.0, 1024.0])

    decoder_path = os.path.join(output_dir, "decoder.onnx")
    torch.onnx.export(
        onnx_model,
        (dummy_embeddings, dummy_coords, dummy_labels,
         dummy_mask_input, dummy_has_mask, dummy_orig_size),
        decoder_path,
        opset_version = 12,
        input_names   = ["image_embeddings", "point_coords", "point_labels",
                         "mask_input", "has_mask_input", "orig_im_size"],
        output_names  = ["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes  = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        do_constant_folding = True,
        dynamo              = False,
    )
    print(f"  Decoder saved: {decoder_path}")

    # -------------------------------------------------------------------------
    # Sanity check
    # -------------------------------------------------------------------------
    import onnx
    for path in [encoder_path, decoder_path]:
        m = onnx.load(path)
        onnx.checker.check_model(m)
        inputs  = [i.name for i in m.graph.input]
        outputs = [o.name for o in m.graph.output]
        print(f"  {os.path.basename(path)}: inputs={inputs}  outputs={outputs}")

    print("\nDone.")
    print("C++ usage:")
    print("  sam.Initialize(SAMInference::Backend::CPU);")
    print("  sam.LoadModels(L\"models/sam/encoder.onnx\", L\"models/sam/decoder.onnx\");")
    print("  auto frame = sam.ProcessFrame(bgrMat);")


if __name__ == "__main__":
    print("=== MobileSAM ONNX Export ===")
    export(OUTPUT_DIR)
