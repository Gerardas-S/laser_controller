"""
export_music_models.py
======================
Export CREPE pitch model to ONNX using torchcrepe (no TensorFlow required).

Install:
    pip install torchcrepe torch onnx onnxruntime

Usage:
    python scripts/export_music_models.py
    python scripts/export_music_models.py --model tiny
    python scripts/export_music_models.py --model full --outdir models/crepe
"""

import argparse
import os
import sys

import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def banner(msg):
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)


def export_crepe(model_size: str, out_dir: str) -> bool:
    banner(f"Exporting CREPE ({model_size}) via torchcrepe")

    try:
        import torch
    except ImportError:
        print("ERROR: pip install torch")
        return False

    try:
        import torchcrepe
        from torchcrepe.model import Crepe
    except ImportError:
        print("ERROR: pip install torchcrepe")
        return False

    # ---- Load model weights from bundled package assets ------------------
    import inspect
    sig = inspect.signature(Crepe.__init__)
    print(f"Crepe.__init__ signature: {sig}")

    # Try positional first, fall back to keyword
    try:
        crepe = Crepe(model_size)
    except TypeError:
        try:
            crepe = Crepe(capacity=model_size)
        except TypeError:
            crepe = Crepe()

    pkg_dir = os.path.dirname(torchcrepe.__file__)

    # torchcrepe bundles weights as .pth files; locate them
    candidates = [
        os.path.join(pkg_dir, 'assets', f'{model_size}.pth'),
        os.path.join(pkg_dir, f'{model_size}.pth'),
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), None)

    if weights_path is None:
        # Last resort: walk package dir to find the .pth
        for root, _, files in os.walk(pkg_dir):
            for f in files:
                if f == f'{model_size}.pth':
                    weights_path = os.path.join(root, f)
                    break
            if weights_path:
                break

    if weights_path is None:
        print(f"Could not find {model_size}.pth in torchcrepe package.")
        print(f"Package directory: {pkg_dir}")
        print(f"Contents: {os.listdir(pkg_dir)}")
        return False

    print(f"Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    crepe.load_state_dict(state)
    crepe.eval()

    n_params = sum(p.numel() for p in crepe.parameters())
    print(f"Parameters: {n_params:,}")

    # ---- Export to ONNX --------------------------------------------------
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "model.onnx")

    dummy = torch.zeros(1, 1024)   # (batch, 1024 samples @16kHz)

    print(f"Exporting to ONNX (opset 12) → {out_path} ...")
    torch.onnx.export(
        crepe,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved: {out_path}  ({size_kb:.0f} KB)")

    # ---- Verify ----------------------------------------------------------
    _verify_onnx(out_path)
    return True


def _verify_onnx(path: str) -> None:
    banner("Verifying ONNX model")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp  = sess.get_inputs()[0]
        out  = sess.get_outputs()[0]
        print(f"  Input : {inp.name}  shape={inp.shape}  type={inp.type}")
        print(f"  Output: {out.name}  shape={out.shape}  type={out.type}")

        # Synthetic test: 440 Hz (A4) sine at 16 kHz
        sr  = 16000
        n   = 1024
        t   = np.arange(n, dtype=np.float32) / sr
        sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        salience = sess.run(None, {inp.name: sig[np.newaxis, :]})[0][0]  # (360,)
        best_bin = int(np.argmax(salience))
        hz = 10.0 * 2.0 ** ((best_bin * 20 + 1997.3796) / 1200.0)
        expected = round((np.log2(440.0 / 10.0) * 1200.0 - 1997.3796) / 20.0)

        print(f"  Synthetic 440 Hz → bin {best_bin} ({hz:.1f} Hz)  "
              f"expected bin ~{expected}")
        if abs(best_bin - expected) <= 3:
            print("  PASS")
        else:
            print("  WARN: result outside ±3 bins of ground truth")

    except Exception as ex:
        print(f"  Verification error: {ex}")


def main():
    parser = argparse.ArgumentParser(
        description="Export CREPE pitch model to ONNX via torchcrepe"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large", "full"],
        default="full",
        help="Model capacity (default: full). tiny≈400KB  full≈22MB",
    )
    parser.add_argument(
        "--outdir",
        default="models/crepe",
        help="Output directory (default: models/crepe)",
    )
    args = parser.parse_args()

    ok = export_crepe(args.model, args.outdir)
    if not ok:
        sys.exit(1)

    print("\nDone. models/crepe/model.onnx is ready for PitchAnalyzer.")


if __name__ == "__main__":
    main()
