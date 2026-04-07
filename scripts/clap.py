"""
Inspect ONNX model structure - works without numpy crashing
"""

import sys

print("Inspecting ONNX model...")
print("="*60)

try:
    import onnx
    print("✓ onnx library loaded")
except ImportError:
    print("❌ Install: pip install onnx")
    sys.exit(1)

MODEL_PATH = "../models/clap/model.onnx"

print(f"\nLoading: {MODEL_PATH}")

try:
    model = onnx.load(MODEL_PATH)
    print("✓ Model loaded")
    
    print("\n" + "="*60)
    print("INPUTS:")
    print("="*60)
    for input in model.graph.input:
        print(f"  Name: {input.name}")
        print(f"  Type: {input.type}")
        print()
    
    print("="*60)
    print("OUTPUTS:")
    print("="*60)
    for output in model.graph.output:
        print(f"  Name: {output.name}")
        print(f"  Type: {output.type}")
        print()
    
    print("="*60)
    print(f"Total nodes: {len(model.graph.node)}")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()