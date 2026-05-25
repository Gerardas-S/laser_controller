"""
4-stage modular pipeline for ILDA laser projection vectorization.

Stages
------
    0. preprocess   — Frame  → Frame     (CLAHE + bilateral, or identity)
    1. edge_detect  — Frame  → EdgeMap   (Canny / HED / NBED / DiffusionEdge)
    2. thinning     — EdgeMap → Graph    (NMS / Zhang-Suen / Steger / none)
    3. vectorize    — Graph  → Polylines (dp_vw / greedy_eulerian / iarussi)
    4. postprocess  — Polylines → JSON   (V-W, intensity/color sample, splines)

Data contracts are defined in `types.py`.  Method names are registered in
`registry.py`.  Locked canonical parameters live in `defaults.py`.

Adding a new method
-------------------
1. Implement it in the appropriate stage module (e.g. `edge_detect.py`).
2. Add one entry to the relevant dispatch table in `registry.py`.
3. (Optional) Add a defaults dict in `defaults.py` if it has tunable params.

That's it — `vectorize.py` doesn't change.
"""

from .types import Frame, Mask, EdgeMap, Graph, Chain, InternalPolyline, JSONPolyline
from .driver import run_pipeline, process_frame

__all__ = [
    'Frame', 'Mask', 'EdgeMap', 'Graph', 'Chain',
    'InternalPolyline', 'JSONPolyline',
    'run_pipeline', 'process_frame',
]
