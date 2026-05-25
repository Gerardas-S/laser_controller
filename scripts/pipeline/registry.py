"""
Dispatch tables for the 4-stage pipeline.

Each table maps a method name (string) to its implementation callable.
The pipeline driver looks up the callable by name and invokes it with the
stage's standardized inputs.

Adding a new method is one line per table.  No other module changes are
required outside the new method's own implementation.
"""

from . import preprocess, edge_detect, thinning, vectorize_graph


# =============================================================================
# Stage 0 — Frame -> Frame
# =============================================================================
PREPROCESS = {
    'none': preprocess.preprocess_none,
    'cb':   preprocess.preprocess_clahe_bilateral,    # CLAHE + bilateral
}

# =============================================================================
# Stage 1 — Frame, Mask -> EdgeMap
# =============================================================================
EDGE = {
    'canny': edge_detect.run_canny,
    'hed':   edge_detect.run_hed,
    'nbed':  edge_detect.run_nbed,
    'de':    edge_detect.run_diffusion_edge,          # DiffusionEdge
}

# =============================================================================
# Stage 2 — EdgeMap, Mask -> Graph
# =============================================================================
THIN = {
    'none':   thinning.thin_none,
    'nms':    thinning.thin_nms,
    'zs':     thinning.thin_zhang_suen,               # Zhang-Suen
    'steger': thinning.thin_steger,
}

# =============================================================================
# Stage 3 — Graph -> list[InternalPolyline]
# =============================================================================
VEC = {
    'dp':      vectorize_graph.vec_dp,                # per-chain (no traversal)
    'ge':      vectorize_graph.vec_greedy_eulerian,   # greedy Eulerian
    'iarussi': vectorize_graph.vec_iarussi,
}


# =============================================================================
# Convenience: full method name validation
# =============================================================================
def validate(stage0: str, edge: str, thin: str, vec: str) -> None:
    """Raise ValueError with a helpful message if any method name is unknown."""
    for tbl, name, val in (
        (PREPROCESS, 'stage0', stage0),
        (EDGE,       'edge',   edge),
        (THIN,       'thin',   thin),
        (VEC,        'vec',    vec),
    ):
        if val not in tbl:
            raise ValueError(
                f'Unknown {name} method {val!r}. '
                f'Valid: {sorted(tbl)}'
            )
