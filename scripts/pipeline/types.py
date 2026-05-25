"""
Standardized data contracts shared by every pipeline stage.

These are intentionally lightweight type aliases over numpy / dict shapes
rather than dataclasses or pydantic models — the pipeline passes raw arrays
between stages with no copying, and dict shapes survive a JSON round-trip
unchanged.  The aliases exist purely for documentation and IDE hints.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Tuple, List
import numpy as np


# =============================================================================
# Raw array contracts
# =============================================================================

# BGR uint8 frame, shape (H, W, 3).  Passed through every stage at native
# resolution — no resizing inside the pipeline.
Frame = np.ndarray

# Boolean SAM2 union mask, shape (H, W).  Optional in most stages; required
# by color sampling (Option D) and by tile-skipping in NBED/DiffusionEdge.
Mask = np.ndarray

# Soft edge-probability map, shape (H, W), float32 in [0, 1].  Stage 1's
# output, Stage 2's input.  Canny's binary output is cast to this format
# (uint8 / 255) so the contract is uniform across all four edge methods.
EdgeMap = np.ndarray


# =============================================================================
# Graph contract  (Stage 2 → Stage 3)
# =============================================================================

class Chain(TypedDict):
    """A single pixel chain between two graph nodes (or a closed loop).

    pixels  : ordered (x, y) coordinates.  May be sub-pixel floats for Steger.
    node_a  : graph node id at chain[0], or None for pure loops.
    node_b  : graph node id at chain[-1], or None for pure loops.
    closed  : True only for chains with no graph node (closed pixel loops).
    """
    pixels: List[Tuple[float, float]]
    node_a: Optional[int]
    node_b: Optional[int]
    closed: bool


class Graph(TypedDict):
    """The unified Stage 2 → Stage 3 contract.

    Every thinning method produces a Graph.  Methods with no natural junctions
    (Steger) emit degenerate graphs where every chain has unique endpoint nodes
    and no chain shares a node with another chain.  Methods with junctions
    (Zhang-Suen, NMS, Canny) emit a fully-connected graph with shared nodes.

    chains   : list of Chain dicts.
    nodes    : dict mapping node id -> (x, y) pixel coordinates.
    soft_map : the underlying EdgeMap (used downstream for intensity sampling).
    mask     : the SAM2 mask (used downstream for color sampling and any
               post-filter mask gates).
    """
    chains: List[Chain]
    nodes:  dict
    soft_map: EdgeMap
    mask: Optional[Mask]


# =============================================================================
# Polyline contracts  (Stage 3 → Stage 4 → JSON)
# =============================================================================

class InternalPolyline(TypedDict, total=False):
    """Stage 3 output before postprocessing.

    Holds raw pixel-space coordinates; intensity/color sampling and [-1, 1]
    normalization happen in Stage 4.
    """
    path:     List[Tuple[float, float]]
    closed:   bool
    node_ids: Optional[Tuple[Optional[int], Optional[int]]]


class JSONPolyline(TypedDict, total=False):
    """Final pipeline output, written to disk by vectorize.py.

    pts          : normalized [-1, 1] coordinates with y-up (laser convention).
    intensities  : per-vertex, in [0, 1].
    colors       : per-vertex [r, g, b], each channel in [0, 1].
    closed       : whether the polyline is closed (cycle).
    outer        : True only for SAM2 silhouette polylines (encode.py protects
                   these from the temporal-persistence filter).
    node_ids     : optional 2-tuple of graph node ids at the polyline endpoints,
                   used by encode.py for zero-blanking chaining at junctions.
    """
    pts:         List[List[float]]
    intensities: List[float]
    colors:      List[List[float]]
    closed:      bool
    outer:       bool
    node_ids:    Optional[Tuple[Optional[int], Optional[int]]]
