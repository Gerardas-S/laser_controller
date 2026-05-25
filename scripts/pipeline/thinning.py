"""
Stage 2 — Thinning.

Converts a soft edge map (EdgeMap, float32 [H,W] in [0,1]) into a Graph
representation that Stage 3 vectorizers consume uniformly.  Every method
returns the same Graph shape, so any (thin × vec) pair composes:

    Graph = {
        'chains':   list of {'pixels': [(x,y),...], 'node_a': int|None,
                              'node_b': int|None, 'closed': bool},
        'nodes':    {nid: (x, y)},
        'soft_map': original EdgeMap (forwarded for intensity sampling),
        'mask':     original SAM2 mask,
    }

Methods
-------
    none        — Canny pass-through: threshold > 0 and chain-decompose
                  (Canny output is already 1-pixel-wide).
    nms         — Non-maximum suppression on the soft map → threshold →
                  chain-decompose.
    zhang_suen  — Threshold → cv2.ximgproc Zhang-Suen thinning →
                  chain-decompose.  Produces full junction graphs.
    steger      — Steger (1998) sub-pixel ridge tracking + tangent-following
                  linker.  Produces a degenerate graph (one synthetic
                  endpoint-node per polyline tip; no junctions).
"""

import math
import cv2
import numpy as np

from .defaults import (
    THIN_NONE_DEFAULTS, THIN_NMS_DEFAULTS,
    THIN_ZHANG_SUEN_DEFAULTS, THIN_STEGER_DEFAULTS,
    THIN_STEGER_HED_OVERRIDES, THIN_STEGER_NBED_OVERRIDES,
    THIN_STEGER_DE_OVERRIDES, THIN_STEGER_CANNY_OVERRIDES,
    THIN_ZS_HED_OVERRIDES, THIN_ZS_NBED_OVERRIDES,
    THIN_ZS_DE_OVERRIDES, THIN_ZS_CANNY_OVERRIDES,
    THIN_NMS_HED_OVERRIDES, THIN_NMS_NBED_OVERRIDES,
    THIN_NMS_DE_OVERRIDES, THIN_NMS_CANNY_OVERRIDES,
)


# =============================================================================
# Per-edge-model override resolution (shared by all Stage 2 dispatchers)
# =============================================================================
# Outer key = Stage 2 method name (matches registry.THIN keys).
# Inner key = Stage 1 edge method name (matches registry.EDGE keys).

_OVERRIDE_TABLES = {
    'steger': {
        'hed':   THIN_STEGER_HED_OVERRIDES,
        'nbed':  THIN_STEGER_NBED_OVERRIDES,
        'de':    THIN_STEGER_DE_OVERRIDES,
        'canny': THIN_STEGER_CANNY_OVERRIDES,
    },
    'zs': {
        'hed':   THIN_ZS_HED_OVERRIDES,
        'nbed':  THIN_ZS_NBED_OVERRIDES,
        'de':    THIN_ZS_DE_OVERRIDES,
        'canny': THIN_ZS_CANNY_OVERRIDES,
    },
    'nms': {
        'hed':   THIN_NMS_HED_OVERRIDES,
        'nbed':  THIN_NMS_NBED_OVERRIDES,
        'de':    THIN_NMS_DE_OVERRIDES,
        'canny': THIN_NMS_CANNY_OVERRIDES,
    },
}


def _resolve_cfg(thin_method: str, base_cfg: dict, stage_ctx: dict) -> dict:
    """Merge per-edge-method overrides on top of the base config.

    Used by every Stage 2 dispatcher so they all handle calibration overrides
    uniformly.  Returns a NEW dict; never mutates `base_cfg` or the overrides.
    """
    cfg = dict(base_cfg)
    edge_method = stage_ctx.get('edge_method')
    if edge_method and thin_method in _OVERRIDE_TABLES:
        overrides = _OVERRIDE_TABLES[thin_method].get(edge_method, {})
        cfg.update(overrides)
    return cfg


# =============================================================================
# Mask helpers
# =============================================================================

_ERODE_KERNEL_DEFAULT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def _erode_mask(mask_u8: np.ndarray, kernel_size: int = 9):
    """Erode a uint8 mask.  Returns None if the result is empty."""
    if kernel_size <= 1:
        return mask_u8 if cv2.countNonZero(mask_u8) > 0 else None
    if kernel_size == 9:
        kernel = _ERODE_KERNEL_DEFAULT
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
    inner = cv2.erode(mask_u8, kernel, iterations=0)
    return inner if cv2.countNonZero(inner) > 0 else None


# =============================================================================
# Skeleton chain decomposition  (shared by NMS / Zhang-Suen / Canny-passthrough)
# =============================================================================

_SKEL_OFF8 = ((-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1))


def skeleton_chains(skel_u8: np.ndarray):
    """Decompose a 1-pixel-wide skeleton into pixel chains and node positions.

    Identical algorithm to vectorize.py's legacy `_skeleton_chains`.

    Returns
    -------
    chains : list of dicts with
              'pixels':  ordered [(x, y), ...] including both end-node
                         pixels (when they exist),
              'node_a':  int node id at chain[0]  (None for pure loops),
              'node_b':  int node id at chain[-1] (None for pure loops),
              'closed':  True only for node-free pixel loops.
    nodes  : dict[int node_id, (x, y)]  -- endpoints (deg==1) and junctions
             (deg>=3).
    """
    H, W = skel_u8.shape[:2]
    skel = skel_u8 > 0
    if not skel.any():
        return [], {}

    pad = np.pad(skel.astype(np.uint8), 1, mode='constant')
    deg = np.zeros((H, W), dtype=np.int32)
    for dy, dx in _SKEL_OFF8:
        deg += pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    deg = np.where(skel, deg, 0)

    is_node = skel & ((deg == 1) | (deg >= 3))

    node_id = np.full((H, W), -1, dtype=np.int32)
    nodes = {}
    ys, xs = np.nonzero(is_node)
    for y, x in zip(ys.tolist(), xs.tolist()):
        nid = len(nodes)
        node_id[y, x] = nid
        nodes[nid] = (int(x), int(y))

    def neighbours(y, x):
        out = []
        for dy, dx in _SKEL_OFF8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                out.append((ny, nx))
        return out

    chains = []
    visited_edges = set()

    def ek(y1, x1, y2, x2):
        a, b = (y1, x1), (y2, x2)
        return (a, b) if a < b else (b, a)

    for ny, nx in zip(ys.tolist(), xs.tolist()):
        for sy, sx in neighbours(ny, nx):
            k0 = ek(ny, nx, sy, sx)
            if k0 in visited_edges:
                continue
            visited_edges.add(k0)
            path = [(nx, ny), (sx, sy)]
            py, px = ny, nx
            cy, cx = sy, sx
            end_node = None
            while True:
                if node_id[cy, cx] >= 0:
                    end_node = int(node_id[cy, cx])
                    break
                nxt = None
                for ny2, nx2 in neighbours(cy, cx):
                    if (ny2, nx2) == (py, px):
                        continue
                    k1 = ek(cy, cx, ny2, nx2)
                    if k1 in visited_edges:
                        continue
                    nxt = (ny2, nx2, k1)
                    break
                if nxt is None:
                    break
                ny2, nx2, k1 = nxt
                visited_edges.add(k1)
                py, px = cy, cx
                cy, cx = ny2, nx2
                path.append((nx2, ny2))
            chains.append({
                'pixels': path,
                'node_a': int(node_id[ny, nx]),
                'node_b': end_node,
                'closed': False,
            })

    # Pure-loop components (no nodes).
    pixel_seen = np.zeros((H, W), dtype=bool)
    for c in chains:
        for (x, y) in c['pixels']:
            pixel_seen[y, x] = True
    rys, rxs = np.nonzero(skel & ~pixel_seen)
    loop_seen = np.zeros((H, W), dtype=bool)
    for y0, x0 in zip(rys.tolist(), rxs.tolist()):
        if loop_seen[y0, x0]:
            continue
        path = [(int(x0), int(y0))]
        loop_seen[y0, x0] = True
        py, px = -1, -1
        cy, cx = int(y0), int(x0)
        while True:
            pick = None
            for ny, nx in neighbours(cy, cx):
                if (ny, nx) == (py, px):
                    continue
                if loop_seen[ny, nx]:
                    if (nx, ny) == (x0, y0) and len(path) > 2:
                        pick = (ny, nx)
                    continue
                pick = (ny, nx)
                break
            if pick is None:
                break
            ny, nx = pick
            if (nx, ny) == (x0, y0):
                break
            loop_seen[ny, nx] = True
            path.append((int(nx), int(ny)))
            py, px = cy, cx
            cy, cx = ny, nx
        if len(path) >= 3:
            chains.append({
                'pixels': path,
                'node_a': None,
                'node_b': None,
                'closed': True,
            })

    return chains, nodes


def chain_tangent(pixels, end='head', k=5):
    """Unit tangent at one end of a chain, pointing FROM that end INTO the chain."""
    n = len(pixels)
    if n < 2:
        return (0.0, 0.0)
    k = min(k, n - 1)
    if end == 'head':
        ax, ay = pixels[0]
        bx, by = pixels[k]
    else:
        ax, ay = pixels[-1]
        bx, by = pixels[-1 - k]
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


# =============================================================================
# Steger ridge tracking  (Steger 1998, Hessian-based sub-pixel ridge detector)
# =============================================================================

def _refine_with_gradient_mask(bgr_frame, edge_map,
                                low_mask_th=0.05, high_mask_th=0.25,
                                gain_high=1.0, gain_mid=1.0, gain_low=1.0):
    """Han & Zhong (2020) gradient-mask filter — sharpens the soft probability
    map by multiplying with image-gradient-magnitude bins."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag    = cv2.magnitude(grad_x, grad_y)
    mag    = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    high_mask = (mag >= high_mask_th).astype(np.float32)
    low_mask  = (mag <  low_mask_th).astype(np.float32)
    mid_mask  = ((mag >= low_mask_th) & (mag < high_mask_th)).astype(np.float32)

    refined = edge_map * (high_mask * gain_high
                          + mid_mask * gain_mid
                          + low_mask * gain_low)
    return np.clip(refined, 0.0, 1.0)


def _gaussian_derivative_kernels(sigma):
    """Analytical Gaussian derivative kernels (Gx, Gy, Gxx, Gyy, Gxy)."""
    r = max(3, int(np.ceil(3.0 * sigma)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    s2 = sigma ** 2
    s4 = sigma ** 4
    gx  = (-x / s2) * g
    gy  = (-y / s2) * g
    gxx = ((x ** 2 - s2) / s4) * g
    gyy = ((y ** 2 - s2) / s4) * g
    gxy = (x * y / s4) * g
    return (gx.astype(np.float32),  gy.astype(np.float32),
            gxx.astype(np.float32), gyy.astype(np.float32),
            gxy.astype(np.float32))


def _steger_ridge_points(soft_f32, sigma=1.5, strength_th=0.005):
    """Steger (1998) sub-pixel ridge detection.

    Returns six 1-D arrays per ridge pixel: (ys, xs, sub_y, sub_x, tx, ty, strength)."""
    eps = 1e-10

    gx_k, gy_k, gxx_k, gyy_k, gxy_k = _gaussian_derivative_kernels(sigma)
    rx  = cv2.filter2D(soft_f32, cv2.CV_32F, gx_k)
    ry  = cv2.filter2D(soft_f32, cv2.CV_32F, gy_k)
    rxx = cv2.filter2D(soft_f32, cv2.CV_32F, gxx_k)
    ryy = cv2.filter2D(soft_f32, cv2.CV_32F, gyy_k)
    rxy = cv2.filter2D(soft_f32, cv2.CV_32F, gxy_k)
    del gx_k, gy_k, gxx_k, gyy_k, gxy_k

    half_trace = (rxx + ryy) * 0.5
    det = rxx * ryy - rxy * rxy
    disc = np.sqrt(np.maximum(0.0, half_trace * half_trace - det, out=det), out=det)
    del det
    lam_p = half_trace + disc
    lam_m = half_trace - disc
    del half_trace, disc

    pick_p = np.abs(lam_p) > np.abs(lam_m)
    lam_n = np.where(pick_p, lam_p, lam_m)
    del lam_p, lam_m, pick_p
    strength_full = -lam_n

    nx_a = lam_n - ryy
    norm_a = np.hypot(nx_a, rxy)
    ny_b = lam_n - rxx
    norm_b = np.hypot(rxy, ny_b)
    del lam_n
    use_a = norm_a >= norm_b
    del norm_a, norm_b
    nx_raw = np.where(use_a, nx_a, rxy)
    ny_raw = np.where(use_a, rxy, ny_b)
    del nx_a, ny_b, use_a
    n_norm = np.hypot(nx_raw, ny_raw)

    deg = n_norm < eps
    if np.any(deg):
        rxx_dom_deg = np.abs(rxx[deg]) >= np.abs(ryy[deg])
        nx_raw[deg] = rxx_dom_deg.astype(np.float32)
        ny_raw[deg] = (~rxx_dom_deg).astype(np.float32)
        n_norm[deg] = 1.0
        del rxx_dom_deg
    del deg
    nx = nx_raw / n_norm
    ny = ny_raw / n_norm
    del nx_raw, ny_raw, n_norm
    tx = -ny
    ty = nx

    denom = rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny
    del rxx, ryy, rxy
    numer = -(rx * nx + ry * ny)
    del rx, ry
    denom_bad = np.abs(denom) <= eps
    np.copyto(denom, 1.0, where=denom_bad)
    p = numer / denom
    del denom, numer
    p[denom_bad] = 0.0
    del denom_bad
    off_x = p * nx
    off_y = p * ny
    del p

    is_ridge = (np.abs(off_x) <= 0.5) & (np.abs(off_y) <= 0.5) & \
               (strength_full >= strength_th)
    ys, xs = np.where(is_ridge)
    del is_ridge
    return (ys.astype(np.int32),
            xs.astype(np.int32),
            ys + off_y[ys, xs],
            xs + off_x[ys, xs],
            tx[ys, xs],
            ty[ys, xs],
            strength_full[ys, xs])


def _steger_link_polylines(ys, xs, sub_y, sub_x, tan_x, tan_y, strength,
                            img_h, img_w, max_step_angle_deg=60.0):
    """Link Steger ridge pixels into continuous polylines via tangent walks."""
    n = len(ys)
    if n == 0:
        return []

    pix_idx = np.full((img_h, img_w), -1, dtype=np.int32)
    pix_idx[ys, xs] = np.arange(n, dtype=np.int32)

    cos_thr = math.cos(math.radians(max_step_angle_deg))
    consumed = np.zeros(n, dtype=bool)
    polylines = []

    order = np.argsort(-strength).tolist()
    offs = [(dy, dx, dy / math.hypot(dy, dx) if (dy or dx) else 0.0,
                     dx / math.hypot(dy, dx) if (dy or dx) else 0.0)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)]

    for seed in order:
        if consumed[seed]:
            continue
        consumed[seed] = True

        forward  = [(float(sub_x[seed]), float(sub_y[seed]))]
        backward = []

        for sign in (+1, -1):
            cur = seed
            cur_tx = sign * float(tan_x[cur])
            cur_ty = sign * float(tan_y[cur])
            target = forward if sign > 0 else backward

            while True:
                cy = int(ys[cur]); cx = int(xs[cur])
                best_i = -1; best_align = -2.0
                for dy, dx, sdy, sdx in offs:
                    ny2 = cy + dy; nx2 = cx + dx
                    if not (0 <= ny2 < img_h and 0 <= nx2 < img_w):
                        continue
                    ni = pix_idx[ny2, nx2]
                    if ni < 0 or consumed[ni]:
                        continue
                    align_step = sdx * cur_tx + sdy * cur_ty
                    if align_step <= 0.0:
                        continue
                    ntx = float(tan_x[ni]); nty = float(tan_y[ni])
                    align_tan = abs(ntx * cur_tx + nty * cur_ty)
                    if align_tan < cos_thr:
                        continue
                    score = align_step + 0.1 * align_tan
                    if score > best_align:
                        best_align = score; best_i = ni

                if best_i < 0:
                    break

                consumed[best_i] = True
                ntx = float(tan_x[best_i]); nty = float(tan_y[best_i])
                if ntx * cur_tx + nty * cur_ty < 0.0:
                    ntx = -ntx; nty = -nty
                cur = best_i
                cur_tx = ntx; cur_ty = nty
                target.append((float(sub_x[cur]), float(sub_y[cur])))

        path = list(reversed(backward)) + forward
        if len(path) >= 2:
            polylines.append({'path': path, 'closed': False})

    return polylines


def _link_polyline_endpoints(polylines, soft_f32, max_gap, max_angle_deg,
                              tangent_k=12, soft_th=0.05, co_circ_deg=12.0):
    """Polyline-tip directed gap closing (Steger safety net)."""
    if max_gap <= 1 or len(polylines) < 2:
        return list(polylines)

    H, W       = soft_f32.shape
    cos_thr    = math.cos(math.radians(max_angle_deg))
    cos_2thr   = math.cos(math.radians(min(89.9, 2.0 * max_angle_deg)))
    co_circ_r  = math.radians(co_circ_deg)
    gap2       = float(max_gap * max_gap)

    tips = []
    for pi, pl in enumerate(polylines):
        if pl.get('closed', False) or len(pl['path']) < 2:
            continue
        path = pl['path']
        x0, y0 = path[0]
        x1, y1 = path[min(len(path) - 1, tangent_k - 1)]
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6: continue
        tips.append({'pl': pi, 'side': 'h', 'xy': (x0, y0),
                     'tan': ((x0 - x1) / L, (y0 - y1) / L)})
        x0, y0 = path[-1]
        x1, y1 = path[max(0, len(path) - tangent_k)]
        L = math.hypot(x0 - x1, y0 - y1)
        if L < 1e-6: continue
        tips.append({'pl': pi, 'side': 't', 'xy': (x0, y0),
                     'tan': ((x0 - x1) / L, (y0 - y1) / L)})

    N = len(tips)
    if N < 2:
        return list(polylines)

    xy_arr = np.array([t['xy'] for t in tips], dtype=np.float64)
    candidates = []
    for i in range(N):
        ax, ay = tips[i]['xy']; tax, tay = tips[i]['tan']
        diffs  = xy_arr - xy_arr[i]
        d2     = (diffs ** 2).sum(axis=1)
        near   = np.where((d2 > 0) & (d2 <= gap2))[0]
        for j in near:
            if j <= i: continue
            if tips[i]['pl'] == tips[j]['pl']: continue
            bx, by = tips[j]['xy']; tbx, tby = tips[j]['tan']
            dx, dy = bx - ax, by - ay
            d = math.sqrt(dx * dx + dy * dy)
            ux, uy = dx / d, dy / d
            dot_a = max(-1.0, min(1.0, tax * ux + tay * uy))
            dot_b = max(-1.0, min(1.0, tbx * (-ux) + tby * (-uy)))
            alpha = math.acos(dot_a); beta = math.acos(dot_b)
            straight_ok = dot_a >= cos_thr and dot_b >= cos_thr
            arc_ok      = (abs(alpha - beta) <= co_circ_r
                           and dot_a >= cos_2thr and dot_b >= cos_2thr)
            if not (straight_ok or arc_ok):
                continue
            n_s = max(3, int(d) + 1); ok = True
            for ks in range(n_s):
                t  = ks / (n_s - 1)
                sx = int(round(ax + t * dx)); sy = int(round(ay + t * dy))
                if 0 <= sy < H and 0 <= sx < W:
                    if soft_f32[sy, sx] < soft_th:
                        ok = False; break
            if not ok: continue
            score = (dot_a + dot_b) * 0.5 - d / max_gap
            candidates.append((score, i, j))

    candidates.sort(reverse=True)

    consumed_tip = [False] * N
    consumed_pl  = [False] * len(polylines)
    work = [list(pl['path']) for pl in polylines]
    closed_flag = [bool(pl.get('closed', False)) for pl in polylines]
    tip_pl   = [t['pl']    for t in tips]
    tip_side = [t['side']  for t in tips]

    def _bridge(p1, p2):
        x0, y0 = p1; x1, y1 = p2
        pixels = []
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        while True:
            if (cx, cy) != p1 and (cx, cy) != p2:
                pixels.append((cx, cy))
            if cx == x1 and cy == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; cx += sx
            if e2 <= dx: err += dx; cy += sy
        return pixels

    for (_, i, j) in candidates:
        if consumed_tip[i] or consumed_tip[j]: continue
        pi, si = tip_pl[i], tip_side[i]
        pj, sj = tip_pl[j], tip_side[j]
        if pi == pj: continue
        if consumed_pl[pi] or consumed_pl[pj]: continue

        path_i = work[pi]; path_j = work[pj]
        if si == 'h':  path_i = list(reversed(path_i))
        if sj == 't':  path_j = list(reversed(path_j))

        bridge = _bridge(path_i[-1], path_j[0])
        merged = path_i + bridge + path_j

        work[pj] = merged
        consumed_pl[pi] = True

        for k in range(N):
            if consumed_tip[k]: continue
            if tip_pl[k] == pi:
                tip_pl[k]   = pj
                tip_side[k] = 'h'
            elif tip_pl[k] == pj:
                tip_side[k] = 't'
        consumed_tip[i] = True
        consumed_tip[j] = True

    return [{'path': work[i], 'closed': closed_flag[i]}
            for i in range(len(work)) if not consumed_pl[i]]


# =============================================================================
# NMS  (revived from the legacy archive in vectorize.py:310-384)
# =============================================================================

def _nms_edge_map(prob_map, smooth_sigma=1.0):
    """Non-maximum suppression on a soft edge-probability map.

    Treats the map as a ridge field, compares each pixel to its two
    neighbours along the gradient direction, and zeros pixels that are
    not local maxima.  Returns a float32 thinned map preserving the
    original values at surviving pixels.
    """
    if smooth_sigma > 0:
        k = max(3, int(np.ceil(3.0 * smooth_sigma)) | 1)
        smoothed = cv2.GaussianBlur(prob_map, (k, k), smooth_sigma)
    else:
        smoothed = prob_map

    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
    angle = np.arctan2(gy, gx)
    q = (np.round(angle / (np.pi / 4)) % 4).astype(np.int8)
    gmag = cv2.magnitude(gx, gy)
    weak_grad = gmag < 1e-3

    p = prob_map
    out = np.zeros_like(p)

    m = (q == 0)
    keep = (p >= np.roll(p, 1, axis=1)) & (p > np.roll(p, -1, axis=1))
    out[m & keep] = p[m & keep]

    m = (q == 1)
    keep = (p >= np.roll(np.roll(p, 1, axis=0), 1, axis=1)) & \
           (p >  np.roll(np.roll(p, -1, axis=0), -1, axis=1))
    out[m & keep] = p[m & keep]

    m = (q == 2)
    keep = (p >= np.roll(p, 1, axis=0)) & (p > np.roll(p, -1, axis=0))
    out[m & keep] = p[m & keep]

    m = (q == 3)
    keep = (p >= np.roll(np.roll(p, 1, axis=0), -1, axis=1)) & \
           (p >  np.roll(np.roll(p, -1, axis=0), 1, axis=1))
    out[m & keep] = p[m & keep]

    out[weak_grad] = p[weak_grad]
    return out


# =============================================================================
# Stage 2 entry points  (registered in registry.THIN)
# =============================================================================

def _build_graph_from_binary(binary_u8: np.ndarray,
                              soft_map: np.ndarray,
                              mask: np.ndarray) -> dict:
    """Helper: skeleton-decompose a binary mask and wrap as a Graph."""
    chains, nodes = skeleton_chains(binary_u8)
    return {
        'chains':   chains,
        'nodes':    nodes,
        'soft_map': soft_map,
        'mask':     mask,
    }


def thin_none(edge_map: np.ndarray, mask: np.ndarray,
              cfg: dict = None, **stage_ctx) -> dict:
    """Pass-through for Canny: threshold > 0 and chain-decompose directly.

    Canny output is already 1-pixel-wide, so no skeletonisation is needed.
    Mask is applied as an AND to restrict edges to the SAM2 subject.
    """
    cfg = cfg or THIN_NONE_DEFAULTS
    inner = _erode_mask((mask.astype(np.uint8) * 255), kernel_size=9) if mask is not None else None
    binary = (edge_map > cfg.get('threshold', 0.0)).astype(np.uint8) * 255
    if inner is not None:
        binary = cv2.bitwise_and(binary, inner)
    return _build_graph_from_binary(binary, edge_map, mask)


def thin_nms(edge_map: np.ndarray, mask: np.ndarray,
              cfg: dict = None, **stage_ctx) -> dict:
    """Non-maximum suppression → threshold → skeleton-decompose.

    Mask handling
    -------------
    The edge map that arrives here is already zeroed outside the SAM2 mask
    (edge_detect applies the mask before returning).  That hard step creates
    an artificial gradient at the mask boundary: NMS fires along it and the
    outer contour ridge is smeared / mislocated.

    Fix: dilate the mask by `mask_dilate_size` pixels before passing to NMS.
    This pushes the zero-cliff outward so the gradient computation sees a
    natural ridge at the true boundary rather than an abrupt step.  After
    thresholding, the binary result is intersected with the (uneroded) SAM2
    mask so no out-of-mask detections survive, while outer-boundary chains
    are preserved.
    """
    cfg = _resolve_cfg('nms', cfg or THIN_NMS_DEFAULTS, stage_ctx)

    # --- Pre-NMS: dilate mask to clean up gradient at boundary ---------------
    dilate_size = cfg.get('mask_dilate_size', 0)
    nms_map = edge_map.astype(np.float32)
    if mask is not None and dilate_size > 1:
        dil_k = int(dilate_size) | 1   # ensure odd
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
        nms_mask_u8 = cv2.dilate(mask.astype(np.uint8) * 255, dil_kernel,
                                  iterations=1)
        nms_map = nms_map * (nms_mask_u8 > 0).astype(np.float32)

    thinned = _nms_edge_map(nms_map, smooth_sigma=cfg.get('smooth_sigma', 1.0))

    # --- Post-NMS: intersect with (optionally eroded) SAM2 mask --------------
    erode_k = cfg.get('mask_erode_kernel', 0)
    if mask is not None:
        mask_u8 = mask.astype(np.uint8) * 255
        if erode_k > 1:
            inner = _erode_mask(mask_u8, kernel_size=erode_k)
        else:
            inner = mask_u8
    else:
        inner = None

    binary = (thinned > cfg.get('threshold', 0.05)).astype(np.uint8) * 255
    if inner is not None:
        binary = cv2.bitwise_and(binary, inner)
    return _build_graph_from_binary(binary, edge_map, mask)


def thin_zhang_suen(edge_map: np.ndarray, mask: np.ndarray,
                     cfg: dict = None, **stage_ctx) -> dict:
    """Threshold → cv2.ximgproc Zhang-Suen thinning → skeleton-decompose.

    Produces a full junction graph (multiple chains share each junction node).
    Used by DiffusionEdge by default.
    """
    cfg = _resolve_cfg('zs', cfg or THIN_ZHANG_SUEN_DEFAULTS, stage_ctx)
    inner_kernel = cfg.get('mask_erode_kernel', 9)
    inner = _erode_mask((mask.astype(np.uint8) * 255), kernel_size=inner_kernel) if mask is not None else None
    if mask is not None and inner is None:
        return _empty_graph(edge_map, mask)

    edge_float = np.clip(edge_map, 0.0, 1.0).astype(np.float32)
    binary = (edge_float > cfg.get('threshold', 0.0)).astype(np.uint8) * 255
    if inner is not None:
        binary = cv2.bitwise_and(binary, inner)
    skel = cv2.ximgproc.thinning(binary,
                                  thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return _build_graph_from_binary(skel, edge_float, mask)


def thin_steger(edge_map: np.ndarray, mask: np.ndarray,
                 cfg: dict = None, **stage_ctx) -> dict:
    """Steger sub-pixel ridge tracking + tangent linker.

    Produces a degenerate Graph: every chain has unique synthetic endpoint
    nodes; no junctions exist.  Stage 3 vectorizers see a forest of disjoint
    paths.

    Optional `edge_method` in stage_ctx selects per-edge-method overrides
    (HED enables the gradient-mask filter; NBED keeps it off).
    `bgr_frame_for_mask` is the original BGR frame used for gradient-mask
    image-gradient computation.
    """
    cfg = _resolve_cfg('steger', cfg or THIN_STEGER_DEFAULTS, stage_ctx)

    if mask is None:
        return _empty_graph(edge_map, mask)

    mask_u8 = mask.astype(np.uint8) * 255

    # Dilate mask for clean Hessian at silhouette boundary.
    dil_k = cfg['mask_dilate_size'] | 1
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
    steger_mask_u8 = cv2.dilate(mask_u8, dil_kernel, iterations=1)
    steger_bool = steger_mask_u8 > 0

    # Post-detection filter mask (optionally eroded).
    if cfg['mask_erode_size'] > 1:
        f_k = cfg['mask_erode_size']
        f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (f_k, f_k))
        filter_u8 = cv2.erode(mask_u8, f_kernel, iterations=1)
        if cv2.countNonZero(filter_u8) == 0:
            return _empty_graph(edge_map, mask)
        filter_bool = filter_u8 > 0
    else:
        filter_bool = mask.astype(bool)

    # Optional gradient-mask refinement (HED only by default).
    if cfg.get('use_gradient_mask', False):
        source = stage_ctx.get('bgr_frame_for_mask')
        if source is None:
            source = np.stack((mask_u8,) * 3, axis=-1)
        refined = _refine_with_gradient_mask(
            source, edge_map.astype(np.float32),
            gain_high=cfg['mask_gain_high'],
            gain_mid=cfg['mask_gain_mid'],
            gain_low=cfg['mask_gain_low'])
    else:
        refined = edge_map.astype(np.float32)

    H, W = refined.shape
    masked_soft = refined * steger_bool.astype(np.float32)
    ys, xs, sub_y, sub_x, tx, ty, strength = _steger_ridge_points(
        masked_soft, sigma=cfg['sigma'], strength_th=cfg['strength_th'])

    if len(ys) > 0:
        keep = filter_bool[ys, xs]
        ys = ys[keep]; xs = xs[keep]
        sub_y = sub_y[keep]; sub_x = sub_x[keep]
        tx = tx[keep]; ty = ty[keep]
        strength = strength[keep]

    if len(ys) == 0:
        return _empty_graph(edge_map, mask)

    polylines = _steger_link_polylines(
        ys, xs, sub_y, sub_x, tx, ty, strength, H, W,
        max_step_angle_deg=cfg['step_angle_deg'])

    if cfg['link_kernel'] > 1 and len(polylines) >= 2:
        for pl in polylines:
            pl['path'] = [(int(round(x)), int(round(y))) for (x, y) in pl['path']]
        polylines = _link_polyline_endpoints(
            polylines, edge_map.astype(np.float32),
            cfg['link_kernel'], cfg['link_angle_deg'],
            tangent_k=cfg['link_tangent_k'],
            soft_th=cfg['link_soft_th'],
            co_circ_deg=cfg['link_co_circ_deg'])

    # Wrap each linked polyline as a degenerate Chain with synthetic endpoint
    # nodes.  Two nodes per chain; no sharing.  Empty `nodes` dict for closed
    # polylines (Steger never produces them, but the schema allows it).
    chains = []
    nodes = {}
    for pl in polylines:
        path = pl['path']
        if len(path) < 2:
            continue
        if pl.get('closed', False):
            chains.append({
                'pixels': path,
                'node_a': None, 'node_b': None,
                'closed': True,
            })
            continue
        na = len(nodes)
        nodes[na] = (int(round(path[0][0])),  int(round(path[0][1])))
        nb = len(nodes)
        nodes[nb] = (int(round(path[-1][0])), int(round(path[-1][1])))
        chains.append({
            'pixels': path,
            'node_a': na, 'node_b': nb,
            'closed': False,
        })

    return {
        'chains':   chains,
        'nodes':    nodes,
        'soft_map': edge_map,
        'mask':     mask,
    }


def _empty_graph(edge_map, mask):
    return {'chains': [], 'nodes': {}, 'soft_map': edge_map, 'mask': mask}
