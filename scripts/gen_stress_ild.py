#!/usr/bin/env python3
"""
gen_stress_ild.py — Generate ILDA Format 5 stress-test animations.

Output (resources/animations/):
  stress_1_cube.ild        ~67  pts/frame  → ~447 fps effective
  stress_2_two_cubes.ild   ~140 pts/frame  → ~214 fps effective
  stress_3_sphere.ild      ~460 pts/frame  → ~65  fps effective
  stress_4_torus.ild       ~1250 pts/frame → ~24  fps effective
  stress_5_choke.ild       ~4000 pts/frame → ~7   fps effective (saturation)
"""

import struct, math, os

OUT_DIR  = r"D:\Projects\C++\laser_controller\resources\animations"
ILDA_MAX = 30000
N_FRAMES = 72   # 72 × 5° = one full 360° rotation


# ── ILDA I/O ─────────────────────────────────────────────────────────────────

def _hdr(name: str, company: str, n_rec: int, frm: int, total: int) -> bytes:
    nb = name.encode('ascii')[:8].ljust(8, b'\x00')
    cb = company.encode('ascii')[:8].ljust(8, b'\x00')
    return struct.pack('>4s3sB8s8sHHHBB',
        b'ILDA', b'\x00\x00\x00', 5, nb, cb,
        n_rec, frm, total, 0, 0)

def _pt(x: float, y: float, r: int, g: int, b: int,
        last: bool = False, blank: bool = False) -> bytes:
    xi = int(max(-32767, min(32767, round(x * ILDA_MAX))))
    yi = int(max(-32767, min(32767, round(y * ILDA_MAX))))
    st = (0x80 if last else 0) | (0x40 if blank else 0)
    return struct.pack('>hhBBBB', xi, yi, st, b, g, r)   # BGR order per spec

def write_ilda(path: str, frames: list) -> None:
    total = len(frames)
    name  = os.path.splitext(os.path.basename(path))[0][:8]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        for i, pts in enumerate(frames):
            f.write(_hdr(name, 'STRESS', len(pts), i, total))
            for j, (x, y, r, g, b, bl) in enumerate(pts):
                f.write(_pt(x, y, r, g, b, last=(j == len(pts)-1), blank=bl))
        f.write(_hdr('', '', 0, 0, 0))   # terminator
    ppp = sum(len(fr) for fr in frames) / total
    fps = 30000 / ppp if ppp else 0
    print(f"  {os.path.basename(path):30s}  {total} frames  "
          f"{ppp:5.0f} pts/frame  ~{fps:5.0f} eff-fps @ 30 k PPS")


# ── 3-D math ─────────────────────────────────────────────────────────────────

def ry(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(x*c+z*s, y, -x*s+z*c) for x,y,z in pts]

def rx(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(x, y*c-z*s, y*s+z*c) for x,y,z in pts]

def rz(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(x*c-y*s, x*s+y*c, z) for x,y,z in pts]

def project(pts3d, dist=2.5):
    out = []
    for x, y, z in pts3d:
        d = max(0.05, dist + z)
        out.append((x / d, y / d))
    return out

def lerp2(a, b, t):
    return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))

def off3(s, dx=0, dy=0, dz=0):
    return [(x+dx, y+dy, z+dz) for x,y,z in s]


# ── Frame builder ─────────────────────────────────────────────────────────────

class FB:
    """Accumulates laser strokes and blank travels into one ILDA frame."""
    __slots__ = ('_pts', '_cur')

    def __init__(self):
        self._pts: list = []
        self._cur = None

    def _emit(self, x, y, r, g, b, blank):
        self._pts.append((x, y, r, g, b, blank))
        self._cur = (x, y)

    def blank_to(self, p, steps=5):
        src = self._cur if self._cur is not None else (0.0, 0.0)
        if src == p:
            return
        for i in range(steps):
            t = (i + 1) / steps
            q = lerp2(src, p, t)
            self._emit(*q, 0, 0, 0, True)

    def draw(self, poly2d, r=255, g=255, b=255, spp=5):
        """
        Draw a polyline.  spp = samples per segment (both endpoints included).
        """
        if not poly2d:
            return
        if self._cur != poly2d[0]:
            self.blank_to(poly2d[0])
        self._emit(*poly2d[0], r, g, b, False)
        for i in range(len(poly2d) - 1):
            p0, p1 = poly2d[i], poly2d[i+1]
            for j in range(1, spp):
                q = lerp2(p0, p1, j / (spp - 1))
                self._emit(*q, r, g, b, False)

    def build(self):
        return list(self._pts)


# ── Shape generators ──────────────────────────────────────────────────────────

def cube_strokes(scale=1.0):
    v = [(x*scale, y*scale, z*scale)
         for x, y, z in [(-1,-1,-1),(+1,-1,-1),(+1,+1,-1),(-1,+1,-1),
                          (-1,-1,+1),(+1,-1,+1),(+1,+1,+1),(-1,+1,+1)]]
    return [
        [v[0],v[1],v[2],v[3],v[0],v[4],v[7],v[6],v[5],v[4]],
        [v[1], v[5]],
        [v[2], v[6]],
        [v[3], v[7]],
    ]

def sphere_strokes(R=1.0, n_lon=8, n_lat=6, lon_pts=12, lat_pts=16):
    out = []
    for i in range(n_lon):
        lon = 2*math.pi * i / n_lon
        pts = []
        for j in range(lon_pts):
            lat = math.pi * j / (lon_pts - 1) - math.pi/2
            pts.append((R*math.cos(lat)*math.cos(lon),
                        R*math.sin(lat),
                        R*math.cos(lat)*math.sin(lon)))
        out.append(pts)
    for j in range(1, n_lat + 1):
        lat = math.pi * j / (n_lat + 1) - math.pi/2
        pts = []
        for i in range(lat_pts + 1):
            lon = 2*math.pi * i / lat_pts
            pts.append((R*math.cos(lat)*math.cos(lon),
                        R*math.sin(lat),
                        R*math.cos(lat)*math.sin(lon)))
        out.append(pts)
    return out

def torus_strokes(R=1.0, r=0.3, n_cross=10, n_long=12,
                  cross_pts=14, long_pts=16):
    out = []
    for i in range(n_cross):
        phi = 2*math.pi * i / n_cross
        cp, sp = math.cos(phi), math.sin(phi)
        pts = []
        for j in range(cross_pts + 1):
            th = 2*math.pi * j / cross_pts
            pts.append(((R + r*math.cos(th))*cp,
                         r*math.sin(th),
                        (R + r*math.cos(th))*sp))
        out.append(pts)
    for j in range(n_long):
        th = 2*math.pi * j / n_long
        ct, st = math.cos(th), math.sin(th)
        pts = []
        for i in range(long_pts + 1):
            phi = 2*math.pi * i / long_pts
            pts.append(((R + r*ct)*math.cos(phi),
                         r*st,
                        (R + r*ct)*math.sin(phi)))
        out.append(pts)
    return out


# ── Scene functions ───────────────────────────────────────────────────────────

TILT = 0.40   # ~23° X tilt for all scenes

def scene_1_cube(angle):
    """Single white rotating cube."""
    fb = FB()
    for s in cube_strokes(0.30):
        fb.draw(project(ry(rx(s, TILT), angle)), 255, 255, 255, spp=5)
    return fb.build()

def scene_2_two_cubes(angle):
    """Two counter-rotating cubes (red left, cyan right)."""
    fb = FB()
    S = 0.25
    for s in cube_strokes(S):
        fb.draw(project(ry(rx(off3(s, dx=-0.35), TILT),  angle)),
                255, 60, 60, spp=5)
    for s in cube_strokes(S):
        fb.draw(project(ry(rx(off3(s, dx=+0.35), TILT), -angle)),
                60, 255, 255, spp=5)
    return fb.build()

def scene_3_sphere(angle):
    """Sphere wireframe — cyan-green."""
    fb = FB()
    for s in sphere_strokes(R=0.38, n_lon=8, n_lat=6,
                             lon_pts=12, lat_pts=16):
        fb.draw(project(ry(rx(s, TILT), angle)), 50, 255, 180, spp=3)
    return fb.build()

def scene_4_torus(angle):
    """Sphere + torus."""
    fb = FB()
    # Sphere (blue-white)
    for s in sphere_strokes(R=0.32, n_lon=8, n_lat=6,
                             lon_pts=12, lat_pts=16):
        fb.draw(project(ry(rx(s, TILT), angle)), 150, 200, 255, spp=3)
    # Torus (orange) — tilted on its side, 1.5× rotation speed
    for s in torus_strokes(R=0.50, r=0.14, n_cross=10, n_long=12,
                            cross_pts=14, long_pts=16):
        fb.draw(project(ry(rx(s, TILT + 0.9), angle * 1.5)),
                255, 160, 40, spp=3)
    return fb.build()

def scene_5_choke(angle):
    """System saturation: dense sphere + dense torus + orbiting sphere + second torus."""
    fb = FB()

    # Central dense sphere (yellow)
    for s in sphere_strokes(R=0.30, n_lon=14, n_lat=10,
                             lon_pts=16, lat_pts=20):
        fb.draw(project(ry(rx(s, TILT), angle)), 255, 255, 80, spp=4)

    # Large dense torus (magenta) — faster spin
    for s in torus_strokes(R=0.52, r=0.16, n_cross=14, n_long=16,
                            cross_pts=16, long_pts=20):
        fb.draw(project(ry(rx(s, TILT + 1.1), angle * 0.7)),
                255, 60, 255, spp=4)

    # Orbiting small sphere (green) — 2× orbital speed
    ox = 0.58 * math.cos(angle * 2)
    oz = 0.58 * math.sin(angle * 2)
    for s in sphere_strokes(R=0.14, n_lon=6, n_lat=4,
                             lon_pts=10, lat_pts=12):
        fb.draw(project(ry(rx(off3(s, dx=ox, dz=oz), TILT), angle)),
                80, 255, 80, spp=3)

    # Second torus (blue) — tilted vertically, counter-spin
    for s in torus_strokes(R=0.38, r=0.11, n_cross=10, n_long=12,
                            cross_pts=12, long_pts=14):
        p = rz(s, angle * 1.3)
        fb.draw(project(ry(rx(p, TILT + 0.5), -angle * 0.9)),
                60, 140, 255, spp=3)

    return fb.build()


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(fname, scene_fn):
    angles = [2 * math.pi * f / N_FRAMES for f in range(N_FRAMES)]
    frames = [scene_fn(a) for a in angles]
    write_ilda(os.path.join(OUT_DIR, fname), frames)

if __name__ == '__main__':
    print("Generating ILDA Format 5 stress-test animations...\n")
    generate("stress_1_cube.ild",       scene_1_cube)
    generate("stress_2_two_cubes.ild",  scene_2_two_cubes)
    generate("stress_3_sphere.ild",     scene_3_sphere)
    generate("stress_4_torus.ild",      scene_4_torus)
    generate("stress_5_choke.ild",      scene_5_choke)
    print("\nDone.")
