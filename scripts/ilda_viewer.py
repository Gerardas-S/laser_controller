#!/usr/bin/env python3
"""
ILDA Visualizer — point-by-point laser path player.

Shows every galvo move in the file: lit segments in their RGB colour,
blank (travel) moves as dim red lines.  Drag the speed slider from
1 PPS up to 30 000 PPS.  Step one point at a time when paused.

Usage
-----
    python ilda_viewer.py [file.ild / file.ilda]

    Drag-and-drop an ILDA file onto the window at any time.
    Press Ctrl+O to open a file dialog.

Keyboard
--------
    SPACE / P       play / pause
    S  or  .        step +1 point  (paused)
    ,               step -1 point  (paused)
    ]  or  →        next frame
    [  or  ←        previous frame
    ↑  or  =        speed ×2
    ↓  or  -        speed ÷2
    T               toggle Trail / Frame-accumulate display mode
    B               toggle blank-move lines on/off
    G               toggle coordinate grid
    R               restart current frame from point 0
    Home            jump to frame 0, point 0
    Ctrl+O          open file dialog
    Q  /  Escape    quit

Supported ILDA formats: 0 (3-D indexed), 1 (2-D indexed),
                        2 (palette), 4 (3-D true colour), 5 (2-D true colour)
"""

import sys, os, struct, math, argparse
from collections import deque
import pygame
from pygame.locals import (
    QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, DROPFILE,
    K_q, K_ESCAPE, K_SPACE, K_p, K_s, K_PERIOD, K_COMMA, K_t, K_b, K_g,
    K_r, K_HOME, K_o, K_RIGHT, K_LEFT, K_UP, K_DOWN,
    K_RIGHTBRACKET, K_LEFTBRACKET, K_EQUALS, K_PLUS, K_MINUS,
    KMOD_CTRL,
)

# ── window layout ────────────────────────────────────────────────────────────
W, H       = 1280, 880
CANVAS_H   = 720          # top portion: laser canvas
BAR_H      = H - CANVAS_H # bottom portion: controls

# ── colours ──────────────────────────────────────────────────────────────────
BG_CANVAS  = ( 6,   6,  10)
BG_BAR     = (16,  16,  26)
GRID_DIM   = (22,  22,  32)
GRID_AXIS  = (35,  35,  55)
BORDER_C   = (50,  50,  80)
TEXT_C     = (160, 170, 200)
TEXT_DIM   = ( 70,  72,  95)
WHITE      = (255, 255, 255)
ACCENT     = ( 70, 110, 240)
BLANK_DRAW = ( 90,  28,  28)  # dim red for blank / travel moves
NO_FILE_C  = ( 55,  58,  80)

# ── speed range ──────────────────────────────────────────────────────────────
PPS_MIN  = 1
PPS_MAX  = 30_000
_LOG_MIN = math.log10(PPS_MIN)
_LOG_MAX = math.log10(PPS_MAX)

def _t_to_pps(t: float) -> float:
    return 10 ** (_LOG_MIN + t * (_LOG_MAX - _LOG_MIN))

def _pps_to_t(pps: float) -> float:
    return (math.log10(max(PPS_MIN, pps)) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)

# ── standard 64-colour ILDA palette ──────────────────────────────────────────
ILDA_PALETTE = [
    (255,  0,  0),(255, 16,  0),(255, 32,  0),(255, 48,  0),
    (255, 64,  0),(255, 80,  0),(255, 96,  0),(255,112,  0),
    (255,128,  0),(255,144,  0),(255,160,  0),(255,176,  0),
    (255,192,  0),(255,208,  0),(255,224,  0),(255,240,  0),
    (255,255,  0),(224,255,  0),(192,255,  0),(160,255,  0),
    (128,255,  0),( 96,255,  0),( 64,255,  0),( 32,255,  0),
    (  0,255,  0),(  0,255, 32),(  0,255, 64),(  0,255, 96),
    (  0,255,128),(  0,255,160),(  0,255,192),(  0,255,224),
    (  0,255,255),(  0,224,255),(  0,192,255),(  0,160,255),
    (  0,128,255),(  0, 96,255),(  0, 64,255),(  0, 32,255),
    (  0,  0,255),( 32,  0,255),( 64,  0,255),( 96,  0,255),
    (128,  0,255),(160,  0,255),(192,  0,255),(224,  0,255),
    (255,  0,255),(255,  0,224),(255,  0,192),(255,  0,160),
    (255,  0,128),(255,  0, 96),(255,  0, 64),(255,  0, 32),
    (255,255,255),(255,210,210),(210,210,255),(210,255,210),
    (  0,  0,  0),( 80, 80, 80),(160,160,160),(255,255,255),
]

# ─────────────────────────────────────────────────────────────────────────────
# ILDA parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ilda(path: str):
    """
    Parse an ILDA file.
    Returns a list of frames; each frame is a list of point dicts:
        { x, y, r, g, b, blank }
    x, y  — signed 16-bit ILDA coordinates (-32767 … +32767)
    r,g,b — 0-255 colour (0,0,0 when blank)
    blank — True if the laser is off (galvo travel move)
    """
    frames  = []
    palette = list(ILDA_PALETTE)

    with open(path, 'rb') as f:
        while True:
            hdr = f.read(32)
            if len(hdr) < 32:
                break
            if hdr[:4] != b'ILDA':
                break

            fmt   = hdr[7]
            # Per ILDA spec Rev 011 §4.1: number-of-records lives at bytes
            # 25-26 (Python offset 24:26), NOT bytes 9-10 (which is the
            # frame-name field).  Reading the wrong offset previously read
            # the first 2 ASCII chars of the name as a huge count and the
            # parser would walk past EOF on every file.
            count = struct.unpack('>H', hdr[24:26])[0]
            # count == 0 is the ILDA EOF marker
            if count == 0:
                break

            # ── Format 2: colour palette ─────────────────────────────────
            if fmt == 2:
                new_pal = []
                for _ in range(count):
                    r, g, b = struct.unpack('BBB', f.read(3))
                    new_pal.append((r, g, b))
                palette = new_pal
                continue

            # ── Point records ────────────────────────────────────────────
            pts = []
            for _ in range(count):
                if fmt == 0:                    # 3-D indexed  — 8 bytes
                    raw    = f.read(8)
                    x, y   = struct.unpack('>hh', raw[0:4])
                    status = raw[6]
                    ci     = raw[7]
                    r, g, b = palette[ci % len(palette)]

                elif fmt == 1:                  # 2-D indexed  — 6 bytes
                    raw    = f.read(6)
                    x, y   = struct.unpack('>hh', raw[0:4])
                    status = raw[4]
                    ci     = raw[5]
                    r, g, b = palette[ci % len(palette)]

                elif fmt == 4:                  # 3-D true colour — 10 bytes
                    raw    = f.read(10)
                    x, y   = struct.unpack('>hh', raw[0:4])
                    status = raw[6]
                    b, g, r = raw[7], raw[8], raw[9]   # ILDA stores B,G,R

                elif fmt == 5:                  # 2-D true colour — 8 bytes
                    raw    = f.read(8)
                    x, y   = struct.unpack('>hh', raw[0:4])
                    status = raw[4]
                    b, g, r = raw[5], raw[6], raw[7]   # ILDA stores B,G,R

                else:
                    # Unknown format — skip (estimate 6 bytes and hope for the best)
                    f.read(6)
                    continue

                blank = bool(status & 0x40)     # bit 6 = blanking
                pts.append(dict(x=x, y=y, r=int(r), g=int(g), b=int(b), blank=blank))

            if pts:
                frames.append(pts)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Slider widget
# ─────────────────────────────────────────────────────────────────────────────

class Slider:
    TRACK_H = 6
    KNOB_R  = 9

    def __init__(self, x, y, w, t=0.5):
        self.x, self.y, self.w = x, y, w
        self.t        = float(t)
        self._drag    = False

    @property
    def pps(self) -> float:
        return _t_to_pps(self.t)

    @pps.setter
    def pps(self, v):
        self.t = _pps_to_t(v)

    def handle_event(self, e):
        if e.type == MOUSEBUTTONDOWN and e.button == 1:
            if self._hit(e.pos):
                self._drag = True
                self._update(e.pos[0])
        elif e.type == MOUSEBUTTONUP and e.button == 1:
            self._drag = False
        elif e.type == MOUSEMOTION and self._drag:
            self._update(e.pos[0])

    def _hit(self, pos):
        return (self.x - self.KNOB_R <= pos[0] <= self.x + self.w + self.KNOB_R and
                self.y - self.KNOB_R <= pos[1] <= self.y + self.KNOB_R)

    def _update(self, mx):
        self.t = max(0.0, min(1.0, (mx - self.x) / self.w))

    def draw(self, surf):
        # track
        tr = pygame.Rect(self.x, self.y - self.TRACK_H // 2, self.w, self.TRACK_H)
        pygame.draw.rect(surf, (35, 35, 55), tr, border_radius=3)
        # filled portion
        fw = max(0, int(self.w * self.t))
        if fw:
            fr = pygame.Rect(self.x, self.y - self.TRACK_H // 2, fw, self.TRACK_H)
            pygame.draw.rect(surf, ACCENT, fr, border_radius=3)
        # knob
        kx = self.x + int(self.w * self.t)
        pygame.draw.circle(surf, (200, 210, 255), (kx, self.y), self.KNOB_R)
        pygame.draw.circle(surf, ACCENT,          (kx, self.y), self.KNOB_R - 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer
# ─────────────────────────────────────────────────────────────────────────────

class Viewer:
    TRAIL_LEN  = 600   # segments kept in trail deque
    CANVAS_PAD = 0.06  # fraction of canvas edge left as margin

    def __init__(self, filepath=None):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('ILDA Visualizer')
        self.clock = pygame.time.Clock()

        self._init_fonts()
        self._init_canvas()
        self._init_state()

        if filepath:
            self.load(filepath)

    # ── init helpers ─────────────────────────────────────────────────────────

    def _init_fonts(self):
        for name in ('Consolas', 'Courier New', 'monospace'):
            try:
                self.f_sm = pygame.font.SysFont(name, 13)
                self.f_md = pygame.font.SysFont(name, 15)
                self.f_lg = pygame.font.SysFont(name, 19, bold=True)
                break
            except Exception:
                pass

    def _init_canvas(self):
        # ILDA canvas surface (redrawn every frame)
        self.canvas = pygame.Surface((W, CANVAS_H))

        # Frame-accumulate surface: we draw new segments onto it incrementally
        self.facc_surf = pygame.Surface((W, CANVAS_H))
        self.facc_surf.fill(BG_CANVAS)
        self._facc_drawn_to  = 0     # how many points are on facc_surf
        self._facc_prev_sx   = W // 2
        self._facc_prev_sy   = CANVAS_H // 2

        # Coordinate transform: ILDA [-32767,+32767] → screen pixels
        usable = min(W, CANVAS_H) * (1.0 - 2 * self.CANVAS_PAD)
        self.scale = usable / 65534.0
        self.cx    = W // 2
        self.cy    = CANVAS_H // 2

        # Speed slider
        sl_x = 130
        sl_w = W - sl_x - 200
        sl_y = CANVAS_H + 30
        # Default slider position is PPS_MAX so the simulator preview matches
        # the real hardware: HeliosOutput::CalculatePPS sends every frame at
        # max PPS, and the firmware auto-loops between RenderThread uploads.
        self.slider = Slider(sl_x, sl_y, sl_w, t=_pps_to_t(PPS_MAX))

    def _init_state(self):
        self.frames     = []
        self.filename   = ''
        self.frame_idx  = 0
        self.point_idx  = 0
        self.playing    = False
        self.accum      = 0.0

        # Trail: each entry = (x1,y1, x2,y2, r,g,b, blank)
        self.trail = deque(maxlen=self.TRAIL_LEN)
        self._prev_sx = self.cx
        self._prev_sy = self.cy

        # Display options
        self.mode        = 'trail'   # 'trail' | 'frame'
        self.show_blanks = True
        self.show_grid   = True

    # ── coordinate helpers ───────────────────────────────────────────────────

    def _to_screen(self, x, y):
        return (int(self.cx + x * self.scale),
                int(self.cy - y * self.scale))

    # ── file loading ─────────────────────────────────────────────────────────

    def load(self, path):
        try:
            frames = parse_ilda(path)
        except Exception as ex:
            print(f'[ILDA Viewer] Error loading "{path}": {ex}')
            return
        if not frames:
            print(f'[ILDA Viewer] No frames found in "{path}"')
            return
        self.frames   = frames
        self.filename = os.path.basename(path)
        pygame.display.set_caption(f'ILDA Visualizer — {self.filename}')
        self._reset()
        print(f'[ILDA Viewer] Loaded "{self.filename}" — '
              f'{len(frames)} frame(s), '
              f'{sum(len(f) for f in frames)} total points')

    def _reset(self):
        self.frame_idx = 0
        self.point_idx = 0
        self.playing   = False
        self.accum     = 0.0
        self.trail.clear()
        self._facc_dirty()
        if self.frames:
            pt = self.frames[0][0]
            self._prev_sx, self._prev_sy = self._to_screen(pt['x'], pt['y'])

    # ── playback ─────────────────────────────────────────────────────────────

    def _current_frame(self):
        if not self.frames:
            return []
        return self.frames[self.frame_idx]

    def _current_point(self):
        f = self._current_frame()
        if not f:
            return None
        return f[min(self.point_idx, len(f) - 1)]

    def _advance(self):
        """Advance one point; wraps to next frame at end of frame."""
        f = self._current_frame()
        if not f:
            return

        if self.point_idx >= len(f) - 1:
            # move to next frame
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            self.point_idx = 0
            self.trail.clear()
            self._facc_dirty()
            pt = self.frames[self.frame_idx][0]
            self._prev_sx, self._prev_sy = self._to_screen(pt['x'], pt['y'])
            return

        self.point_idx += 1
        pt = f[self.point_idx]
        sx, sy = self._to_screen(pt['x'], pt['y'])
        self.trail.append((self._prev_sx, self._prev_sy, sx, sy,
                           pt['r'], pt['g'], pt['b'], pt['blank']))
        self._prev_sx, self._prev_sy = sx, sy

    def _step_back(self):
        if not self.frames:
            return
        if self.point_idx == 0:
            # go to end of previous frame
            self.frame_idx = (self.frame_idx - 1) % len(self.frames)
            self.point_idx = len(self.frames[self.frame_idx]) - 1
            self._facc_dirty()
        else:
            self.point_idx = max(0, self.point_idx - 1)
        # Rebuild trail from frame start
        self._rebuild_trail()

    def _rebuild_trail(self):
        """Recompute trail deque up to current point."""
        self.trail.clear()
        f = self._current_frame()
        if not f or self.point_idx == 0:
            pt = f[0] if f else None
            if pt:
                self._prev_sx, self._prev_sy = self._to_screen(pt['x'], pt['y'])
            return
        start = max(0, self.point_idx - self.TRAIL_LEN)
        pt    = f[start]
        px, py = self._to_screen(pt['x'], pt['y'])
        for i in range(start + 1, self.point_idx + 1):
            pt = f[i]
            sx, sy = self._to_screen(pt['x'], pt['y'])
            self.trail.append((px, py, sx, sy,
                               pt['r'], pt['g'], pt['b'], pt['blank']))
            px, py = sx, sy
        self._prev_sx, self._prev_sy = px, py

    def _goto_frame(self, idx):
        self.frame_idx = idx % len(self.frames)
        self.point_idx = 0
        self.trail.clear()
        self._facc_dirty()
        if self.frames:
            pt = self.frames[self.frame_idx][0]
            self._prev_sx, self._prev_sy = self._to_screen(pt['x'], pt['y'])

    def update(self, dt):
        if not self.playing or not self.frames:
            return
        self.accum += dt * self.slider.pps
        # Cap how many points we advance per screen frame to avoid
        # the UI freezing on very high PPS + large frames
        steps = int(self.accum)
        steps = min(steps, 8000)
        self.accum -= steps
        for _ in range(steps):
            self._advance()

    # ── frame-accumulate surface helpers ────────────────────────────────────

    def _facc_dirty(self):
        self.facc_surf.fill(BG_CANVAS)
        self._facc_drawn_to = 0
        f = self._current_frame()
        if f:
            pt = f[0]
            self._facc_prev_sx, self._facc_prev_sy = self._to_screen(pt['x'], pt['y'])
        else:
            self._facc_prev_sx, self._facc_prev_sy = self.cx, self.cy

    def _facc_update(self):
        """Draw any points in [_facc_drawn_to, point_idx) onto facc_surf."""
        f = self._current_frame()
        if not f:
            return
        while self._facc_drawn_to < self.point_idx:
            i  = self._facc_drawn_to + 1
            pt = f[i]
            sx, sy = self._to_screen(pt['x'], pt['y'])
            if not (pt['blank'] and not self.show_blanks):
                col = BLANK_DRAW if pt['blank'] else (
                    max(pt['r'], 8), max(pt['g'], 8), max(pt['b'], 8))
                lw = 1 if pt['blank'] else 2
                pygame.draw.line(self.facc_surf, col,
                                 (self._facc_prev_sx, self._facc_prev_sy),
                                 (sx, sy), lw)
            self._facc_prev_sx, self._facc_prev_sy = sx, sy
            self._facc_drawn_to += 1

    # ── drawing ──────────────────────────────────────────────────────────────

    def _draw_grid(self, surf):
        # Minor grid lines every 8192 units
        for v in range(-32768, 32769, 8192):
            # vertical
            x1, y1 = self._to_screen(v, -32767)
            x2, y2 = self._to_screen(v,  32767)
            pygame.draw.line(surf, GRID_DIM, (x1, y1), (x2, y2))
            # horizontal
            x1, y1 = self._to_screen(-32767, v)
            x2, y2 = self._to_screen( 32767, v)
            pygame.draw.line(surf, GRID_DIM, (x1, y1), (x2, y2))

        # Centre axes (slightly brighter)
        cx_s, _ = self._to_screen(0, 0)
        _, cy_s  = self._to_screen(0, 0)
        pygame.draw.line(surf, GRID_AXIS, (cx_s, 0),        (cx_s, CANVAS_H))
        pygame.draw.line(surf, GRID_AXIS, (0,    cy_s),     (W,    cy_s))

        # ILDA boundary box
        corners = [(-32767, -32767), (32767, -32767),
                   (32767,  32767), (-32767,  32767)]
        screen_corners = [self._to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(surf, BORDER_C, screen_corners, 1)

    def _draw_trail(self, surf):
        n = len(self.trail)
        if n == 0:
            return
        for i, (x1, y1, x2, y2, r, g, b, blank) in enumerate(self.trail):
            if blank and not self.show_blanks:
                continue
            alpha = (i + 1) / n          # 0 (oldest) … 1 (newest)
            if blank:
                col = (int(BLANK_DRAW[0] * alpha),
                       int(BLANK_DRAW[1] * alpha),
                       int(BLANK_DRAW[2] * alpha))
            else:
                col = (max(8, int(r * alpha)),
                       max(8, int(g * alpha)),
                       max(8, int(b * alpha)))
            lw = 1 if blank else 2
            pygame.draw.line(surf, col, (x1, y1), (x2, y2), lw)

    def _draw_cursor(self, surf):
        pt = self._current_point()
        if not pt:
            return
        sx, sy = self._to_screen(pt['x'], pt['y'])
        col = (220, 60, 60) if pt['blank'] else WHITE
        sz  = 7
        pygame.draw.line(surf, col, (sx - sz, sy), (sx + sz, sy), 1)
        pygame.draw.line(surf, col, (sx, sy - sz), (sx, sy + sz), 1)
        pygame.draw.circle(surf, col, (sx, sy), 3)

    def _draw_no_file(self, surf):
        msgs = [
            'No ILDA file loaded.',
            '',
            'Drag & drop a .ild / .ilda file here',
            'or press Ctrl+O to browse.',
        ]
        for i, m in enumerate(msgs):
            lbl = self.f_md.render(m, True, NO_FILE_C)
            surf.blit(lbl, (W // 2 - lbl.get_width() // 2,
                             CANVAS_H // 2 - 30 + i * 22))

    def _draw_bar(self):
        # Background
        pygame.draw.rect(self.screen, BG_BAR,
                         pygame.Rect(0, CANVAS_H, W, BAR_H))
        pygame.draw.line(self.screen, (40, 42, 65),
                         (0, CANVAS_H), (W, CANVAS_H), 1)

        pps = self.slider.pps

        # ── speed label ──────────────────────────────────────────────────
        y_sl = self.slider.y
        spd_label = self.f_sm.render('SPEED', True, TEXT_DIM)
        self.screen.blit(spd_label, (14, y_sl - 7))

        self.slider.draw(self.screen)

        # PPS value (right of slider)
        if pps >= 1000:
            pps_str = f'{pps:,.0f} PPS'
        else:
            pps_str = f'{pps:.1f} PPS'
        pps_lbl = self.f_lg.render(pps_str, True, WHITE)
        self.screen.blit(pps_lbl, (W - pps_lbl.get_width() - 14, y_sl - 10))

        # ── frame / point info ────────────────────────────────────────────
        y2 = CANVAS_H + 55
        if self.frames:
            fi   = self.frame_idx + 1
            ft   = len(self.frames)
            pi   = self.point_idx + 1
            pt   = len(self._current_frame())
            pt_o = self._current_point()

            # Current point colour swatch
            if pt_o:
                swatch_col = (pt_o['r'], pt_o['g'], pt_o['b']) if not pt_o['blank'] else BLANK_DRAW
                pygame.draw.rect(self.screen, swatch_col,
                                 pygame.Rect(14, y2, 14, 14))
                pygame.draw.rect(self.screen, BORDER_C,
                                 pygame.Rect(14, y2, 14, 14), 1)

            blank_str = 'BLANK' if (pt_o and pt_o['blank']) else 'LIT  '
            play_str  = '▶ PLAY  ' if self.playing else '❙❙ PAUSE'

            info1 = (f'  Frame {fi:>4} / {ft:<4}   '
                     f'Point {pi:>5} / {pt:<5}   '
                     f'{blank_str}   {play_str}')
            self.screen.blit(self.f_md.render(info1, True, TEXT_C),
                             (14, y2))

            # Coordinate readout
            if pt_o:
                coord = (f'   X {pt_o["x"]:>7}   Y {pt_o["y"]:>7}'
                         f'   R {pt_o["r"]:>3}  G {pt_o["g"]:>3}  B {pt_o["b"]:>3}')
                self.screen.blit(self.f_sm.render(coord, True, TEXT_DIM),
                                 (14, y2 + 20))
        else:
            self.screen.blit(
                self.f_md.render('No file loaded', True, TEXT_DIM),
                (14, y2))

        # ── mode / option badges ──────────────────────────────────────────
        y3 = CANVAS_H + 98
        badges = [
            (f'[T] {"TRAIL " if self.mode=="trail" else "FRAME "}',
             ACCENT if self.mode == 'trail' else (180, 90, 30)),
            (f'[B] {"BLANKS ON " if self.show_blanks else "BLANKS OFF"}',
             (90, 170, 90) if self.show_blanks else TEXT_DIM),
            (f'[G] {"GRID ON" if self.show_grid else "GRID OFF"}',
             (90, 140, 170) if self.show_grid else TEXT_DIM),
        ]
        bx = 14
        for txt, col in badges:
            lbl = self.f_sm.render(txt, True, col)
            self.screen.blit(lbl, (bx, y3))
            bx += lbl.get_width() + 24

        # ── keyboard hints ────────────────────────────────────────────────
        y4 = CANVAS_H + 120
        hints = ('SPACE play/pause   S step+1   , step-1   '
                 '← → frame   ↑ ↓ speed   R restart frame   '
                 'Home reset   Ctrl+O open   Q quit')
        self.screen.blit(self.f_sm.render(hints, True, TEXT_DIM), (14, y4))

    def draw(self):
        self.canvas.fill(BG_CANVAS)

        if self.show_grid:
            self._draw_grid(self.canvas)

        if not self.frames:
            self._draw_no_file(self.canvas)
        elif self.mode == 'trail':
            self._draw_trail(self.canvas)
            self._draw_cursor(self.canvas)
        else:  # frame-accumulate
            self._facc_update()
            self.canvas.blit(self.facc_surf, (0, 0))
            self._draw_cursor(self.canvas)

        self.screen.blit(self.canvas, (0, 0))
        self._draw_bar()
        pygame.display.flip()

    # ── event handling ───────────────────────────────────────────────────────

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == QUIT:
                return False

            self.slider.handle_event(e)

            if e.type == DROPFILE:
                self.load(e.file)

            elif e.type == KEYDOWN:
                ctrl = pygame.key.get_mods() & KMOD_CTRL

                if e.key in (K_q, K_ESCAPE):
                    return False

                elif ctrl and e.key == K_o:
                    p = _open_dialog()
                    if p:
                        self.load(p)

                elif e.key in (K_SPACE, K_p):
                    self.playing = not self.playing
                    self.accum   = 0.0

                elif e.key in (K_s, K_PERIOD) and not self.playing:
                    self._advance()

                elif e.key == K_COMMA and not self.playing:
                    self._step_back()

                elif e.key in (K_RIGHT, K_RIGHTBRACKET) and self.frames:
                    self._goto_frame(self.frame_idx + 1)

                elif e.key in (K_LEFT, K_LEFTBRACKET) and self.frames:
                    self._goto_frame(self.frame_idx - 1)

                elif e.key in (K_UP, K_EQUALS, K_PLUS):
                    self.slider.t = min(1.0, self.slider.t + 0.07)

                elif e.key in (K_DOWN, K_MINUS):
                    self.slider.t = max(0.0, self.slider.t - 0.07)

                elif e.key == K_t:
                    self.mode = 'frame' if self.mode == 'trail' else 'trail'
                    self._facc_dirty()

                elif e.key == K_b:
                    self.show_blanks = not self.show_blanks
                    self._facc_dirty()  # rebuild frame surface without/with blanks

                elif e.key == K_g:
                    self.show_grid = not self.show_grid

                elif e.key == K_r:
                    if self.frames:
                        self.point_idx = 0
                        self.trail.clear()
                        self._facc_dirty()
                        pt = self._current_frame()[0]
                        self._prev_sx, self._prev_sy = self._to_screen(pt['x'], pt['y'])

                elif e.key == K_HOME:
                    self._reset()

        return True

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            dt      = self.clock.tick(60) / 1000.0
            running = self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# File dialog helper
# ─────────────────────────────────────────────────────────────────────────────

def _open_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title='Open ILDA File',
            filetypes=[('ILDA files', '*.ild *.ilda'), ('All files', '*.*')],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='ILDA Visualizer — point-by-point laser path player')
    ap.add_argument('file', nargs='?', help='ILDA file to open (.ild / .ilda)')
    args = ap.parse_args()

    v = Viewer(args.file)
    v.run()


if __name__ == '__main__':
    main()
