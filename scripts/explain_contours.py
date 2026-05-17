"""
Visual explanation of findContours and approxPolyDP.
Saves explanation.png in the project root.
"""

import numpy as np
import cv2
import math

W, H = 1600, 900
canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas[:] = (18, 18, 24)

# ── colours ──────────────────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
GRAY       = (100, 100, 110)
DIM        = (50,  50,  60)
ACCENT_B   = (80,  140, 255)   # blue
ACCENT_G   = (60,  220, 120)   # green
ACCENT_R   = (255,  80,  80)   # red
ACCENT_Y   = (255, 210,  60)   # yellow
PANEL_BG   = (26,  26,  34)

def filled_rect(img, x, y, w, h, col):
    cv2.rectangle(img, (x, y), (x+w, y+h), col, -1)

def label(img, text, x, y, scale=0.55, col=WHITE, bold=False):
    t = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2 if bold else 1
    cv2.putText(img, text, (x, y), t, scale, col, thick, cv2.LINE_AA)

def divider(img, x, y1, y2):
    cv2.line(img, (x, y1), (x, y2), (50, 50, 70), 1)

# =============================================================================
# Panel 1 (left): findContours
# =============================================================================
P1X, P1Y = 30, 30
PW, PH   = 720, 840

filled_rect(canvas, P1X, P1Y, PW, PH, PANEL_BG)
label(canvas, 'findContours', P1X+12, P1Y+34, scale=0.75, col=ACCENT_B, bold=True)
label(canvas, 'traces the boundary of every white region in a binary image,',
      P1X+12, P1Y+60, scale=0.45, col=GRAY)
label(canvas, 'returning an ordered list of boundary pixel positions.',
      P1X+12, P1Y+78, scale=0.45, col=GRAY)

# ── 1a: binary image (zoom = 18px per pixel, 16x16 grid) ─────────────────────
CELL = 22
COLS, ROWS = 20, 16
GX, GY = P1X + 20, P1Y + 110

label(canvas, 'Input: binary image  (white = edge pixel, black = background)',
      GX, GY - 10, scale=0.44, col=GRAY)

# Draw a simple thick "C" shape as the binary mask
# (represents a thresholded edge blob — wider than one pixel, like EDTER output)
mask = np.zeros((ROWS, COLS), dtype=np.uint8)
# Outer ring of C
for r in range(2, 14):
    for c in range(2, 14):
        dist_c = math.sqrt((r - 8)**2 + (c - 8)**2)
        if 4.5 <= dist_c <= 7.0:
            if not (r > 8 and c > 10):   # open the right side = C shape
                mask[r, c] = 1

for r in range(ROWS):
    for c in range(COLS):
        x0 = GX + c * CELL
        y0 = GY + r * CELL
        col = (220, 220, 220) if mask[r, c] else (30, 30, 40)
        cv2.rectangle(canvas, (x0, y0), (x0+CELL-2, y0+CELL-2), col, -1)

# Grid lines
for r in range(ROWS+1):
    cv2.line(canvas, (GX, GY+r*CELL), (GX+COLS*CELL, GY+r*CELL), DIM, 1)
for c in range(COLS+1):
    cv2.line(canvas, (GX+c*CELL, GY), (GX+c*CELL, GY+ROWS*CELL), DIM, 1)

# ── 1b: show findContours output — highlight boundary pixels ─────────────────
CX = GX + COLS*CELL + 35
label(canvas, 'Output: boundary pixels — the contour',
      CX, GY - 10, scale=0.44, col=GRAY)

mask_u8 = mask * 255
cnts, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw background grid (dimmer)
for r in range(ROWS):
    for c in range(COLS):
        x0 = CX + c * CELL
        y0 = GY + r * CELL
        col = (30, 30, 40) if mask[r, c] == 0 else (40, 60, 40)
        cv2.rectangle(canvas, (x0, y0), (x0+CELL-2, y0+CELL-2), col, -1)

# Draw contour pixels highlighted
contour_set = set()
for cnt in cnts:
    for pt in cnt:
        px, py = pt[0]
        contour_set.add((px, py))

for (px, py) in contour_set:
    if 0 <= px < COLS and 0 <= py < ROWS:
        x0 = CX + px * CELL
        y0 = GY + py * CELL
        cv2.rectangle(canvas, (x0, y0), (x0+CELL-2, y0+CELL-2), ACCENT_B, -1)

for r in range(ROWS+1):
    cv2.line(canvas, (CX, GY+r*CELL), (CX+COLS*CELL, GY+r*CELL), DIM, 1)
for c in range(COLS+1):
    cv2.line(canvas, (CX+c*CELL, GY), (CX+c*CELL, GY+ROWS*CELL), DIM, 1)

# Number order of first N contour points to show ordering
cnt0 = cnts[0]
for i, pt in enumerate(cnt0[:18]):
    px, py = pt[0]
    x0 = CX + px*CELL + 4
    y0 = GY + py*CELL + CELL - 4
    label(canvas, str(i), x0, y0, scale=0.3, col=ACCENT_Y)

# Arrow showing ordering
for i in range(min(7, len(cnt0)-1)):
    ax, ay = cnt0[i][0];   sx = CX + ax*CELL + CELL//2;  sy = GY + ay*CELL + CELL//2
    bx, by = cnt0[i+1][0]; ex = CX + bx*CELL + CELL//2;  ey = GY + by*CELL + CELL//2
    cv2.arrowedLine(canvas, (sx, sy), (ex, ey), ACCENT_Y, 1, tipLength=0.4)

# ── 1c: explanation text ──────────────────────────────────────────────────────
EY = GY + ROWS*CELL + 25
lines_1 = [
    ('How it works:', WHITE, True),
    ('  1. Scans image top-left to bottom-right looking for a white pixel', GRAY, False),
    ('     that has a black neighbour (= it is on the boundary).', GRAY, False),
    ('  2. From that start pixel it walks the boundary clockwise,', GRAY, False),
    ('     always keeping white on the left, black on the right.', GRAY, False),
    ('  3. Records every pixel visited in order until it returns', GRAY, False),
    ('     to the start — that is one contour.', GRAY, False),
    ('  4. Repeats until all white regions have been traced.', GRAY, False),
    ('', GRAY, False),
    ('CHAIN_APPROX_NONE  — stores every single boundary pixel (raw).', ACCENT_B, False),
    ('CHAIN_APPROX_SIMPLE — only stores corners of straight runs;', ACCENT_G, False),
    ('  a horizontal line of 50 pixels becomes just 2 points.', ACCENT_G, False),
    ('  We use SIMPLE — fewer points, same geometry.', ACCENT_G, False),
    ('', GRAY, False),
    ('Result: a list of (x,y) pixel positions in traversal order.', WHITE, False),
    ('The points are dense — every single boundary pixel is included.', GRAY, False),
    ('approxPolyDP (next step) drastically reduces this count.', ACCENT_Y, False),
]
for i, (txt, col, bold) in enumerate(lines_1):
    label(canvas, txt, P1X+12, EY + i*20, scale=0.43, col=col, bold=bold)

# =============================================================================
# Panel 2 (right): approxPolyDP  — Ramer-Douglas-Peucker
# =============================================================================
P2X = P1X + PW + 30
filled_rect(canvas, P2X, P1Y, PW, PH, PANEL_BG)
label(canvas, 'approxPolyDP  (Ramer-Douglas-Peucker algorithm)',
      P2X+12, P1Y+34, scale=0.75, col=ACCENT_G, bold=True)
label(canvas, 'reduces hundreds of dense contour pixels to a small polygon',
      P2X+12, P1Y+60, scale=0.45, col=GRAY)
label(canvas, 'by discarding points that lie close to the line connecting their neighbours.',
      P2X+12, P1Y+78, scale=0.45, col=GRAY)

# ── 2a: draw a noisy curve and show RDP stages ───────────────────────────────
RX, RY = P2X + 20, P1Y + 110
RW, RH  = PW - 40, 280

label(canvas, 'A dense contour (many points, noisy staircase from pixel grid):',
      RX, RY - 10, scale=0.44, col=GRAY)

# Generate a wavy shape with lots of points
def wavy_curve(n=120):
    pts = []
    for i in range(n):
        t = i / (n-1)
        x = int(RX + t * RW * 0.9 + 20)
        # Two bumps plus noise
        base_y = RY + RH//2 - int(math.sin(t * math.pi) * 90
                                  + math.sin(t * 2.5 * math.pi) * 30)
        noise  = int((hash(i*17+3) % 7) - 3)   # deterministic pixel-grid noise
        pts.append([x, base_y + noise])
    return np.array(pts, dtype=np.int32)

dense = wavy_curve(120)

# Draw dense contour
for i in range(len(dense)-1):
    cv2.line(canvas, tuple(dense[i]), tuple(dense[i+1]), (60, 80, 60), 1)
for pt in dense:
    cv2.circle(canvas, tuple(pt), 2, ACCENT_G, -1)

label(canvas, f'{len(dense)} points', RX + int(RW*0.9) + 5,
      RY + RH//2, scale=0.4, col=ACCENT_G)

# ── 2b: RDP illustration ──────────────────────────────────────────────────────
RDPY = RY + RH + 20
label(canvas, 'RDP algorithm — step by step:', RX, RDPY, scale=0.50, col=WHITE, bold=True)

# Stage 1: full span, find max distance point
S1Y = RDPY + 30
label(canvas, 'Step 1: Draw line from first to last point. Find the point furthest from it.',
      RX, S1Y - 5, scale=0.44, col=GRAY)

# Use a simpler illustration curve for the RDP steps
def simple_curve():
    pts = []
    for i in range(25):
        t = i / 24.0
        x = int(RX + 60 + t * (RW - 120))
        y = int(S1Y + 60 - math.sin(t * math.pi) * 55
                + math.sin(t * 3*math.pi) * 18)
        pts.append((x, y))
    return pts

sc = simple_curve()
A, B = sc[0], sc[-1]

# Draw the dense points lightly
for i in range(len(sc)-1):
    cv2.line(canvas, sc[i], sc[i+1], (45, 65, 45), 1)

# Draw the base line A→B
cv2.line(canvas, A, B, ACCENT_Y, 1)
cv2.circle(canvas, A, 5, WHITE, -1)
cv2.circle(canvas, B, 5, WHITE, -1)
label(canvas, 'A', A[0]-14, A[1]+5,  scale=0.45, col=WHITE)
label(canvas, 'B', B[0]+6,  B[1]+5,  scale=0.45, col=WHITE)

# Find the point with max perpendicular distance from A→B
def perp_dist(p, a, b):
    ax, ay = a;  bx, by = b;  px, py = p
    dx, dy = bx-ax, by-ay
    mag = math.sqrt(dx*dx + dy*dy)
    if mag == 0: return 0
    return abs(dx*(ay-py) - ax*(dy) + bx*(dy) - by*(dx)) / mag  # simplified
    # correct formula:
def perp_dist2(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    line_len = math.sqrt((bx-ax)**2+(by-ay)**2)
    if line_len == 0: return math.sqrt((px-ax)**2+(py-ay)**2)
    return abs((by-ay)*px - (bx-ax)*py + bx*ay - by*ax) / line_len

dists = [perp_dist2(p, A, B) for p in sc]
max_i = max(range(len(sc)), key=lambda i: dists[i])
M = sc[max_i]

# Draw perpendicular to max point
foot_t = ((M[0]-A[0])*(B[0]-A[0]) + (M[1]-A[1])*(B[1]-A[1])) / \
         max(1, (B[0]-A[0])**2 + (B[1]-A[1])**2)
foot_t = max(0, min(1, foot_t))
foot = (int(A[0] + foot_t*(B[0]-A[0])), int(A[1] + foot_t*(B[1]-A[1])))
cv2.line(canvas, M, foot, ACCENT_R, 1)
cv2.circle(canvas, M, 6, ACCENT_R, -1)
label(canvas, f'max dist = {dists[max_i]:.0f}px', M[0]+8, M[1]-8, scale=0.4, col=ACCENT_R)
label(canvas, 'epsilon', foot[0]+4, foot[1]+4, scale=0.35, col=ACCENT_Y)
cv2.arrowedLine(canvas, (foot[0]+55, foot[1]), (foot[0]+3, foot[1]),
                ACCENT_Y, 1, tipLength=0.25)

# Step 2 label
S2Y = S1Y + 150
label(canvas, 'Step 2: max dist > epsilon?  YES → keep M, split into A→M and M→B, recurse.',
      RX, S2Y - 5, scale=0.44, col=GRAY)
label(canvas, '                             NO  → discard ALL points between A and B.',
      RX, S2Y + 13, scale=0.44, col=GRAY)

# Draw split illustration
sc2 = sc[:max_i+1]
sc3 = sc[max_i:]

for i in range(len(sc2)-1): cv2.line(canvas, sc2[i], sc2[i+1], (45,65,45), 1)
for i in range(len(sc3)-1): cv2.line(canvas, sc3[i], sc3[i+1], (45,65,45), 1)
cv2.line(canvas, A, M, ACCENT_Y, 2)
cv2.line(canvas, M, B, (255,170,30), 2)
cv2.circle(canvas, A, 5, WHITE, -1)
cv2.circle(canvas, M, 6, ACCENT_R, -1)
cv2.circle(canvas, B, 5, WHITE, -1)
label(canvas, 'recurse on A→M', A[0], A[1]+18, scale=0.38, col=ACCENT_Y)
label(canvas, 'recurse on M→B', M[0]+8, M[1]+18, scale=0.38, col=(255,170,30))

# Final result
S3Y = S2Y + 100
label(canvas, 'Final result after full recursion — only the "corner" points survive:',
      RX, S3Y, scale=0.44, col=GRAY)

approx = cv2.approxPolyDP(dense.reshape(-1,1,2), 6.0, False)
approx_pts = approx.reshape(-1,2)
# Shift approx to S3Y row
offset_y = S3Y + 20 - int(np.mean(approx_pts[:,1]))
approx_shifted = [(int(p[0]), int(p[1]+offset_y)) for p in approx_pts]
dense_shifted  = [(int(p[0]), int(p[1]+offset_y)) for p in dense]

# Draw original dense lightly
for i in range(len(dense_shifted)-1):
    cv2.line(canvas, dense_shifted[i], dense_shifted[i+1], (40,60,40), 1)

# Draw approx polygon
for i in range(len(approx_shifted)-1):
    cv2.line(canvas, approx_shifted[i], approx_shifted[i+1], ACCENT_G, 2)
for pt in approx_shifted:
    cv2.circle(canvas, pt, 5, WHITE, -1)
    cv2.circle(canvas, pt, 3, ACCENT_G, -1)

label(canvas, f'{len(dense)} pts  →  {len(approx_pts)} pts   (epsilon = 6px)',
      RX + int(RW*0.5), S3Y + 30, scale=0.42, col=WHITE)

# ── explanation text ──────────────────────────────────────────────────────────
EY2 = S3Y + 110
lines_2 = [
    ('Key insight:', WHITE, True),
    ('  Points that fall nearly ON the line between their neighbours', GRAY, False),
    ('  are redundant — removing them changes the shape by less than epsilon.', GRAY, False),
    ('  Points at real corners or curves are FAR from the line — they survive.', GRAY, False),
    ('', GRAY, False),
    ('epsilon parameter (--hed-epsilon, --edter-epsilon, etc.):', ACCENT_Y, True),
    ('  Small epsilon (e.g. 1.0) = strict = more points kept = more detail', GRAY, False),
    ('  Large epsilon (e.g. 5.0) = loose = fewer points = coarser polygon', GRAY, False),
    ('  Too small: thousands of micro-segments, laser flickers', ACCENT_R, False),
    ('  Too large: real corners are discarded, shape is wrong', ACCENT_R, False),
    ('', GRAY, False),
    ('In our pipeline:', WHITE, True),
    ('  findContours gives us ~hundreds of pixels per edge blob.', GRAY, False),
    ('  approxPolyDP reduces this to ~5-20 points per stroke.', GRAY, False),
    ('  Those surviving points become the laser galvanometer waypoints.', GRAY, False),
    ('  The galvo moves in straight lines between them — which is correct,', GRAY, False),
    ('  since approxPolyDP guarantees the polygon is within epsilon of the', GRAY, False),
    ('  original contour everywhere.', GRAY, False),
]
for i, (txt, col, bold) in enumerate(lines_2):
    label(canvas, txt, P2X+12, EY2 + i*20, scale=0.43, col=col, bold=bold)

# =============================================================================
# Save
# =============================================================================
out = r'D:\Projects\C++\laser_controller\docs\contour_explanation.png'
cv2.imwrite(out, canvas)
print(f'Saved: {out}')
