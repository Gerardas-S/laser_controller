#!/usr/bin/env python3
"""
Wrapper: run laser_controller_canny encode.py for all 4 canny polyline files.

Polylines : resources/polylines/{stem}_canny.json
Output    : resources/animations/{stem}_canny.ild
"""

import subprocess
import sys
import os

REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANNY_REPO  = os.path.join(os.path.dirname(REPO), 'laser_controller_canny')
SCRIPT      = os.path.join(CANNY_REPO, 'scripts', 'encode.py')
POLYS_DIR   = os.path.join(REPO, 'resources', 'polylines')
ANIM_DIR    = os.path.join(REPO, 'resources', 'animations')

if not os.path.exists(SCRIPT):
    print(f'[ERROR] Script not found: {SCRIPT}', flush=True)
    sys.exit(1)

os.makedirs(ANIM_DIR, exist_ok=True)

STEMS = ['bmx', 'car', 'sphere', 'vault']

for stem in STEMS:
    polylines = os.path.join(POLYS_DIR, f'{stem}_canny.json')
    output    = os.path.join(ANIM_DIR,  f'{stem}_canny.ild')

    if not os.path.exists(polylines):
        print(f'[skip] Polylines not found: {polylines}', flush=True)
        continue

    print(f'\n{"="*60}', flush=True)
    print(f'[run] {stem}', flush=True)
    print(f'      polylines : {polylines}', flush=True)
    print(f'      output    : {output}', flush=True)
    print(f'{"="*60}', flush=True)

    cmd = [
        sys.executable, SCRIPT,
        '--polylines', polylines,
        '--output',    output,
        '--exclude-outer',
    ]

    result = subprocess.run(cmd, cwd=CANNY_REPO)
    if result.returncode != 0:
        print(f'[ERROR] {stem} failed with exit code {result.returncode}', flush=True)
    else:
        print(f'[done] {stem} -> {output}', flush=True)

print('\n[run_encode_canny] All done.', flush=True)
