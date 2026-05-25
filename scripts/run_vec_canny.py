#!/usr/bin/env python3
"""
Wrapper: run vectorize.py (old canny pipeline) from the laser_controller Copy.

Script    : D:/Projects/C++/laser_controller - Copy/scripts/vectorize.py
Videos    : D:/Projects/C++/laser_controller - Copy/resources/*.mp4
Masks     : D:/Projects/C++/laser_controller - Copy/resources/masks/{stem}.npz
Output    : D:/Projects/C++/laser_controller/resources/polylines/{stem}_canny.json
"""

import subprocess
import sys
import os

COPY_REPO  = r'D:\Projects\C++\laser_controller - Copy'
SCRIPT     = os.path.join(COPY_REPO, 'scripts', 'vectorize.py')
VIDEOS_DIR = os.path.join(COPY_REPO, 'resources')
MASKS_DIR  = os.path.join(COPY_REPO, 'resources', 'masks')

THIS_REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLYS_DIR  = os.path.join(THIS_REPO, 'resources', 'polylines')

if not os.path.exists(SCRIPT):
    print(f'[ERROR] Script not found: {SCRIPT}', flush=True)
    sys.exit(1)

os.makedirs(POLYS_DIR, exist_ok=True)

STEMS = ['bmx', 'car', 'sphere', 'vault']

for stem in STEMS:
    video  = os.path.join(VIDEOS_DIR, f'{stem}.mp4')
    mask   = os.path.join(MASKS_DIR,  f'{stem}.npz')
    output = os.path.join(POLYS_DIR,  f'{stem}_canny.json')

    if not os.path.exists(video):
        print(f'[skip] Video not found: {video}', flush=True)
        continue
    if not os.path.exists(mask):
        print(f'[skip] Mask not found: {mask}', flush=True)
        continue

    print(f'\n{"="*60}', flush=True)
    print(f'[run] {stem}', flush=True)
    print(f'      video  : {video}', flush=True)
    print(f'      mask   : {mask}', flush=True)
    print(f'      output : {output}', flush=True)
    print(f'{"="*60}', flush=True)

    cmd = [
        sys.executable, SCRIPT,
        '--video',  video,
        '--masks',  mask,
        '--output', output,
        '--method', 'canny',
    ]

    result = subprocess.run(cmd, cwd=COPY_REPO)
    if result.returncode != 0:
        print(f'[ERROR] {stem} failed (exit {result.returncode})', flush=True)
    else:
        print(f'[done] {stem} -> {output}', flush=True)

print('\n[run_vec_canny] All done.', flush=True)
