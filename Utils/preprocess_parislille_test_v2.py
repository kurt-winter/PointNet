#!/usr/bin/env python3
"""
preprocess_parislille_test.py

Preprocess Paris-Lille-3D test_10_classes for submission, now covering _every_ point.

Usage:
  python preprocess_parislille_test.py \
    --input_dir  /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes \
    --output_dir /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/processed_blocks \
    --block_size 1.0 \
    --stride     0.5 \
    --num_point  4096
"""

import os
import argparse
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir',  required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--block_size', type=float, default=1.0)
    p.add_argument('--stride',     type=float, default=0.5)
    p.add_argument('--num_point',  type=int,   default=4096)
    return p.parse_args()

def load_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    coords = np.vstack([
        ply['vertex']['x'],
        ply['vertex']['y'],
        ply['vertex']['z']
    ]).T.astype(np.float32)
    return coords

def make_blocks(pts, num_point, block_size, stride):
    x, y = pts[:,0], pts[:,1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_centers = np.arange(x_min, x_max + 1e-6, stride, dtype=np.float32)
    #nx = math.ceil((x_max - x_min - block_size) / stride) + 1
    #x_centers = x_min + np.arange(nx) * stride
    #x_centers[-1] = x_max - block_size/2

    y_centers = np.arange(y_min, y_max + 1e-6, stride, dtype=np.float32)
    #ny = math.ceil((y_max - y_min - block_size) / stride) + 1
    #y_centers = y_min + np.arange(ny) * stride
    #y_centers[-1] = y_max - block_size/2

    blocks = []
    total = len(x_centers)*len(y_centers)
    with tqdm(total=total, desc="Generating blocks") as pbar:
        for xc in x_centers:
            for yc in y_centers:
                mask = (
                    (x >= xc - block_size/2) & (x <= xc + block_size/2) &
                    (y >= yc - block_size/2) & (y <= yc + block_size/2)
                )
                inds = np.nonzero(mask)[0]
                if inds.size == 0:
                    pbar.update(1)
                    continue
                # sample exactly num_point (with or w/o replacement)
                replace = inds.size < num_point
                pick = np.random.choice(inds, size=num_point, replace=replace)
                blocks.append((pts[pick], pick.astype(np.int64)))
                pbar.update(1)
    return blocks

def process_ply(ply_path, out_root, block_size, stride, num_point):
    name = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = os.path.join(out_root, name, 'processed_blocks')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n? {name}")
    pts = load_ply_points(ply_path)
    print(f"  points: {pts.shape[0]:,}")

    blocks = make_blocks(pts, num_point, block_size, stride)
    print(f"  blocks: {len(blocks):,}")

    for i, (b_xyz, b_inds) in enumerate(tqdm(blocks, desc="Saving")):
        np.save(os.path.join(out_dir, f"block_{i:05d}_xyz.npy"),  b_xyz)
        np.save(os.path.join(out_dir, f"block_{i:05d}_inds.npy"), b_inds)

def main():
    args = parse_args()
    ply_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.ply')])
    for ply in tqdm(ply_files, desc="All PLYs"):
        process_ply(
            os.path.join(args.input_dir, ply),
            args.output_dir,
            args.block_size,
            args.stride,
            args.num_point
        )
    print("\nDone.")

if __name__ == '__main__':
    main()
