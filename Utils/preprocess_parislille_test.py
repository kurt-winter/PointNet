#!/usr/bin/env python3
"""
preprocess_parislille_test.py

Preprocess Paris-Lille-3D test_10_classes for submission.  

Given:
  /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/
     dijon_9.ply
     ajaccio_2.ply
     ajaccio_57.ply

For each PLY, this script will:
  1. Load all XYZ coordinates (N�3) from the PLY.
  2. Build a 2D grid over (x,y) using a sliding window with:
       block_size (e.g. 1.0m) and stride (e.g. 0.5m for 50% overlap).
  3. For each block center, gather all points whose (x,y) lie within
     �block_size/2 of that center.  If that block contains = min_points,
     randomly sample exactly num_point points; if it contains fewer,
     randomly repeat points with replacement to reach num_point.
  4. For each sampled block, save:
       block_#####_xyz.npy   ? shape [num_point, 3], float32  
       block_#####_inds.npy  ? shape [num_point],     int64
     The �inds� array holds the original PLY-index (0�N�1) of each 
     sampled point, so that later we can vote on the original points.

Usage:
  python preprocess_parislille_test.py \
    --input_dir  /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes \
    --output_dir /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/processed_blocks \
    --block_size 1.0 \
    --stride     0.5 \
    --num_point  4096 \
    --min_points 1024
"""

import os
import argparse
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("Preprocess Test Blocks for Paris-Lille-3D")
    parser.add_argument('--input_dir',  required=True,
                        help="Directory containing test .ply files")
    parser.add_argument('--output_dir', required=True,
                        help="Root dir where processed_blocks will be created")
    parser.add_argument('--block_size', type=float, default=1.0,
                        help="Edge length of each square block (in same units as PLY coords)")
    parser.add_argument('--stride', type=float, default=0.5,
                        help="Step size between block centers (e.g. block_size * (1-overlap))")
    parser.add_argument('--num_point', type=int, default=4096,
                        help="Number of points per block (must match training)")
    parser.add_argument('--min_points', type=int, default=1024,
                        help="Discard blocks with fewer than this many points before sampling")
    return parser.parse_args()

def load_ply_points(ply_path):
    """
    Load XYZ coordinates from a PLY file. 
    Returns: pts, a numpy array of shape (N,3) dtype=float32.
    """
    ply = PlyData.read(ply_path)
    x = np.array(ply['vertex']['x'])
    y = np.array(ply['vertex']['y'])
    z = np.array(ply['vertex']['z'])
    pts = np.vstack((x, y, z)).T.astype(np.float32)  # shape (N,3)
    return pts

def make_blocks(pts, num_point, block_size, stride, min_points):
    """
    Given pts (N,3), slide a window over (x,y), collect points per block,
    and sample exactly num_point points per block.

    Returns a list of tuples: [(block_xyz, block_inds), ...] where
      - block_xyz:  (num_point, 3) float32 array of coordinates 
      - block_inds: (num_point,) int64 array of the original global indices
    """
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create a grid of block centers along x,y
    x_centers = np.arange(x_min, x_max + 1e-6, stride, dtype=np.float32)
    y_centers = np.arange(y_min, y_max + 1e-6, stride, dtype=np.float32)

    blocks = []

    total_centers = len(x_centers) * len(y_centers)
    # Wrap nested loops in tqdm to see overall progress
    with tqdm(total=total_centers, desc="Generating blocks", unit="center") as pbar:
        for xc in x_centers:
            for yc in y_centers:
                # Find all points whose x,y lie inside the block
                x_cond = (x_coords >= xc - block_size/2) & (x_coords <= xc + block_size/2)
                y_cond = (y_coords >= yc - block_size/2) & (y_coords <= yc + block_size/2)
                mask = x_cond & y_cond
                inds = np.nonzero(mask)[0]  # global indices of points in this block
                if inds.size >= min_points:
                    # Sample exactly num_point indices (with or w/o replacement)
                    if inds.size >= num_point:
                        chosen = np.random.choice(inds, size=num_point, replace=False)
                    else:
                        chosen = np.random.choice(inds, size=num_point, replace=True)

                    block_xyz = pts[chosen, :]   # (num_point, 3)
                    blocks.append((block_xyz, chosen.astype(np.int64)))
                pbar.update(1)

    return blocks

def process_ply(ply_path, output_root, block_size, stride, num_point, min_points):
    """
    Process a single .ply file into overlapping blocks. 
    Saves each block as two files:
      block_#####_xyz.npy   (num_point � 3, float32)
      block_#####_inds.npy  (num_point, int64)
    under:
      output_root/{cloud_name}/processed_blocks/
    """
    cloud_name = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = os.path.join(output_root, cloud_name, 'processed_blocks')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing {cloud_name}.ply ? {out_dir}")
    pts = load_ply_points(ply_path)
    total_pts = pts.shape[0]
    print(f"  Total points loaded: {total_pts:,}")

    # Build blocks (with progress bar inside)
    blocks = make_blocks(pts, num_point, block_size, stride, min_points)
    print(f"  Number of valid blocks: {len(blocks):,}")

    # Save blocks with tqdm
    for i, (b_xyz, b_inds) in enumerate(tqdm(blocks, desc="Saving blocks", unit="blk")):
        np.save(os.path.join(out_dir, f"block_{i:05d}_xyz.npy"),  b_xyz)
        np.save(os.path.join(out_dir, f"block_{i:05d}_inds.npy"), b_inds)

def main():
    args = parse_args()

    ply_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.ply')])
    if not ply_files:
        print(f"No .ply files found in {args.input_dir}")
        return

    # Wrap the outer loop in tqdm so you know which .ply is being processed
    for ply_name in tqdm(ply_files, desc="All PLY files", unit="file"):
        ply_path = os.path.join(args.input_dir, ply_name)
        process_ply(
            ply_path,
            args.output_dir,
            args.block_size,
            args.stride,
            args.num_point,
            args.min_points
        )
    print(f"\nAll done. Processed blocks are under:\n  {args.output_dir}/<cloud_name>/processed_blocks/")

if __name__ == '__main__':
    main()
