#!/usr/bin/env python3
"""
preprocess_parislille_reflectance_trainblocks_parallel_sliced.py

Split each training .ply into fixed-size blocks with:
  - coords + reflectance + class labels
    *_block_XXXXX_xyz.npy   shape [num_point,4]
    *_block_XXXXX_labels.npy shape [num_point,]

Parallelized over PLY files.

usage:
python preprocess_parislille_reflectance_trainblocks_parallel_sliced.py \
  --input_dir  /data/AI_Dev_Data/Paris-Lille-3D/training_10_classes \
  --output_dir /data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/processed_blocks \
  --block_size 1.0 --stride 0.5 --num_point 4096 --workers 32
"""
import os
import glob
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def split_to_blocks(coords, labels, block_size, stride, num_point, min_points=100):
    """
    Yield (pts_block, labels_block) for each spatial block.
    coords : (N,4) float32 array [x,y,z,reflectance]
    labels : (N,)   int32 array of class IDs
    """
    xyz = coords[:, :3]
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    xs = np.arange(x_min, x_max, stride)
    ys = np.arange(y_min, y_max, stride)

    for x0 in xs:
        for y0 in ys:
            mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + block_size) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + block_size)
            )
            idx_all = np.nonzero(mask)[0]
            n = idx_all.size
            if n < min_points:
                continue

            # sample exactly num_point indices
            if n >= num_point:
                chosen = np.random.choice(idx_all, num_point, replace=False)
            else:
                chosen = np.random.choice(idx_all, num_point, replace=True)

            yield coords[chosen], labels[chosen]


def process_one_ply(ply_path, output_dir, block_size, stride, num_point):
    """
    Read ply_path, split into blocks (coords+reflectance+class),
    write out .npy files to output_dir.
    """
    base = os.path.splitext(os.path.basename(ply_path))[0]
    out = os.path.join(output_dir, base)
    os.makedirs(out, exist_ok=True)

    # --- Read all points once ---
    ply = PlyData.read(ply_path)         # no fast=True
    v   = ply['vertex'].data
    coords = np.stack([v['x'], v['y'], v['z'], v['reflectance']], axis=1).astype(np.float32)
    labels = np.array(v['class'], dtype=np.int32)

    # --- Split and dump ---
    n_blocks = 0
    for pts, lbs in split_to_blocks(coords, labels, block_size, stride, num_point):
        name = f"{base}_block_{n_blocks:05d}"
        np.save(os.path.join(out, f"{name}_xyz.npy"), pts)
        np.save(os.path.join(out, f"{name}_labels.npy"), lbs)
        n_blocks += 1

    print(f"[{base}] wrote {n_blocks} blocks ? {out}")
    return base, n_blocks


def main():
    p = argparse.ArgumentParser(
        description="Parallel preprocess Paris-Lille-3D TRAIN .ply ? blocks (+labels)")
    p.add_argument("--input_dir",  required=True,
                   help="folder of training .ply files")
    p.add_argument("--output_dir", required=True,
                   help="where to write subfolders with *_xyz.npy and *_labels.npy")
    p.add_argument("--block_size", type=float, default=1.0)
    p.add_argument("--stride",     type=float, default=0.5)
    p.add_argument("--num_point",  type=int,   default=4096)
    p.add_argument("--workers",    type=int,   default=max(1, mp.cpu_count()-1),
                   help="number of PLYs processed in parallel")
    args = p.parse_args()

    # find all PLYs
    ply_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    if not ply_paths:
        print(f"No .ply files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(ply_paths)} PLYs ? processing with {args.workers} workers")

    # partial for pool
    fn = partial(
        process_one_ply,
        output_dir=args.output_dir,
        block_size=args.block_size,
        stride=args.stride,
        num_point=args.num_point
    )

    # process with a pool + progress bar
    with mp.Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(fn, ply_paths),
                      total=len(ply_paths),
                      desc="PLY?blocks"):
            pass

    print("? All PLYs processed.")


if __name__ == "__main__":
    main()
