#!/usr/bin/env python3
"""
Parallelized block-splitting for Paris-Lille-3D training set (xyz+reflectance+labels)
Usage:
  python preprocess_parislille_reflectance_trainblocks_parallel.py \
  --input_dir  /data/AI_Dev_Data/Paris-Lille-3D/training_10_classes \
  --output_dir /data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/processed_blocks \
  --block_size 1.0 --stride 0.5 --num_point 4096 --workers 8
"""
import os
import glob
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

def split_to_blocks(coords, labels, block_size, stride, num_point):
    # brute force tiling
    xmin, ymin = coords[:, :2].min(axis=0)
    xmax, ymax = coords[:, :2].max(axis=0)
    x_starts = np.arange(xmin, xmax, stride)
    y_starts = np.arange(ymin, ymax, stride)
    for x0 in x_starts:
        for y0 in y_starts:
            x1, y1 = x0 + block_size, y0 + block_size
            mask = (
                (coords[:,0] >= x0) & (coords[:,0] < x1) &
                (coords[:,1] >= y0) & (coords[:,1] < y1)
            )
            pts = coords[mask]
            lbs = labels[mask]
            n = pts.shape[0]
            if n == 0:
                continue
            # sample exactly num_point
            if n >= num_point:
                idxs = np.random.choice(n, num_point, replace=False)
            else:
                idxs = np.random.choice(n, num_point, replace=True)
            block_pts = pts[idxs]
            block_lb  = lbs[idxs]
            yield block_pts, block_lb


def process_cloud(ply_path, output_dir, block_size, stride, num_point):
    base = os.path.splitext(os.path.basename(ply_path))[0]
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data
    coords = np.stack([v['x'], v['y'], v['z'], v['reflectance']], axis=1).astype(np.float32)
    labels = np.array(v['class'], dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for pts, lbs in split_to_blocks(coords, labels, block_size, stride, num_point):
        np.save(os.path.join(output_dir, f"{base}_block_{count:05d}_xyz.npy"), pts)
        np.save(os.path.join(output_dir, f"{base}_block_{count:05d}_labels.npy"), lbs)
        count += 1
    print(f"Written {count} blocks for {base}")


def main():
    p = argparse.ArgumentParser(description="Preprocess Paris-Lille-3D training blocks with reflectance + labels")
    p.add_argument("--input_dir",  required=True, help="raw PLY folder (training_10_classes)")
    p.add_argument("--output_dir", required=True, help="where to save processed_blocks (flat .npy files)")
    p.add_argument("--block_size", type=float, default=1.0)
    p.add_argument("--stride",     type=float, default=0.5)
    p.add_argument("--num_point",  type=int,   default=4096)
    p.add_argument("--workers",    type=int,   default=max(1, mp.cpu_count()-1))
    args = p.parse_args()

    ply_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    print(f"Found {len(ply_files)} PLYs to process, using {args.workers} workers")
    os.makedirs(args.output_dir, exist_ok=True)

    pool = mp.Pool(args.workers)
    fn = partial(
        process_cloud,
        output_dir=args.output_dir,
        block_size=args.block_size,
        stride=args.stride,
        num_point=args.num_point
    )
    # use imap to show progress
    for _ in tqdm(pool.imap_unordered(fn, ply_files), total=len(ply_files)):
        pass
    pool.close()
    pool.join()
    print("All clouds processed.")

if __name__ == '__main__':
    main()
