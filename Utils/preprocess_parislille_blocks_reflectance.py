#!/usr/bin/env python3
import os, argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData

def split_to_blocks(coords, inds, block_size, stride, num_point, min_points=100):
    xyz = coords[:, :3]
    x_min, y_min = xyz[:,0].min(), xyz[:,1].min()
    x_max, y_max = xyz[:,0].max(), xyz[:,1].max()
    xs = np.arange(x_min, x_max, stride)
    ys = np.arange(y_min, y_max, stride)

    for x0 in xs:
        for y0 in ys:
            mask = (
                (xyz[:,0] >= x0) & (xyz[:,0] < x0 + block_size) &
                (xyz[:,1] >= y0) & (xyz[:,1] < y0 + block_size)
            )
            idx_block = np.nonzero(mask)[0]
            if len(idx_block) < min_points:
                continue
            # sample exactly num_point points
            if len(idx_block) >= num_point:
                choose = np.random.choice(idx_block, num_point, replace=False)
            else:
                choose = np.random.choice(idx_block, num_point, replace=True)
            yield coords[choose], inds[choose]

def process_cloud(ply_path, out_dir, block_size, stride, num_point):
    os.makedirs(out_dir, exist_ok=True)
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data
    coords = np.stack([v['x'], v['y'], v['z'], v['reflectance']], axis=1).astype(np.float32)
    inds   = np.arange(len(coords), dtype=np.int32)
    base   = os.path.splitext(os.path.basename(ply_path))[0]
    for bid, (pts, idxs) in enumerate(tqdm(split_to_blocks(coords, inds, block_size, stride, num_point),
                                           desc=f"  blocks for {base}")):
        np.save(os.path.join(out_dir, f"{base}_block_{bid:05d}_xyz.npy"),  pts)
        np.save(os.path.join(out_dir, f"{base}_block_{bid:05d}_inds.npy"), idxs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True, help="folder of .ply files")
    parser.add_argument("--output_dir", required=True, help="where to write processed_blocks")
    parser.add_argument("--block_size", type=float, default=1.0)
    parser.add_argument("--stride",     type=float, default=0.5)
    parser.add_argument("--num_point",  type=int,   default=4096)
    args = parser.parse_args()

    ply_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.ply'))
    for fn in ply_files:
        in_ply  = os.path.join(args.input_dir, fn)
        base    = os.path.splitext(fn)[0]
        out_sub = os.path.join(args.output_dir, base, "processed_blocks")
        process_cloud(in_ply, out_sub, args.block_size, args.stride, args.num_point)

if __name__ == "__main__":
    main()
