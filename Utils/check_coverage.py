#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

def count_ply_vertices(ply_path):
    """Read only the header of a (binary or ASCII) PLY to get the 'element vertex' count."""
    with open(ply_path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"EOF before end_header in {ply_path}")
            try:
                text = line.decode('ascii')
            except UnicodeDecodeError:
                # skip binary portion once header is passed
                continue
            if text.startswith('element vertex'):
                return int(text.split()[2])
            if text.strip() == 'end_header':
                break
    raise RuntimeError(f"No element vertex line in {ply_path}")

def count_covered_indices(blocks_dir):
    """Union all *_inds.npy under blocks_dir and return number of unique indices."""
    covered = set()
    for fn in glob.glob(os.path.join(blocks_dir, '*_inds.npy')):
        inds = np.load(fn).ravel()
        covered.update(int(i) for i in inds)
    return len(covered)

def main(test_root):
    # 1) Find all .ply in test_root
    ply_files = sorted(glob.glob(os.path.join(test_root, '*.ply')))
    if not ply_files:
        print(f"No .ply files found in {test_root}")
        return

    # 2) Blocks actually live in test_root/processed_blocks/<cloud>/processed_blocks
    blocks_base = os.path.join(test_root, 'processed_blocks')
    if not os.path.isdir(blocks_base):
        print(f"Missing top-level processed_blocks folder: {blocks_base}")
        return

    print(f"{'cloud':<12}  total_pts   covered_pts   coverage")
    print("-"*48)
    for ply in ply_files:
        cloud = os.path.splitext(os.path.basename(ply))[0]
        total = count_ply_vertices(ply)

        blocks_dir = os.path.join(blocks_base, cloud, 'processed_blocks')
        if not os.path.isdir(blocks_dir):
            print(f"{cloud:<12}  missing blocks folder: {blocks_dir}")
            continue

        covered = count_covered_indices(blocks_dir)
        pct = covered / total * 100
        print(f"{cloud:<12}  {total:10,d}   {covered:10,d}   {pct:6.2f}%")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Check test-set coverage by processed blocks")
    p.add_argument('test_root', help="Root folder containing <cloud>.ply and processed_blocks/")
    args = p.parse_args()
    main(args.test_root)
