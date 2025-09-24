from plyfile import PlyData
import numpy as np
import glob

def inspect_ply(path):
    ply = PlyData.read(path)
    classes = ply['vertex'].data['class']
    unique, counts = np.unique(classes, return_counts=True)
    print(f"\n--- {path} ---")
    print("Unique class IDs:", unique)
    print("Counts per class:", dict(zip(unique, counts)))

if __name__ == "__main__":
    # point this glob at a handful of your PLYs in train, val and test sets:
    files = glob.glob("/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/**/*.ply", recursive=True)
    # just sample the first 5
    for p in files[:5]:
        inspect_ply(p)