import numpy as np

output_txts = ["ajaccio_2.txt", "ajaccio_57.txt", "dijon_9.txt"]

for output_txt in output_txts:
    # load just one of your output txts:
    p = np.loadtxt(output_txt, dtype=int)

    print(output_txt)
    # count how many times each submission class appears
    vals, counts = np.unique(p, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"class {v:2d} ? {c} points, {100.0*c/len(p):5.2f}%")
