"""Measure the average distance between neighbouring phosphorus atoms (the P-P step of the RNA
backbone) over RNA3DB. The result is a sensible value for --pp-dist in the training scripts,
which use it to rescale predicted protein traces when scoring with RMSD / TM-score."""
import argparse

import numpy as np
from tqdm import tqdm

from Utils import load_data, parse_rna
from protify_pipeline import SEQ_PATH, STRUCT_PATH, setup_environment, structure_paths


def mean(arr):
    total = sum(sum(i) for i in arr)
    count = sum(len(i) for i in arr)
    return total / count if count else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rnas", type=int, default=1646, help="use the first N entries of RNA3DB")
    parser.add_argument("--max-len", type=int, default=200)
    args = parser.parse_args()

    setup_environment()
    seqs, components, macro_tags = load_data(SEQ_PATH, 0, args.num_rnas - 1, max_len=args.max_len)
    paths = structure_paths(seqs, components, macro_tags, STRUCT_PATH)

    lengths = []
    for tag in tqdm(seqs):
        pts, breaks = parse_rna(paths[tag], return_breaks=True)
        pts = pts.numpy()
        # Only count steps between neighbouring residues of the same chain, not across gaps
        lengths.append([np.linalg.norm(pts[i] - pts[i + 1]) for i in range(len(pts) - 1) if not breaks[i]])
    print(mean(lengths))


if __name__ == "__main__":
    main()
