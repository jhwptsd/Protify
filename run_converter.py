"""Convert RNA sequences to protein sequences with a trained converter.

    python run_converter.py GGUAUGAGGGUAUU AAAAAAAAAA
"""
import argparse

import torch

from Converter import load_converter
from protify_pipeline import indices_to_strings, prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="*", default=["AAAAAAAAAA", "GGUAUGAGGGUAUU"],
                        help="RNA sequences (A, C, G, U)")
    parser.add_argument("--checkpoint", default="ConverterWeights/converter.pt")
    args = parser.parse_args()

    conv = load_converter(args.checkpoint, "cpu")
    conv.eval()

    x, lengths, mask = prepare_batch([s.upper().replace("T", "U") for s in args.sequences], "cpu")
    with torch.no_grad():
        aa_indices = torch.tensor(conv(x, src_key_padding_mask=mask))  # argmax indices, (batch, seq)

    for rna, protein in zip(args.sequences, indices_to_strings(aa_indices, lengths)):
        print(f"{rna}\t{protein}")


if __name__ == "__main__":
    main()
