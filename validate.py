"""Score structure predictions against known RNA structures.

Baseline mode compares a folder of ready-made RNA structure predictions (e.g. AlphaFold or
trRosettaRNA output) with the true structures, using the P-atom trace:

    python validate.py --predicted Validation/Predicted/trRosettaRNA

Converter mode runs the full Protify pipeline instead: each true structure's sequence goes through
the converter, the resulting protein is folded with AlphaFold2, and the CA trace is scored against the
true RNA's P trace. Needs the ColabFold dependencies, and a GPU for reasonable speed:

    python validate.py --converter ConverterWeights/converter.pt
"""
import argparse
import os

import numpy as np
import torch

from Utils import (AngleLoss, NUCLEOTIDES, RMSD, correct_protein_coords, kabsch_algorithm, parse_protein,
                   parse_rna, tm_score)


def rna_sequence(path, pdb=True):
    """One-letter sequence of the A/U/C/G residues in the first model of a structure file."""
    from Bio.PDB import MMCIFParser, PDBParser
    parser = PDBParser(QUIET=True) if pdb else MMCIFParser(QUIET=True)
    model = next(iter(parser.get_structure("RNA", path)))
    return "".join(r.get_resname().strip() for chain in model for r in chain
                   if r.get_resname().strip() in NUCLEOTIDES)


def score(pred, true):
    """Superimpose two P/CA traces (truncated to the shorter) and return (RMSD, TM-score, length)."""
    n = min(len(pred), len(true))
    pred, true = kabsch_algorithm(pred[:n], true[:n])
    return RMSD(pred, true).item(), -tm_score(pred, true, n).item(), n


def find_prediction(folder, name):
    for ext in (".pdb", ".cif"):
        path = os.path.join(folder, name + ext)
        if os.path.isfile(path):
            return path
    return None


def baseline_results(actual_dir, predicted_dir):
    results = []
    for filename in sorted(os.listdir(actual_dir)):
        name, ext = os.path.splitext(filename)
        pred_path = find_prediction(predicted_dir, name)
        if pred_path is None:
            continue
        true = parse_rna(os.path.join(actual_dir, filename), pdb=ext == ".pdb")
        pred = parse_rna(pred_path, pdb=pred_path.endswith(".pdb"))
        results.append((name, *score(pred, true)))
    return results


def converter_results(actual_dir, checkpoint, pp_dist, work_dir="ValidationWork"):
    import torch.multiprocessing as mp
    from Converter import load_converter
    from protify_pipeline import FoldingWorkers, indices_to_strings, prepare_batch
    from Utils import empty_dir, write_fastas

    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = load_converter(checkpoint, device)
    conv.eval()

    names, truths, proteins = [], {}, {}
    for filename in sorted(os.listdir(actual_dir)):
        name, ext = os.path.splitext(filename)
        path = os.path.join(actual_dir, filename)
        seq = rna_sequence(path, pdb=ext == ".pdb")
        if not 0 < len(seq) <= conv.pos_encoder.max_len:
            print(f"Skipping {name}: length {len(seq)} is outside what the converter supports")
            continue
        x, lengths, mask = prepare_batch([seq], device)
        with torch.no_grad():
            indices = torch.tensor(conv(x, src_key_padding_mask=mask))
        names.append(name)
        truths[name] = parse_rna(path, pdb=ext == ".pdb")
        proteins[name] = indices_to_strings(indices, lengths)[0]

    fasta_dir = os.path.join(work_dir, "fastas")
    write_fastas(proteins, fasta_dir)
    workers = FoldingWorkers(work_dir)
    try:
        pdbs = workers.fold([os.path.join(fasta_dir, f"{n}.fasta") for n in names], names)
        results = []
        for name in names:
            pred = correct_protein_coords(parse_protein(pdbs[name]), pp_dist)
            angle = AngleLoss()(parse_protein(pdbs[name]), truths[name]).item()
            results.append((name, *score(pred, truths[name]), angle))
    finally:
        workers.close()
        empty_dir(work_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actual", default="Validation/Actual", help="folder of true structures (.pdb)")
    parser.add_argument("--predicted", default="Validation/Predicted/AlphaFold",
                        help="baseline mode: folder of predicted RNA structures (.pdb or .cif)")
    parser.add_argument("--converter", help="converter mode: path to a trained converter checkpoint")
    parser.add_argument("--pp-dist", type=float, default=None,
                        help="converter mode: P-P spacing for rescaling the CA trace (default from protify_pipeline)")
    parser.add_argument("--plot", default=None, help="save an RMSD-vs-length plot to this file")
    args = parser.parse_args()

    if args.converter:
        from protify_pipeline import DEFAULT_PP_DIST
        results = converter_results(args.actual, args.converter, args.pp_dist or DEFAULT_PP_DIST)
        print("name\tRMSD\tTM\tlength\tangle loss")
    else:
        results = baseline_results(args.actual, args.predicted)
        print("name\tRMSD\tTM\tlength")

    if not results:
        raise SystemExit("No structures were scored: check --actual and --predicted")

    for row in results:
        print(row[0] + "\t" + "\t".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row[1:]))

    rmsds = [r[1] for r in results]
    tms = [r[2] for r in results]
    lengths = [r[3] for r in results]
    print(f"Mean RMSD: {sum(rmsds) / len(rmsds):.3f}")
    print(f"Mean TM-score: {sum(tms) / len(tms):.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.scatter(lengths, rmsds, label="Data")
        if len(set(lengths)) > 1:  # a log fit needs at least two distinct lengths
            m, c = np.polyfit(np.log(lengths), rmsds, 1)
            x_fit = np.linspace(min(lengths), max(lengths), 100)
            ax.plot(x_fit, m * np.log(x_fit) + c, "r-", label="Logarithmic fit")
        ax.set_xlabel("Length (nt)")
        ax.set_ylabel("RMSD (Å)")
        ax.legend()
        fig.savefig(args.plot)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
