"""Replace each residue of a protein backbone with a nucleotide, producing an all-atom RNA model.

Each nucleotide's P atom is placed on the matching C-alpha, and the nucleotide is rotated so the
normal of its P/C4'/C1' triangle matches the normal of the residue's N/CA/C backbone plane.
This is a rough prototype: it does not try to produce a physically sensible RNA.

    python substitute.py Proteintest.cif --random 68 --out generated_test.cif
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Monomer import Monomer

HEADER = """data_generated-sequence
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_alt_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def parse_backbone(path):
    """Return the C-alpha positions and unit N/CA/C plane normals of every residue (first model)."""
    from Bio.PDB import MMCIFParser, PDBParser
    parser = PDBParser(QUIET=True) if path.endswith(".pdb") else MMCIFParser(QUIET=True)
    model = next(iter(parser.get_structure("Protein", path)))

    points, normals = [], []
    for chain in model:
        for residue in chain:
            if not all(name in residue for name in ("N", "CA", "C")):
                continue
            ca = residue["CA"].coord
            normal = np.cross(residue["N"].coord - ca, residue["C"].coord - ca)
            points.append(ca)
            normals.append(normal / np.linalg.norm(normal))
    return np.array(points), np.array(normals)


def substitute(rna, protein):
    """Return mmCIF atom lines for `rna` placed along the backbone of `protein` (a structure file).
    Only as many nucleotides as there are protein residues are placed."""
    points, normals = parse_backbone(protein)
    out = ""
    length = 1
    for i in range(min(len(rna), len(points))):
        temp = Monomer("nucleotide")
        temp.load_template(rna[i])
        temp.add_name(rna[i])
        temp = temp.align_to_normal(normals[i])
        temp = temp.apply_transformation(*points[i])
        out += temp.print(start=length, number=i + 1)
        length = length + len(temp)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("protein", help="protein structure (.cif or .pdb)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", help="RNA sequence (A, C, G, U)")
    group.add_argument("--random", type=int, metavar="N", help="use a random RNA sequence of length N")
    parser.add_argument("--out", default="generated_test.cif")
    args = parser.parse_args()

    seq = args.sequence.upper() if args.sequence else [random.choice("AUGC") for _ in range(args.random)]
    with open(args.out, "w") as f:
        f.write(HEADER + substitute(seq, args.protein))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
