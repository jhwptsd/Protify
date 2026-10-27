import hashlib
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import MMCIFParser, PDBParser

# Index to amino acid dictionary
# Largely arbitrary, but must stay consistent for any given converter
AA_DICT = {
    0: "A",
    1: "R",
    2: "N",
    3: "D",
    4: "C",
    5: "Q",
    6: "E",
    7: "G",
    8: "H",
    9: "I",
    10: "L",
    11: "K",
    12: "M",
    13: "F",
    14: "P",
    15: "S",
    16: "T",
    17: "W",
    18: "Y",
    19: "V"
}

# Residue names counted as RNA nucleotides when reading structures
NUCLEOTIDES = {'A', 'U', 'C', 'G'}

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

# Parse RNA3db Sequences file tree
def parse_json(path, a, b, max_len=150):
    # Entries are numbered in file order; entries a..b (inclusive) are kept, minus any longer than max_len
    num = -1
    seqs = {}
    comps = []
    macros = []
    with open(path) as f:
        data = json.load(f)
    for i, j_dict in data.items():
        for j, k_dict in j_dict.items():
            for k, details in k_dict.items():
                num += 1
                if num > b:
                    return seqs, comps, macros
                if num < a or details["length"] > max_len:
                    continue
                seqs[k] = details["sequence"]
                comps.append(i)
                macros.append(j)
    return seqs, comps, macros



def load_data(path, a=0, b=float('inf'), max_len=150):
    # Load up sequences, components, and macro-tags
    seqs, components, macro_tags=parse_json(path, a, b, max_len=max_len)
    print(f"Found {len(seqs)} usable RNA strands...")
    return seqs, components, macro_tags

def encode_rna(seq):
    # One-hot encode an RNA sequence. Characters other than A/U/C/G are dropped, which matches
    # parse_rna (it only reads A/U/C/G residues), so lengths stay comparable with the structure.
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return torch.tensor([mapping[i] for i in seq if i in mapping], dtype=torch.float32)

def write_fastas(seqs, directory="FASTAs"):
    # Write a dict of {tag: seq} to as many FASTA files as needed
    os.makedirs(directory, exist_ok=True)
    for tag, seq in list(seqs.items()):
        path = os.path.join(directory, f"{tag}.fasta")
        if os.path.exists(path):
            continue
        with open(path, "w") as f:
            f.write(f">{tag}\n{seq}\n")

def empty_dir(path, delete=True):
    # Empty any directory
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
    if delete:
      os.rmdir(path)

def RMSD(p1, p2):
    # Root-mean-square deviation between corresponding points (truncated to the shorter set)
    min_len = min(len(p1), len(p2))
    return torch.sqrt(torch.mean(torch.sum((p1[:min_len] - p2[:min_len])**2, dim=-1)))

def rna_d0(lt):
    # TM-score length normalisation for RNA (the one used by US-align)
    if lt < 12:
        return 0.3
    if lt < 16:
        return 0.4
    if lt < 20:
        return 0.5
    if lt < 24:
        return 0.6
    if lt < 30:
        return 0.7
    return 0.6 * (lt - 0.5) ** 0.5 - 2.5

def tm_score(p1, p2, lt):
    # Negative TM-score between two aligned point sets of length lt (so lower is better, like a loss).
    # Assumes residue i of p1 corresponds to residue i of p2; there is no sequence alignment.
    d0 = rna_d0(lt)
    distance = torch.norm(p1 - p2, dim=-1)
    return -torch.mean(1 / (1 + (distance / d0).pow(2)))

class AngleLoss(nn.Module):
    # Compares the virtual bond angles and torsions of two C-alpha / P traces.
    # Both are invariant to scale and rigid motion, so no alignment is needed.
    def __init__(self):
        super(AngleLoss, self).__init__()

    def compute_angles(self, points):
        vectors = points[1:] - points[:-1]

        # Bond angles: angle between consecutive virtual bonds (in [0, pi])
        dot_products = torch.sum(vectors[:-1] * vectors[1:], dim=-1)
        norms = torch.norm(vectors[:-1], dim=-1) * torch.norm(vectors[1:], dim=-1)
        cosines = (dot_products / (norms + 1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)
        bond_angles = torch.acos(cosines)

        # Torsion (dihedral) angles over each run of three consecutive bonds, in (-pi, pi]
        b1, b2, b3 = vectors[:-2], vectors[1:-1], vectors[2:]
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        m1 = torch.cross(n1, b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8), dim=-1)
        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(m1 * n2, dim=-1)
        torsion_angles = torch.atan2(y, x)

        return bond_angles, torsion_angles

    def forward(self, pred_points, target_points):
        # Compute angles for predicted and target structures
        pred_bond_angles, pred_torsion_angles = self.compute_angles(pred_points)
        target_bond_angles, target_torsion_angles = self.compute_angles(target_points)
        min_len_1 = min(len(pred_bond_angles), len(target_bond_angles))
        min_len_2 = min(len(pred_torsion_angles), len(target_torsion_angles))

        total_loss = pred_points.sum() * 0  # zero with the right dtype/device for very short chains
        if min_len_1 > 0:
            total_loss = total_loss + F.mse_loss(pred_bond_angles[:min_len_1], target_bond_angles[:min_len_1])
        if min_len_2 > 0:
            # 1 - cos(difference) respects the periodicity of torsion angles
            diff = pred_torsion_angles[:min_len_2] - target_torsion_angles[:min_len_2]
            total_loss = total_loss + torch.mean(1 - torch.cos(diff))
        return total_loss

def _atom_name(atom):
    # mmCIF atom names such as C1' may arrive wrapped in double quotes depending on the parser
    return atom.get_name().strip().strip('"')

def _parser_for(path, pdb):
    return PDBParser(QUIET=True) if pdb else MMCIFParser(QUIET=True)

def _first_model(structure):
    return next(iter(structure))

def parse_rna(path, pdb=False, return_breaks=False):
    """Return the phosphorus (P) trace of an RNA structure as an (N, 3) tensor, with the first
    P atom moved to the origin. Only the first model is read, and residues without a P atom
    (e.g. the 5' end) are skipped.

    With return_breaks=True, also return a boolean list of length N-1 whose entry i is True when
    P atom i and P atom i+1 are not neighbouring residues of the same chain."""
    structure = _parser_for(path, pdb).get_structure("RNA", path)

    points = []
    breaks = []
    prev = None  # (chain id, residue number) of the previous P atom
    for chain in _first_model(structure):
        for residue in chain:
            if residue.get_resname().strip() not in NUCLEOTIDES:
                continue
            p_atom = next((a for a in residue if _atom_name(a) == "P"), None)
            if p_atom is None:
                continue
            key = (chain.id, residue.id[1])
            if prev is not None:
                breaks.append(not (key[0] == prev[0] and key[1] - prev[1] == 1))
            prev = key
            points.append(p_atom.coord)

    points = _to_tensor(points)
    if return_breaks:
        return points, breaks
    return points

def parse_protein(path):
    """Return the C-alpha trace of a protein structure (PDB format) as an (N, 3) tensor, with the
    first C-alpha moved to the origin. Only the first model is read."""
    structure = PDBParser(QUIET=True).get_structure("Protein", path)

    points = []
    for chain in _first_model(structure):
        for residue in chain:
            if "CA" in residue:
                points.append(residue["CA"].coord)

    return _to_tensor(points)

def _to_tensor(points):
    if not points:
        return torch.empty(0, 3, dtype=torch.float32)
    points = torch.tensor(np.array(points), dtype=torch.float32)
    return points - points[0]

def kabsch_algorithm(P, Q):
    # Center both point sets, then rotate Q onto P (optimal in the least-squares sense).
    # Returns the centered P and the rotated, centered Q.
    P_centroid = torch.mean(P, dim=0)
    Q_centroid = torch.mean(Q, dim=0)
    P = P - P_centroid
    Q = Q - Q_centroid

    H = P.T @ Q
    U, _, Vt = torch.linalg.svd(H)

    # Flip the last axis if needed so that R is a proper rotation, not a reflection
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vt.T @ D @ U.T

    return P, Q @ R

def protein_to_rna(protein, rna_path, corrector, tm=False, angle=False):
    # Loss between a predicted protein (PDB file) and the true RNA structure (mmCIF file).
    #   angle=True: compare virtual bond/torsion angles of the CA and P traces (no alignment needed)
    #   otherwise:  rescale the CA trace to P-P spacing, superimpose, and return RMSD (or -TM-score if tm)
    prot_points = parse_protein(protein)
    rna_points = parse_rna(rna_path)
    if angle:
        return AngleLoss()(prot_points, rna_points)
    prot_points = correct_protein_coords(prot_points, corrector)
    min_len = min(len(prot_points), len(rna_points))
    prot_points, rna_points = kabsch_algorithm(prot_points[:min_len], rna_points[:min_len])
    if tm:
        return tm_score(prot_points, rna_points, min_len)
    return RMSD(prot_points, rna_points)

def correct_protein_coords(points, corrector):
    # Rebuild the trace with every consecutive-point step rescaled to length `corrector`,
    # keeping the direction of each step (CA-CA is ~3.8A, P-P is ~6A).
    if len(points) < 2:
        return points
    steps = F.normalize(points[1:] - points[:-1], dim=1) * corrector
    return torch.cat([points[:1], points[:1] + torch.cumsum(steps, dim=0)], dim=0)
