import hashlib
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import *

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

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

# Parse RNA3db Sequences file tree
def parse_json(path, a, b, max_len=150):
    num = -1
    seqs = {}
    comps = []
    macros = []
    f = open(path)
    data = json.load(f)
    for i, j_dict in data.items():
        for j, k_dict in j_dict.items():
            for k, details in k_dict.items():
                num += 1
                if num > b:
                    break
                if details["length"] > max_len:
                    continue
                if a <= num <= b:
                    seqs[k] = details["sequence"]
                    comps.append(i)
                    macros.append(j)
    f.close()
    return seqs, comps, macros



def load_data(path, a=0, b=float('inf'), max_len=150):
    # Load up sequences, components, and macro-tags
    seqs, components, macro_tags=parse_json(path, a, b, max_len=max_len)
    print(f"Found {len(seqs)} usable RNA strands...")
    return seqs, components, macro_tags

def encode_rna(seq):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return torch.tensor([mapping[i] for i in seq if i in mapping], dtype=torch.float32, requires_grad=True)

def write_fastas(seqs):
    # Write a dict of {tag: seq} to as many FASTA files as needed
    for tag, seq in list(seqs.items()):
        if os.path.exists(f'FASTAs/{tag}.fasta'):
            continue
        f = open(f"FASTAs/{tag}.fasta", "w+")
        f.write(f">{tag}\n{seq}")
        f.close()

def empty_dir(path, delete=True):
    # Empty any directory
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
          os.remove(os.path.join(path, f))
        else:
          empty_dir(os.path.join(path, f))
    if delete:
      os.rmdir(path)

def RMSD(p1, p2):
    p1 = p1.clone()
    p2 = p2.clone()
    
    min_len = min(len(p1), len(p2))
    return torch.sqrt(torch.mean((p1[:min_len] - p2[:min_len])**2))

def tm_score(p1, p2, lt):
    d0 = 1.24 * (lt ** (1/3)) - 1.8
    distance = torch.norm(p1 - p2, dim=-1)
    return -torch.mean(1 / (1 + (distance / d0).pow(2)))

class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def compute_angles(self, points):
        vectors = points[1:] - points[:-1]
        
        # Compute bond angles
        dot_products = torch.sum(vectors[:-1] * vectors[1:], dim=-1)
        norms = torch.norm(vectors[:-1], dim=-1) * torch.norm(vectors[1:], dim=-1)
        bond_angles = torch.acos(dot_products / (norms + 1e-8))
        
        # Compute torsion angles
        cross_products = torch.cross(vectors[:-1], vectors[1:], dim=-1)
        dot_cross = torch.sum(cross_products[:-1] * cross_products[1:], dim=-1)
        dot_vectors = torch.sum(vectors[:-2] * cross_products[1:], dim=-1)
        torsion_angles = torch.atan2(dot_cross, dot_vectors)
        
        return bond_angles, torsion_angles

    def forward(self, pred_points, target_points):
        # Compute angles for predicted and target structures
        pred_bond_angles, pred_torsion_angles = self.compute_angles(pred_points)
        target_bond_angles, target_torsion_angles = self.compute_angles(target_points)
        
        bond_loss = F.mse_loss(pred_bond_angles, target_bond_angles)
        torsion_loss = F.mse_loss(pred_torsion_angles, target_torsion_angles)

        total_loss = bond_loss + torsion_loss
        return total_loss

def parse_rna(path, return_skips=False, pdb=False):
    parser = MMCIFParser(QUIET=True) if not pdb else PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", path)
    data = []
    nucleotides = {'A', 'U', 'C', 'G'}

    skips = []
    count = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in nucleotides:
                    for atom in residue:
                        vector = atom.get_vector()
                        data.append((vector[0], vector[1], vector[2], atom.get_name().strip()))
                else:
                    skips.append(count)
                    count += 1

    points = []
    angle_points = []
    norms = []

    correction_factor = torch.zeros(3, dtype=torch.float32, requires_grad=False)

    for x, y, z, atom in data:
        point = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True) + correction_factor

        if atom == "P":
            if torch.all(correction_factor == 0): 
                correction_factor = -point.clone()
            points.append(point)
            angle_points.append(point)

        elif atom in {"\"C1\'\"", "\"C4\'\""}:
            angle_points.append(point)

            if len(angle_points) >= 3:
                v1 = angle_points[-1] - angle_points[-2]
                v2 = angle_points[-3] - angle_points[-2]
                norms.append(torch.cross(v1, v2))

                angle_points = [] 
    if return_skips:
        return (
            torch.stack(points, dim=0) if points else torch.empty(0, 3, requires_grad=True),
            torch.stack(norms, dim=0) if norms else torch.empty(0, 3, requires_grad=True),
            skips
        )
    return (
        torch.stack(points, dim=0) if points else torch.empty(0, 3, requires_grad=True),
        torch.stack(norms, dim=0) if norms else torch.empty(0, 3, requires_grad=True)
    )

def parse_protein(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Protein", path)
    data = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    vector = atom.get_vector()
                    data.append((vector[0], vector[1], vector[2], atom.get_name()))

    points = []
    angle_points = []
    norms = []

    correction_factor = torch.zeros(3, dtype=torch.float32, requires_grad=False)

    for x, y, z, atom in data:
        point = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True) + correction_factor

        if atom == "CA":
            if torch.all(correction_factor == 0):
                correction_factor = -point.clone()
            points.append(point)
            angle_points.append(point)

        elif atom in {"N", "C"}:
            angle_points.append(point)

            if len(angle_points) >= 3:
                v1 = angle_points[-1] - angle_points[-2]
                v2 = angle_points[-3] - angle_points[-2]
                norms.append(torch.cross(v1, v2))

                angle_points = [] 

    return (
        torch.stack(points, dim=0) if points else torch.empty(0, 3, requires_grad=True),
        torch.stack(norms, dim=0) if norms else torch.empty(0, 3, requires_grad=True)
    )


     
def kabsch_algorithm(P, Q):
    P, Q = P.clone(), Q.clone()
    
    H = P.T @ Q
    U, _, Vt = torch.linalg.svd(H)
    
    Vt_copy = Vt.clone()
    if torch.det(Vt_copy.T @ U.T) < 0:
        Vt_copy[-1, :] = -Vt_copy[-1, :]

    R = Vt_copy.T @ U.T

    return P, Q @ R
     
   
def protein_to_rna(protein, rna_path, corrector, tm=False, angle=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
    if angle:
        a = AngleLoss()
        return a(prot_points,rna_points)
    prot_points = correct_protein_coords(prot_points, corrector)
    if len(prot_points)>len(rna_points):
        prot_points = prot_points[:len(rna_points)]
    elif len(rna_points)>len(prot_points):
        rna_points = rna_points[:len(prot_points)]
    prot_points, rna_points = kabsch_algorithm(prot_points, rna_points)
    if tm:
        return tm_score(prot_points, rna_points, len(rna_points))
    return RMSD(prot_points, rna_points)

def correct_protein_coords(points, corrector):
    vectors = points[1:] - points[:-1]
    norms = torch.norm(vectors, dim=1, keepdim=True)
    normalized_vectors = vectors / (norms+1e-8)
    corrected_vectors = normalized_vectors * corrector
    corrected_points = torch.cat([points[:1], points[:-1] + corrected_vectors], dim=0)
    return corrected_points