import hashlib
import json
import os
import torch
import numpy as np
import sys
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
    return torch.tensor([mapping[i] for i in seq if i in mapping], dtype=torch.float32)

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
    p1, p2 = torch.as_tensor(p1, dtype=torch.float32), torch.as_tensor(p2, dtype=torch.float32)
    min_len = min(len(p1), len(p2))
    return torch.sqrt(torch.mean((p1[:min_len] - p2[:min_len])**2))

def tm_score(p1, p2, lt):
    p1, p2 = torch.as_tensor(p1, dtype=torch.float32), torch.as_tensor(p2, dtype=torch.float32)
    d0 = 1.24 * (lt ** (1/3)) - 1.8
    distance = torch.norm(p1 - p2, dim=-1)
    return -torch.mean(1 / (1 + (distance / d0).pow(2)))

def parse_rna(path):
    parser = MMCIFParser()
    structure = parser.get_structure("RNA", path)
    data = []
    nucleotides = {'A', 'U', 'C', 'G'}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in nucleotides:
                    for atom in residue:
                        vector = atom.get_vector()
                        data.append((vector[0], vector[1], vector[2], atom.get_name()))


    points = []
    angle_points = []
    norms = []

    correction_factor = torch.zeros(3, dtype=torch.float32, requires_grad=False)

    for x, y, z, atom in data:
        x = float(x)
        y = float(y)
        z = float(z)

        point = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True) + correction_factor
        if atom == "P":
            if (correction_factor==torch.zeros(3)).all():
                correction_factor = torch.tensor([-x, -y, -z])
                points.append(point)
            angle_points.append(point)
        elif atom == "\"C1'\"":
            angle_points.append(point)
        elif atom == "\"C4'\"":
            angle_points.append(point)
            v1 = angle_points[-1]-angle_points[-2]
            v2 = angle_points[-3]-angle_points[-2]
            norms.append(torch.cross(v1, v2))
            angle_points = []

    return torch.tensor(points, requires_grad=True, dtype=torch.float32), torch.tensor(norms, requires_grad=True, dtype=torch.float32)
def parse_protein(path):
    parser = PDBParser()
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
        x = float(x)
        y = float(y)
        z = float(z)

        point = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True) + correction_factor
        if atom == "CA":
            print("adding...")
            if (correction_factor==torch.zeros(3)).all():
                correction_factor = torch.tensor([-x, -y, -z])
                points.append(point)
                angle_points.append(point)
        elif atom == "N":
            print("adding...")
            angle_points.append(point)
        elif atom == "C":
            print("adding...")
            angle_points.append(point)
            v1 = angle_points[-1]-angle_points[-2]
            v2 = angle_points[-3]-angle_points[-2]
            norms.append(torch.cross(v1, v2))
            angle_points = []

    return torch.tensor(points, requires_grad=True), torch.tensor(norms, requires_grad=True)
     
def kabsch_algorithm(P, Q):
    P, Q = torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(Q, dtype=torch.float32)
    
    H = P.T @ Q
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return P, Q @ R
     
   
def protein_to_rna(protein, rna_path, corrector, tm=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
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