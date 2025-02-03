import hashlib
import json
import os
import torch
import numpy as np
import sys
from Bio.PDB import *

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
    # Convert RNA sequence to nums to feed into Converter
    out = []
    for i in seq:
        if i=="A":
            out.append([1,0,0,0])
        elif i=="U":
            out.append([0,1,0,0])
        elif i=="C":
            out.append([0,0,1,0])
        elif i=="G":
            out.append([0,0,0,1])
    return out

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
    if len(p1)>len(p2):
      loss = torch.sqrt(torch.mean((p1[:len(p2)] - p2)**2))
    else:
      loss = torch.sqrt(torch.mean((p1 - p2[:len(p1)])**2))
    return loss

def tm_score(p1, p2, lt):
    d0 = lambda l: 1.24 * torch.power(l-15, 3) - 1.8
    loss = torch.mean(1/(1+torch.power(torch.abs(torch.norm(p1-p2))/d0(lt),2)))
    return loss

def parse_rna(path):
    try:
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

            point = np.add(np.array([x,y,z]), correction_factor)

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
                norms.append(np.cross(v1, v2))
                angle_points = []
        points = np.array(points)
        norms = np.array(norms)
        return torch.tensor(points, requires_grad=True, dtype=torch.float32), torch.tensor(norms, requires_grad=True, dtype=torch.float32)

    except Exception as e:
        print("Oops. %s" % e)
        sys.exit(1)

def parse_protein(path):
    try:
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

            point = np.add(np.array([x,y,z]), correction_factor)
            if atom == "CA":
              if (correction_factor==torch.zeros(3)).all():
                correction_factor = torch.tensor([-x, -y, -z])
              points.append(point)
              angle_points.append(point)
            elif atom == "N":
                angle_points.append(point)
            elif atom == "C":
                angle_points.append(point)
                v1 = angle_points[-1]-angle_points[-2]
                v2 = angle_points[-3]-angle_points[-2]
                norms.append(np.cross(v1, v2))
                angle_points = []

        points = np.array(points)
        norms = np.array(norms)
        return torch.tensor(points, requires_grad=True), torch.tensor(norms, requires_grad=True)

    except Exception as e:
        print("Oops. %s" % e)
        sys.exit(1)
     
def kabsch_algorithm(P, Q):
    P_original = Q.detach().numpy()
    Q_aligned = P.detach().numpy()
    Q = Q.detach().numpy()
    P = P.detach().numpy()
    
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    Q_aligned = Q_aligned @ R
    return P_original, Q_aligned
     
   
def protein_to_rna(protein, rna_path, corrector, tm=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
    prot_points = correct_protein_coords(prot_points, corrector)
    print(prot_points.size(), rna_points.size())
    prot_points, rna_points = kabsch_algorithm(prot_points, rna_points)
    if tm:
        return tm_score(prot_points, rna_points)
    return RMSD(prot_points, rna_points)

def correct_protein_coords(points, corrector):
    correction_factor = corrector.unsqueeze(0)

    # Calculate vector differences between consecutive points
    vectors = points[1:] - points[:-1]
    norms = torch.norm(vectors, dim=1, keepdim=True)
    normalized_vectors = vectors / norms

    # Apply correction factor
    corrected_vectors = normalized_vectors * correction_factor

    corrected_points = torch.zeros_like(points)
    corrected_points[0] = points[0]
    corrected_points[1:] = points[:-1] + corrected_vectors

    return corrected_points

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout