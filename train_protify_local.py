# Import libraries
import os
import re
import hashlib
import random
import math
import json
import jax

import sys
from sys import version_info
python_version = f"{version_info.major}.{version_info.minor}"


## MAKE SURE PIP IS INSTALLED
os.system("pip install biopython")
from Bio.PDB import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

USE_AMBER = True
USE_TEMPLATES = False
PYTHON_VERSION = python_version


# Check if necessary packages and files are downloaded
if USE_AMBER or USE_TEMPLATES:
  if not os.path.isfile("CONDA_READY"):
    print("installing conda...")
    os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh")
    os.system("bash Miniforge3-Linux-x86_64.sh -bfp /usr/local")
    os.system("mamba config --set auto_update_conda false")
    os.system("touch CONDA_READY")

if USE_TEMPLATES and not os.path.isfile("HH_READY") and USE_AMBER and not os.path.isfile("AMBER_READY"):
  print("installing hhsuite and amber...")
  os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=7.7.0 python='{PYTHON_VERSION}' pdbfixer")
  os.system("touch HH_READY")
  os.system("touch AMBER_READY")
else:
  if USE_TEMPLATES and not os.path.isfile("HH_READY"):
    print("installing hhsuite...")
    os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python='{PYTHON_VERSION}'")
    os.system("touch HH_READY")
  if USE_AMBER and not os.path.isfile("AMBER_READY"):
    print("installing amber...")
    os.system(f"mamba install -y -c conda-forge openmm=7.7.0 python='{PYTHON_VERSION}' pdbfixer")
    os.system("touch AMBER_READY")

if not os.path.exists("rna3db-mmcifs"):
  print("Downloading RNA3Db...")
  # Donwload RNA3Db structure files
  os.system('wget https://github.com/marcellszi/rna3db/releases/download/incremental-update/rna3db-mmcifs.v2.tar.xz')
  print("Extracting structure files...")
  os.system('sudo tar -xf rna3db-mmcifs.v2.tar.xz')
  os.system('rm rna3db-mmcifs.v2.tar.xz')

  # Donwload RNA3Db sequence files
  os.system('wget https://github.com/marcellszi/rna3db/releases/download/incremental-update/rna3db-jsons.tar.gz')
  print("Extracting sequence files...")
  os.system('tar -xzf rna3db-jsons.tar.gz')
  os.system('rm rna3db-jsons.tar.gz')
seq_path = "rna3db-jsons/cluster.json"
struct_path = "rna3db-mmcifs"

from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type
from colabfold.plot import plot_msa_v2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
import shutil

# Set device to CUDA and use benchmarking for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.5)
torch.backends.cudnn.benchmark = True

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

# Converter class - essentially just a Transformer model
class Converter(nn.Module):
    def __init__(self, max_seq_len=150, d_model=64, nhead=8, num_layers=6, dim_feedforward=256, dropout=0.1):
        super(Converter, self).__init__()

        self.d_model = d_model

        self.input_embedding = nn.Linear(4, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        self.transformer = nn.Transformer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_feedforward,
                                    num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        self.output_linear = nn.Linear(d_model, 20)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, src_key_padding_mask=None):
        # x shape: (seq_len, batch_size, 4)

        x = self.input_embedding(x)  # Now: (seq_len, batch_size, d_model)

        x = self.pos_encoder(x)

        x = self.transformer(x, x, src_key_padding_mask=src_key_padding_mask)

        x = self.output_linear(x)  # Now: (seq_len, batch_size, 20)
        x = self.softmax(x)

        # Convert softmaxxed matrices into one-dimensional indeces
        with torch.no_grad():
            out = []
            for i in range(len(x)):
                out.append([])
                for j in range(len(x[i])):
                    out[-1].append((torch.argmax(x[i][j].detach().cpu())).item())
        return out

# Classic positional encoder - good stuff!
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def create_padding_mask(sequences, pad_value=0):
    # sequences shape: (seq_len, batch_size, 1)
    return (sequences.squeeze(-1) == pad_value).t()  # (batch_size, seq_len)


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


seqs = {} # All sequences - may get quite large

# Used for file tree searching
components = []
macro_tags = []


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

def load_data(path, a=0, b=float('inf'), max_len=150):
    # Load up sequences, components, and macro-tags
    seqs, components, macro_tags=parse_json(path, a, b, max_len=max_len)
    print(f"Found {len(seqs)} usable RNA strands...")
    return seqs, components, macro_tags

def batch_data(iterable, n=1):
    # Random data batching function
    l = len(iterable)
    iter = [(t, s) for t, s in list(iterable.items())]
    random.shuffle(iter)
    for ndx in range(0, l, n):
        yield iter[ndx:min(ndx + n, l)]

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
      
def get_structure(tag, path):
    # Return the structure of an RNA molecule given its tag and the path to the structure directory
    # File directory:
    # root
    #  -- colabfold.dir
    #  -- train_protify.ipynb
    #  -- data.dir
    #  ---- component 1.dir
    #  ------ tag 1.dir
    #  -------- tag 1a.cif
    #  -------- tag 1b.cif
    # ...
    index = list(seqs.keys()).index(tag)
    component = components[index]
    macro_tag = macro_tags[index]

    path = f"{path}/train_set/{component}/{macro_tag}/{tag}.cif"
    return path

### Advanced settings
model_type = "auto" 
num_recycles = "1" 
recycle_early_stop_tolerance = "auto"
relax_max_iterations = 200 
pairing_strategy = "greedy"



max_msa = "auto"
num_seeds = 1 
use_dropout = True

num_recycles = None if num_recycles == "auto" else int(num_recycles)
recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
if max_msa == "auto": max_msa = None

#@markdown ### MSA options (custom MSA upload, single sequence, pairing mode)
msa_mode = "single_sequence" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
#@markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA, "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.
            
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
        
def protein_to_rna(protein, rna_path, corrector, tm=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
    prot_points = correct_protein_coords(prot_points, corrector)
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

def input_features_callback(input_features):
  pass

def prediction_callback(protein_obj, length,
                        prediction_result, input_features, mode):
  model_name, relaxed = mode
  pass

def train(seqs, epochs=50, batch_size=32,tm_score=False, max_seq_len=150, converter=None, pp_dist=6.8):
    os.makedirs("/ConverterWeights", exist_ok=True)
    os.makedirs('FASTAs', exist_ok=True)
    print("Converter Weights folder generated.")
    try:
        K80_chk = os.popen('nvidia-smi | grep "Tesla K80" | wc -l').read()
    except:
        K80_chk = "0"
        pass
    if "1" in K80_chk:
        print("WARNING: found GPU Tesla K80: limited to total length < 1000")
        if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
            del os.environ["TF_FORCE_UNIFIED_MEMORY"]
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
            del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

    # For some reason we need that to get pdbfixer to import
    if f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
        sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")

    if converter==None:
      conv = Converter(max_seq_len=max_seq_len)
    else:
      conv = converter
    conv.train()
    corrector = [nn.Parameter(torch.tensor(pp_dist, requires_grad=True, dtype=torch.float32))] # Can't be bothered to do research, so I'll just regress it
    optimizer = torch.optim.AdamW(conv.parameters(), lr=1.5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(seqs), epochs=epochs)
    dist_optimizer = torch.optim.AdamW(corrector, lr=1.5e-4)
    dist_scheduler = torch.optim.lr_scheduler.OneCycleLR(dist_optimizer, max_lr=0.01, steps_per_epoch=len(seqs), epochs=epochs)

    model_type = "alphafold2"
    download_alphafold_params(model_type, Path("."))
    for epoch in range(epochs):
        for batch in batch_data(seqs, batch_size):
            optimizer.zero_grad(set_to_none=True)
            dist_optimizer.zero_grad(set_to_none=True)
            # batch: ([(tag, seq), (tag, seq),...])

            # LAYER 1: RNA-AMINO CONVERSION
            tags = [s[0] for s in batch]
            processed_seqs = [torch.tensor(np.transpose(encode_rna(s[1]), (0, 1)), dtype=torch.float32) for s in batch] # (batch, seq, base)
            # Send sequences through the converter
            aa_seqs = [conv(s) for s in processed_seqs][0] # (seq, batch, aa)

            # Reconvert to letter representation
            aa_seqs_strings = [''.join(AA_DICT[n] for n in seq) for seq in aa_seqs]

            # Create the final dictionary using `zip` for cleaner logic
            final_seqs = dict(zip(tags, aa_seqs_strings))

            # Write the final sequences to FASTA
            write_fastas(final_seqs)

           

            num_relax = 1 #@param [0, 1, 5] {type:"raw"}
            #@markdown - specify how many of the top ranked structures to relax using amber
            template_mode = "none" #@param ["none", "pdb100","custom"]
            #@markdown - `none` = no template information is used. `pdb100` = detect templates in pdb100 (see [notes](#pdb100)). `custom` - upload and search own templates (PDB or mmCIF format, see [notes](#custom_templates))

            use_cluster_profile = True

            loss = []
            lengths = 0

            for i in tqdm(range(len(final_seqs))):
              torch.cuda.empty_cache()
              lengths=lengths+len(list(final_seqs.values())[i])
              with torch.no_grad(), torch.autocast(device_type="cuda"):
                queries, _ = get_queries(f'FASTAs/{list(final_seqs.keys())[i]}.fasta')
                jobname = add_hash(list(final_seqs.keys())[i], list(final_seqs.values())[i])
                print(jax.devices())
                results =  run(
                    queries=queries,
                    result_dir=jobname,
                    use_templates=USE_TEMPLATES,
                    custom_template_path=None,
                    num_relax=num_relax,
                    msa_mode=msa_mode,
                    model_type=model_type,
                    num_models=1,
                    num_recycles=num_recycles,
                    relax_max_iterations=relax_max_iterations,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    num_seeds=num_seeds,
                    use_dropout=use_dropout,
                    model_order=[1,2,3,4,5],
                    is_complex=False,
                    data_dir=Path("."),
                    keep_existing_results=False,
                    rank_by="auto",
                    pair_mode=pair_mode,
                    pairing_strategy=pairing_strategy,
                    stop_at_score=float(100),
                    prediction_callback=prediction_callback,
                    dpi=100,
                    zip_results=False,
                    save_all=False,
                    max_msa=max_msa,
                    use_cluster_profile=use_cluster_profile,
                    input_features_callback=input_features_callback,
                    save_recycles=False,
                    #user_agent="colabfold/google-colab-main",
                    use_gpu_relax=True,
                )
                path = ""
                for file in os.listdir(f"{jobname}"):
                  if file.endswith(".pdb"):
                    path = os.path.join(f"{jobname}", file)
                    break
              temp_loss = (protein_to_rna(path, get_structure(list(final_seqs.keys())[i], struct_path), corrector[0], tm=tm_score))

                # Download generated/actual for qualitative comp
                # shutil.copy(path, "/content/generated.pdb")
                # shutil.copy(get_structure(list(final_seqs.keys())[i], struct_path), "/content/actual.cif")
              with torch.no_grad():
                loss.append(temp_loss)
                empty_dir(f"{jobname}")

            lengths = lengths/batch_size

            empty_dir("FASTAs", delete=False)
            loss = torch.mean(torch.stack(loss))
            print(f"\n\nCurrent Loss: {loss}")
            print(f"Average Loss per Residue: {loss/lengths}")
            print(f"Correction factor: {corrector}\n\n")
            loss.backward()
            
            nn.utils.clip_grad_norm_(c.parameters(), 1.0)
            nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            
            optimizer.step()
            dist_optimizer.step()
            scheduler.step()
            dist_scheduler.step()
        torch.save(conv, f'/ConverterWeights/converter_epoch_{epoch}.pt')
        torch.save(conv.state_dict(), f'/ConverterWeights/converter_params_epoch_{epoch}.pt')
        torch.save(corrector, f'/ConverterWeights/corrector_epoch_{epoch}.pt')
        
seqs, components, macro_tags = load_data(seq_path, 0, 1645, max_len=100)

try:
   c = torch.load('/ConverterWeights/converter.pt')
   corrector = torch.load('/ConverterWeights/corrector.pt')
except:
   c = Converter(max_seq_len=200)
   corrector = [nn.Parameter(torch.tensor(6.0, requires_grad=True, dtype=torch.float32))]

c = nn.DataParallel(c)
c.to(device)

#try:
print("Training...")
train(seqs, epochs=100, batch_size=64, max_seq_len=100, converter=c, pp_dist=float(corrector[0]))
# except:
#     print("Error. Exiting training loop")
#     torch.save(c, f'/ConverterWeights/converter.pt')
#     torch.save(corrector, f'/ConverterWeights/corrector.pt')

torch.save(c, f'/ConverterWeights/converter.pt')
torch.save(corrector, f'/ConverterWeights/corrector.pt')



