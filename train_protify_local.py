# Import libraries
import importlib
import os
import sys
from sys import version_info

import torch.multiprocessing as mp

import torch.utils
import torch.utils.data
python_version = f"{version_info.major}.{version_info.minor}"


## MAKE SURE PIP IS INSTALLED
os.system("pip install biopython")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

USE_AMBER = True
USE_TEMPLATES = False
PYTHON_VERSION = python_version

TF_FORCE_UNIFIED_MEMORY = 1


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

from colabfold.download import download_alphafold_params
from colabfold.batch import get_queries, run
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm

# Set device to CUDA and use benchmarking for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from Converter import Converter
from Utils import *

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
    index = list(old_seqs.keys()).index(tag)
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

def input_features_callback(input_features):
  pass

def prediction_callback(protein_obj, length,
                        prediction_result, input_features, mode):
  model_name, relaxed = mode
  pass

class SeqDataset(torch.utils.data.Dataset):
   def __init__(self, seqs):
      self.seqs = seqs
    
   def __len__(self):
      return len(self.seqs)
   
   def __getitem__(self, idx):
      return list(self.seqs.values())[idx], list(self.seqs.keys())[idx] # (seq, tag)

def train(seqs, epochs=50, batch_size=32,tm_score=False, max_seq_len=150, converter=None, pp_dist=6.8):

    os.makedirs("/ConverterWeights", exist_ok=True)
    os.makedirs('FASTAs', exist_ok=True)

    num_gpus = torch.cuda.device_count()

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
      conv = Converter(max_seq_len=max_seq_len).to(device)
    else:
      conv = converter
    conv.train()
    corrector = [nn.Parameter(torch.tensor(pp_dist, requires_grad=True, dtype=torch.float32)).to(device)] # Can't be bothered to do research, so I'll just regress it
    optimizer = torch.optim.AdamW(conv.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(seqs), epochs=epochs)
    dist_optimizer = torch.optim.AdamW(corrector, lr=0.001)
    dist_scheduler = torch.optim.lr_scheduler.OneCycleLR(dist_optimizer, max_lr=0.01, steps_per_epoch=len(seqs), epochs=epochs)

    dataloader = torch.utils.data.DataLoader(seqs, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    model_type = "alphafold2"
    download_alphafold_params(model_type, Path("."))
    for epoch in range(epochs):
        for seqs, tags in dataloader:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            optimizer.zero_grad(set_to_none=True)
            dist_optimizer.zero_grad(set_to_none=True)
            # batch: ([(tag, seq), (tag, seq),...])

            # LAYER 1: RNA-AMINO CONVERSION
            processed_seqs = [torch.tensor(np.transpose(encode_rna(s), (0, 1)), dtype=torch.float32) for s in seqs] # (batch, seq, base)
            lengths = [len(s) for s in processed_seqs]

            # Pad sequences
            for i in range(len(processed_seqs)):
               m = nn.ZeroPad2d((0,0,0,max_seq_len-lengths[i]))
               processed_seqs[i] = m(processed_seqs[i])
            processed_seqs = torch.stack(processed_seqs).to(device)

            # Send sequences through the converter
            aa_seqs = conv(processed_seqs) # (seq, batch, aa)

            with torch.autocast(device_type="cuda"):
                # Reconvert to letter representation
                aa_seqs_strings = [''.join(AA_DICT[aa_seqs[i][n]] for n in range(0, lengths[i])) for i in range(len(aa_seqs))]
                final_seqs = dict(zip(tags, aa_seqs_strings))

                # Write the final sequences to FASTA
                write_fastas(final_seqs)

                loss = []
                lengths = np.sum(np.array([len(list(final_seqs.values())[i]) for i in range(len(final_seqs))]))

                with mp.Pool(processes=num_gpus) as pool:
                    jobnames = list(final_seqs.keys())
                    fasta_files = [f'FASTAs/{name}.fasta' for name in jobnames]
                    gpu_assignments = [i % num_gpus for i in range(len(fasta_files))]

                    results = pool.starmap(run_parallel, zip(fasta_files, jobnames, gpu_assignments))

                pool.close()
                pool.join()

                for jobname, path in results:
                    temp_loss = (protein_to_rna(path, get_structure(jobname, struct_path), corrector[0], tm=tm_score))
                    loss.append(temp_loss)
                    empty_dir(jobname)

            lengths = lengths/batch_size

            empty_dir("FASTAs", delete=False)
            loss = torch.mean(torch.stack(loss))
            print(f"\n\nCurrent Loss: {loss}")
            print(f"Average Loss per Residue: {loss/lengths}")
            print(f"Correction factor: {corrector}\n\n")
            loss.requires_grad = True
            loss.backward()
            
            nn.utils.clip_grad_norm_(c.parameters(), 1.0)
            
            optimizer.step()
            dist_optimizer.step()
            scheduler.step()
            dist_scheduler.step()
        torch.save(conv, f'/ConverterWeights/converter_epoch_{epoch}.pt')
        torch.save(conv.state_dict(), f'/ConverterWeights/converter_params_epoch_{epoch}.pt')
        torch.save(corrector, f'/ConverterWeights/corrector_epoch_{epoch}.pt')


def run_parallel(fasta_file, jobname, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    torch.cuda.set_device(0)

    queries, _ = get_queries(fasta_file)
    
    results =  run(
                    queries=queries,
                    result_dir=jobname,
                    use_templates=USE_TEMPLATES,
                    custom_template_path=None,
                    num_relax=0,
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
                    use_cluster_profile=True,
                    input_features_callback=input_features_callback,
                    save_recycles=False,
                    use_gpu_relax=True,
                    )
    
    path = None
    for file in os.listdir(jobname):
        if file.endswith(".pdb"):
            path = os.path.join(jobname, file)
            break
    return jobname, path

if __name__=="__main__":

 #   sys.stderr = open(os.devnull, 'w')
    mp.set_start_method('spawn', force=True)

    old_seqs, components, macro_tags = load_data(seq_path, 0, 1645, max_len=100)
    seqs = SeqDataset(old_seqs)

    try:
        c = torch.load('/ConverterWeights/converter.pt')
        corrector = torch.load('/ConverterWeights/corrector.pt')
    except:
        c = Converter(max_seq_len=200)
        corrector = [nn.Parameter(torch.tensor(6.0, requires_grad=True, dtype=torch.float32))]

    c = c.to(device)

    #try:
    print("Training...")
    train(seqs, epochs=10, batch_size=4, max_seq_len=100, converter=c, pp_dist=float(corrector[0]))
    # except:
    #     print("Error. Exiting training loop")
    #     torch.save(c, f'/ConverterWeights/converter.pt')
    #     torch.save(corrector, f'/ConverterWeights/corrector.pt')

    torch.save(c, f'/ConverterWeights/converter.pt')
    torch.save(corrector, f'/ConverterWeights/corrector.pt')



