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
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from Converter import Converter
from Utils import *

seqs = {} # All sequences - may get quite large

# Used for file tree searching
components = []
macro_tags = []

      
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
# Any option other than "single_sequence" will make a ton of API calls, which is unsustainable
msa_mode = "single_sequence" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
#@markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA, "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.

# Vestigial functions - not used in this implementation
def input_features_callback(input_features):
  pass

def prediction_callback(protein_obj, length,
                        prediction_result, input_features, mode):
  model_name, relaxed = mode
  pass

# Pytorch Dataset class for sequences
class SeqDataset(torch.utils.data.Dataset):
   def __init__(self, seqs):
      self.seqs = seqs
    
   def __len__(self):
      return len(self.seqs)
   
   def __getitem__(self, idx):
      return list(self.seqs.values())[idx], list(self.seqs.keys())[idx] # (seq, tag)

# Good old training function
def train(seqs, epochs=50, batch_size=32,tm_score=False, max_seq_len=150, converter=None, pp_dist=8.81457219731867):

    losses = []
    loss_pr = []
    best_loss = float('inf')
    # Produce directories for FASTAs and weights
    os.makedirs("ConverterWeights", exist_ok=True)
    os.makedirs('FASTAs', exist_ok=True)

    # A little more vestigial code
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

    # Fallback if Converter doesn't exist
    if converter==None:
      conv = Converter(max_seq_len=max_seq_len).to(device)
    else:
      conv = converter
    conv.train()

    # Set up optimizers
    optimizer = torch.optim.AdamW(conv.parameters(), lr=0.001)

    dataloader = torch.utils.data.DataLoader(seqs, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    loss_fn = ProteinFoldLoss(pp_dist, tm_score)

    model_type = "alphafold2"
    download_alphafold_params(model_type, Path("."))
    
    # Training loop
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for seqs, tags in dataloader:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda"):
                # batch: ([(tag, seq), (tag, seq),...])

                # LAYER 1: RNA-AMINO CONVERSION
                processed_seqs = [torch.tensor(encode_rna(s), dtype=torch.float32, requires_grad=True) for s in seqs] # (batch, seq, base)
                lengths = ([len(s) for s in processed_seqs])

                # Pad sequences
                for i in range(len(processed_seqs)):
                  m = nn.ZeroPad2d((0,0,0,max_seq_len-lengths[i]))
                  processed_seqs[i] = m(processed_seqs[i])
                
                processed_seqs = torch.stack(processed_seqs).to(device)

                # Send sequences through the converter
                aa_seqs = conv(processed_seqs) # (seq, batch, aa)
                # Reconvert to letter representation
                aa_seqs_strings = ["".join(map(AA_DICT.get, aa_seq[:length])) for aa_seq, length in zip(aa_seqs, lengths)]
                final_seqs = dict(zip(tags, aa_seqs_strings))

                # Write the final sequences to FASTA
                with torch.no_grad():
                  write_fastas(final_seqs)

                # LAYER 2: PROTEIN FOLDING
                loss = loss_fn(final_seqs)

            lengths = torch.tensor(sum(lengths) / batch_size, dtype=torch.float32)
            losses.append(loss)

            empty_dir("FASTAs", delete=False)
            print(f"\n\nCurrent Loss: {loss}")
            print(f"Average Loss per Residue: {loss/lengths}")
            loss_pr.append(loss/lengths)
            loss.backward()
            
            nn.utils.clip_grad_norm_(conv.parameters(), 1.0)
            
            optimizer.step()
            if tm_score:
               if loss > best_loss:
                  print("New best loss! Saving model...")
                  torch.save(conv, f'ConverterWeights/converter.pt')
                  best_loss = loss
            else:
              if loss < best_loss:
                best_loss = loss
                print("New best loss! Saving model...")
                torch.save(conv, f'ConverterWeights/converter.pt')

        with open("losses.txt", "w") as txt_file:
          txt_file.write("\n".join([losses]) + "\n")
        with open("losses_pr.txt", "w") as txt_file:
          txt_file.write("\n".join(loss_pr) + "\n")
        torch.save(conv, f'ConverterWeights/converter_epoch_{epoch}.pt')
        torch.save(conv.state_dict(), f'ConverterWeights/converter_params_epoch_{epoch}.pt')


class ProteinFoldLoss(nn.Module):
   def __init__(self, pp_dist, tm_score):
      super(ProteinFoldLoss, self).__init__()
      self.pp_dist = pp_dist
      self.tm_score = tm_score
  
   def forward(self, final_seqs):
      num_gpus = torch.cuda.device_count()

      with torch.no_grad():
        jobnames = list(final_seqs.keys())
        fasta_files = [f'FASTAs/{name}.fasta' for name in jobnames]
        gpu_assignments = np.arange(len(fasta_files)) % num_gpus

        # Parallel process amino acid sequences
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.starmap(run_parallel, zip(fasta_files, jobnames, gpu_assignments))

      loss = torch.stack([
      protein_to_rna(path, get_structure(jobname, struct_path), self.pp_dist, tm=self.tm_score) for jobname, path in results])
      for jobname, _ in results:
          empty_dir(jobname)

      return loss.mean()
        


def run_parallel(fasta_file, jobname, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    max_seq_len = 80
    mp.set_start_method('spawn', force=True)

    old_seqs, components, macro_tags = load_data(seq_path, 0, 1645, max_len=max_seq_len)
    seqs = SeqDataset(old_seqs)

    try:
        c = torch.load('ConverterWeights/converter.pt')
    except:
        c = Converter(max_seq_len=max_seq_len)

    c = c.to(device)

    #try:
    print("Training...")
    train(seqs, epochs=10, batch_size=4, max_seq_len=max_seq_len, converter=c, tm_score=False)
    # except:
    #     print("Error. Exiting training loop")
    #     torch.save(c, f'/ConverterWeights/converter.pt')

    torch.save(c, f'ConverterWeights/converter.pt')



