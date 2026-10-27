"""Shared plumbing for the Protify training and validation scripts: data download, batching,
and the (non-differentiable) AlphaFold folding step that turns proteins into a loss.

Importing this module has no side effects. Heavy imports (colabfold/JAX) happen inside the
folding workers, after each worker has been pinned to its GPU."""
import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Converter import padding_mask_from_lengths
from Utils import AA_DICT, encode_rna, empty_dir, protein_to_rna, write_fastas

SEQ_PATH = "rna3db-jsons/cluster.json"
STRUCT_PATH = "rna3db-mmcifs"
WORK_DIR = "FoldingWork"

# Average distance used to rescale CA traces to P-P spacing (see phosphodiester_length.py)
DEFAULT_PP_DIST = 8.81457219731867

RNA3DB_URL = "https://github.com/marcellszi/rna3db/releases/download/incremental-update"

# ColabFold settings. Single-sequence mode avoids thousands of MSA server calls.
FOLD_SETTINGS = dict(
    model_type="auto",
    num_recycles=1,
    recycle_early_stop_tolerance=None,
    relax_max_iterations=200,
    pairing_strategy="greedy",
    max_msa=None,
    num_seeds=1,
    use_dropout=True,
    msa_mode="single_sequence",
    pair_mode="unpaired_paired",
)


def _run(cmd):
    subprocess.run(cmd, check=True)


def setup_environment():
    """Download and unpack RNA3DB (structures and sequences) if it isn't already present."""
    if not os.path.exists(STRUCT_PATH):
        print("Downloading RNA3Db structure files...")
        _run(["wget", "-q", f"{RNA3DB_URL}/rna3db-mmcifs.v2.tar.xz"])
        print("Extracting structure files...")
        _run(["tar", "-xf", "rna3db-mmcifs.v2.tar.xz"])
        os.remove("rna3db-mmcifs.v2.tar.xz")
    if not os.path.exists(SEQ_PATH):
        print("Downloading RNA3Db sequence files...")
        _run(["wget", "-q", f"{RNA3DB_URL}/rna3db-jsons.tar.gz"])
        print("Extracting sequence files...")
        _run(["tar", "-xzf", "rna3db-jsons.tar.gz"])
        os.remove("rna3db-jsons.tar.gz")


def structure_paths(seqs, components, macro_tags, root=STRUCT_PATH):
    """Map each RNA tag to its mmCIF file. Layout: root/train_set/<component>/<macro tag>/<tag>.cif"""
    return {tag: f"{root}/train_set/{comp}/{macro}/{tag}.cif"
            for tag, comp, macro in zip(seqs.keys(), components, macro_tags)}


class SeqDataset(torch.utils.data.Dataset):
    """Dataset of (rna sequence, tag) pairs."""
    def __init__(self, seqs):
        self.tags = list(seqs.keys())
        self.seqs = list(seqs.values())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.tags[idx]


def prepare_batch(seqs, device):
    """One-hot encode and pad a list of RNA strings.
    Returns (input (B, L, 4), lengths (B,), padding mask (B, L) that is True at padding)."""
    encoded = [encode_rna(s) for s in seqs]
    lengths = torch.tensor([len(e) for e in encoded])
    x = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0).to(device)
    mask = padding_mask_from_lengths(lengths, x.size(1), device=device)
    return x, lengths, mask


def indices_to_strings(indices, lengths):
    """Convert a (B, L) tensor of amino-acid indices into strings, dropping padding."""
    return ["".join(AA_DICT[int(i)] for i in row[:int(n)]) for row, n in zip(indices, lengths)]


def _pin_gpu(gpu_id, num_gpus):
    # Runs once in each worker before JAX is imported, so each worker only sees one GPU
    if num_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def run_parallel(fasta_file, jobname, work_dir=WORK_DIR):
    """Fold one FASTA file with ColabFold's AlphaFold2. Returns (jobname, path to PDB)."""
    from colabfold.batch import get_queries, run

    s = FOLD_SETTINGS
    result_dir = os.path.join(work_dir, jobname)
    queries, _ = get_queries(fasta_file)

    run(
        queries=queries,
        result_dir=result_dir,
        use_templates=False,
        custom_template_path=None,
        num_relax=0,
        msa_mode=s["msa_mode"],
        model_type=s["model_type"],
        num_models=1,
        num_recycles=s["num_recycles"],
        relax_max_iterations=s["relax_max_iterations"],
        recycle_early_stop_tolerance=s["recycle_early_stop_tolerance"],
        num_seeds=s["num_seeds"],
        use_dropout=s["use_dropout"],
        model_order=[1, 2, 3, 4, 5],
        is_complex=False,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=s["pair_mode"],
        pairing_strategy=s["pairing_strategy"],
        stop_at_score=float(100),
        prediction_callback=None,
        dpi=100,
        zip_results=False,
        save_all=False,
        max_msa=s["max_msa"],
        use_cluster_profile=True,
        input_features_callback=None,
        save_recycles=False,
        use_gpu_relax=True,
    )

    pdbs = sorted(f for f in os.listdir(result_dir) if f.endswith(".pdb"))
    if not pdbs:
        raise RuntimeError(f"AlphaFold produced no PDB for {jobname}")
    return jobname, os.path.join(result_dir, pdbs[0])


class FoldingWorkers:
    """One long-lived single-process pool per GPU, so JAX and the model code are loaded once
    per GPU instead of once per batch. Falls back to a single CPU worker without GPUs."""
    def __init__(self, work_dir=WORK_DIR):
        from colabfold.download import download_alphafold_params
        download_alphafold_params("alphafold2", Path("."))
        self.work_dir = work_dir
        self.num_gpus = torch.cuda.device_count()
        ctx = mp.get_context("spawn")
        self.pools = [ctx.Pool(1, initializer=_pin_gpu, initargs=(i, self.num_gpus))
                      for i in range(max(1, self.num_gpus))]

    def fold(self, fasta_files, jobnames):
        """Fold many FASTA files, spread round-robin over the workers. Returns {jobname: pdb path}."""
        pending = [self.pools[i % len(self.pools)].apply_async(run_parallel, (fasta, name, self.work_dir))
                   for i, (fasta, name) in enumerate(zip(fasta_files, jobnames))]
        return dict(r.get() for r in pending)

    def close(self):
        for pool in self.pools:
            pool.close()
            pool.join()


class ProteinFoldLoss:
    """Fold a batch of protein sequences and score them against the true RNA structures.
    Not differentiable: it is used as a reward for the converter."""
    def __init__(self, struct_paths, workers, pp_dist=DEFAULT_PP_DIST, use_tm=False, angle=True,
                 work_dir=WORK_DIR):
        self.struct_paths = struct_paths
        self.workers = workers
        self.pp_dist = pp_dist
        self.use_tm = use_tm
        self.angle = angle
        self.work_dir = work_dir

    def __call__(self, final_seqs, ref_tags=None):
        """final_seqs: {jobname: protein sequence}. ref_tags: {jobname: RNA tag}, default identity.
        Returns a (B,) tensor of losses in the order of final_seqs."""
        ref_tags = ref_tags or {name: name for name in final_seqs}
        jobnames = list(final_seqs)
        fasta_dir = os.path.join(self.work_dir, "fastas")
        write_fastas(final_seqs, fasta_dir)

        try:
            with torch.no_grad():
                pdbs = self.workers.fold([os.path.join(fasta_dir, f"{n}.fasta") for n in jobnames], jobnames)
                losses = [protein_to_rna(pdbs[n], self.struct_paths[ref_tags[n]], self.pp_dist,
                                         tm=self.use_tm, angle=self.angle) for n in jobnames]
        finally:
            empty_dir(self.work_dir)
        return torch.stack(losses)


FAKE_OUT_DIR = "ConverterWeights_fake"


def build_task(args):
    """Return (rna sequences, loss function, cleanup function, weights dir) for the chosen mode.
    With --fake-fold, AlphaFold and RNA3DB are replaced by the synthetic task in fake_fold.py."""
    if getattr(args, "fake_fold", False):
        from fake_fold import FakeFoldLoss, synthetic_rna_dataset
        print("Using the fake folder: synthetic RNAs and a synthetic fold score (no AlphaFold, no RNA3DB).")
        if args.loss != "angle":
            print("Note: --loss is ignored in fake mode; the angle loss is always used.")
        seqs = synthetic_rna_dataset(args.num_rnas, args.max_seq_len, args.seed)
        return seqs, FakeFoldLoss(seqs), (lambda: None), FAKE_OUT_DIR

    from Utils import load_data
    setup_environment()
    seqs, components, macro_tags = load_data(SEQ_PATH, 0, args.num_rnas - 1, max_len=args.max_seq_len)
    paths = structure_paths(seqs, components, macro_tags, STRUCT_PATH)
    workers = FoldingWorkers()
    loss_fn = ProteinFoldLoss(paths, workers, pp_dist=args.pp_dist, use_tm=args.loss == "tm",
                              angle=args.loss == "angle")
    return seqs, loss_fn, workers.close, "ConverterWeights"
