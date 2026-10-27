# Protify

Protify is a work-in-progress research project that tries to fold RNA with protein-folding tools.

A Transformer (the *converter*) reads an RNA sequence and writes a protein sequence whose backbone geometry should resemble the RNA's. A protein-folding model (AlphaFold2, via [ColabFold](https://github.com/sokrypton/ColabFold)) then folds that protein, and its predicted structure stands in for the RNA's.

```
RNA sequence ──converter──▶ protein sequence ──AlphaFold2──▶ protein structure ──▶ compare to true RNA structure
```

Training data is [RNA3DB](https://github.com/marcellszi/rna3db), which the training scripts download on first run. The loss compares the folded protein's C-alpha trace with the RNA's phosphorus (P) trace.

## What's in the repo

| File | Purpose |
|---|---|
| `Converter.py` | The converter model (a small `nn.Transformer`: one-hot A/U/C/G in, 20 amino-acid probabilities out). |
| `Utils.py` | Data loading, structure parsing, Kabsch alignment, RMSD / TM-score and the angle loss. |
| `protify_pipeline.py` | Shared training plumbing: RNA3DB download, batching, and the AlphaFold folding workers. |
| `train_rl.py` | Trains the converter with REINFORCE (recommended). |
| `train_supervised.py` | Trains the converter by distilling its best-of-N samples (N folds per RNA, so slower). |
| `validate.py` | Scores predictions against the true structures in `Validation/Actual/`. |
| `run_converter.py` | Runs a trained converter on RNA sequences you give it. |
| `fake_fold.py` | A synthetic stand-in for AlphaFold used by `--fake-fold` (see below). |
| `phosphodiester_length.py` | Measures the average P-P distance over RNA3DB (for `--pp-dist`). |
| `graph_loss.py` | Plots a training curve from `losses.txt`. |
| `substitution/` | Experimental: swaps protein residues for nucleotides to build an all-atom RNA model. |
| `tests/test_protify.py` | Unit tests for the code above (the other files in `tests/` belong to ColabFold). |

Everything else (`colabfold/`, the notebooks, `MsaServer/`, `beta/`, `batch/`, `test-data/`, the rest of `tests/`) comes from ColabFold.

Because AlphaFold is not differentiable, the converter can't be trained by backpropagating through the folding step. `train_rl.py` samples a protein sequence per RNA, folds it, and uses the score as a reward (a policy gradient with a leave-one-out baseline). `train_supervised.py` samples several sequences per RNA, folds all of them, and trains the converter with cross-entropy on the best one.

## Setup

You need a CUDA-enabled machine with Nvidia GPUs. We trained on Amazon EC2 G5 instances. It also runs on CPU, but folding will be very slow.

```
git clone https://github.com/jhwptsd/Protify.git
cd Protify
pip install .
pip install "colabfold[alphafold]"
pip install torch==2.0.1
pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The scripts need `wget` and `tar` to fetch RNA3DB and the AlphaFold2 weights on first run.

## Training

```
python train_rl.py              # or: python train_supervised.py
```

Useful options (see `--help` for all of them):

| Option | Meaning |
|---|---|
| `--epochs`, `--batch-size` | Training length and batch size (default 10 and 4). |
| `--max-seq-len` | Skip RNAs longer than this (default 80); also the longest sequence the converter accepts. |
| `--loss {angle,rmsd,tm}` | What to score folds with. `angle` (default) compares virtual bond angles and torsions and needs no alignment. `rmsd` and `tm` superimpose the traces after rescaling the protein to `--pp-dist` spacing. |
| `--fresh` | Start from random weights instead of resuming from `ConverterWeights/converter.pt`. |
| `--num-samples` | (`train_supervised.py`) Candidate sequences folded per RNA. |

Outputs:

- `losses.txt` (`train_rl.py`) or `losses_sl.txt` (`train_supervised.py`): the loss and loss per residue for every batch. Plot it with `python graph_loss.py losses.txt --window 20`.
- `ConverterWeights/converter.pt`: the model from the epoch with the lowest average loss.
- `ConverterWeights/converter_epoch_N.pt` and `converter_params_epoch_N.pt`: the full model and its `state_dict` after each epoch.

### Trying it without a GPU (`--fake-fold`)

```
python train_rl.py --fake-fold --fresh --epochs 5 --batch-size 8 --max-seq-len 30 --num-rnas 200
```

`--fake-fold` replaces AlphaFold and RNA3DB with a cheap synthetic task, so training runs in seconds on a laptop CPU with only `torch` and `biopython` installed (no ColabFold, no downloads). Each amino acid has a fixed bond angle and torsion, and a protein "folds" into the chain those angles describe. A nucleotide's target geometry is that of one particular amino acid, so a converter that learns to map A, U, C and G to those amino acids reaches zero loss.

Use it to check that the training loop learns, runs end to end and saves checkpoints: the loss should fall well below its starting value. It says nothing about whether the real approach works. Outputs go to `ConverterWeights_fake/` and `losses_fake.txt`, so real checkpoints are never touched. It works with `train_supervised.py` too, and `--loss` is ignored (the angle loss is always used).

### About the checkpoints in `ConverterWeights/`

They were trained before a bug fix in the positional encoding: positions were not being encoded, so the model could not tell where in the sequence a nucleotide was. `load_converter` makes the old files load, but they should be treated as stale. Train with `--fresh` to replace them.

### About `--pp-dist`

`--pp-dist` only matters for `--loss rmsd` and `--loss tm`. The default (8.81 Å) came from an earlier version of `phosphodiester_length.py` that mixed up residue gaps. Typical consecutive P-P distances in RNA are around 6 Å. Run `python phosphodiester_length.py` and pass the result as `--pp-dist`.

## Running a trained converter

```
python run_converter.py GGUAUGAGGGUAUU AAAAAAAAAA
```

prints each RNA sequence with the protein sequence the converter chose for it. Fold that protein with any of the ColabFold notebooks.

## Validation

`Validation/Actual/` holds true structures of 12 RNAs. `Validation/Predicted/` holds AlphaFold and trRosettaRNA predictions for the same RNAs.

```
python validate.py --predicted Validation/Predicted/trRosettaRNA   # baseline: how good are ready-made RNA predictions?
python validate.py --converter ConverterWeights/converter.pt       # Protify itself: converter -> AlphaFold2 -> score
```

Both print RMSD (Å) and TM-score per structure (P-atom trace, after superposition) and their means. Add `--plot rmsd.png` to save an RMSD-vs-length plot. Residues are compared one-to-one in order, truncated to the shorter chain; there is no sequence alignment.

## Tests

```
pytest tests/test_protify.py
```

The Protify tests are skipped if `torch` or `biopython` isn't installed. A plain `pytest` also runs the ColabFold tests, which need its dependencies.

## Credits

The folding code is [ColabFold](https://github.com/sokrypton/ColabFold) (see `LICENSE`). RNA structures come from [RNA3DB](https://github.com/marcellszi/rna3db).
