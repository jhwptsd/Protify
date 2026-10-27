"""Train the RNA -> protein converter by distilling its own best samples (best-of-N / expert iteration).

AlphaFold is not differentiable, so the converter can't be trained by backpropagating a fold loss
(the previous version of this script computed a loss that never reached the converter's weights).
Instead, for every RNA in a batch we sample N candidate protein sequences, fold them all, keep the
candidate whose fold scores best against the true RNA structure, and train the converter with
ordinary cross-entropy to reproduce it. That is supervised learning on self-generated targets.
It costs N folds per RNA; train_rl.py needs only one."""
import argparse
import os
import warnings

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from Converter import Converter, load_converter
from protify_pipeline import DEFAULT_PP_DIST, SeqDataset, build_task, indices_to_strings, prepare_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset, loss_fn, converter, epochs=10, batch_size=4, num_samples=4, lr=1e-3,
          out_dir="ConverterWeights", losses_path="losses_sl.txt"):
    os.makedirs(out_dir, exist_ok=True)
    conv = converter
    optimizer = torch.optim.AdamW(conv.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses, losses_per_residue = [], []
    best_epoch_loss = float("inf")

    for epoch in range(epochs):
        epoch_losses = []
        for rna_seqs, tags in dataloader:
            x, lengths, mask = prepare_batch(rna_seqs, device)

            # Sample N candidates per RNA
            conv.eval()
            with torch.no_grad():
                probs = conv(x, src_key_padding_mask=mask, sm=True)  # (B, L, 20)
                dist = torch.distributions.Categorical(probs=probs)
                samples = torch.stack([dist.sample() for _ in range(num_samples)])  # (N, B, L)

            candidates = {}  # jobname -> protein sequence
            reference = {}   # jobname -> RNA tag
            for k in range(num_samples):
                for tag, seq in zip(tags, indices_to_strings(samples[k].cpu(), lengths)):
                    candidates[f"{tag}_s{k}"] = seq
                    reference[f"{tag}_s{k}"] = tag
            batch_loss = loss_fn(candidates, reference).to(device).view(num_samples, len(tags))  # (N, B)

            # Keep the best candidate for each RNA and learn to reproduce it
            best_loss, best_k = batch_loss.min(dim=0)  # (B,)
            targets = samples[best_k, torch.arange(len(tags), device=device)]  # (B, L)

            conv.train()
            probs = conv(x, src_key_padding_mask=mask, sm=True)
            log_probs = torch.log(probs.clamp_min(1e-9))
            nll = F.nll_loss(log_probs.transpose(1, 2), targets, reduction="none")  # (B, L)
            token_mask = (~mask).float()
            train_loss = (nll * token_mask).sum() / token_mask.sum()

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(conv.parameters(), 1.0)
            optimizer.step()

            # Report the average fold loss over all candidates: it should go down over time
            loss = batch_loss.mean().item()
            per_residue = (batch_loss / lengths.to(device).float()).mean().item()
            losses.append(loss)
            losses_per_residue.append(per_residue)
            epoch_losses.append(loss)
            print(f"\n\nCurrent Loss: {loss} (best candidates: {best_loss.mean().item()})")
            print(f"Average Loss per Residue: {per_residue}")
            print(f"Cross-entropy on best candidates: {train_loss.item()}")

        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            print("New best epoch loss! Saving model...")
            torch.save(conv, f"{out_dir}/converter.pt")

        with open(losses_path, "w") as txt_file:
            txt_file.writelines(f"{l}\t{p}\n" for l, p in zip(losses, losses_per_residue))
        torch.save(conv, f"{out_dir}/converter_epoch_{epoch}.pt")
        torch.save(conv.state_dict(), f"{out_dir}/converter_params_epoch_{epoch}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4, help="candidate sequences folded per RNA")
    parser.add_argument("--max-seq-len", type=int, default=80, help="skip RNAs longer than this")
    parser.add_argument("--num-rnas", type=int, default=1646, help="use the first N entries of RNA3DB")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pp-dist", type=float, default=DEFAULT_PP_DIST)
    parser.add_argument("--loss", choices=["angle", "rmsd", "tm"], default="angle")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the saved converter.pt and start from random weights")
    parser.add_argument("--fake-fold", action="store_true",
                        help="replace AlphaFold and RNA3DB with a cheap synthetic task (runs on a laptop CPU; "
                             "writes to ConverterWeights_fake/ and losses_fake.txt)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    torch.manual_seed(args.seed)
    mp.set_start_method("spawn", force=True)

    seqs, loss_fn, cleanup, out_dir = build_task(args)
    losses_path = "losses_fake.txt" if args.fake_fold else "losses_sl.txt"

    checkpoint = f"{out_dir}/converter.pt"
    if os.path.exists(checkpoint) and not args.fresh:
        print(f"Resuming from {checkpoint}")
        conv = load_converter(checkpoint, device, max_seq_len=args.max_seq_len)
    else:
        conv = Converter(max_seq_len=args.max_seq_len).to(device)

    try:
        print("Training...")
        train(SeqDataset(seqs), loss_fn, conv, epochs=args.epochs, batch_size=args.batch_size,
              num_samples=args.num_samples, lr=args.lr, out_dir=out_dir, losses_path=losses_path)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
