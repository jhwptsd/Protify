"""Train the RNA -> protein converter with REINFORCE.

Each step samples a protein sequence for every RNA in the batch, folds it with AlphaFold2, scores
the fold against the true RNA structure, and nudges the converter towards sequences that scored
better than the rest of the batch. AlphaFold isn't differentiable, hence the policy gradient."""
import argparse
import math
import os
import warnings

import torch
import torch.multiprocessing as mp

from Converter import Converter, load_converter
from protify_pipeline import DEFAULT_PP_DIST, SeqDataset, build_task, indices_to_strings, prepare_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def leave_one_out_advantage(rewards):
    """Reward minus the mean of the *other* rewards in the batch (an unbiased baseline)."""
    n = rewards.numel()
    if n < 2:
        return rewards
    return rewards - (rewards.sum() - rewards) / (n - 1)


def train(dataset, loss_fn, converter, epochs=10, batch_size=4, lr=1e-3, entropy_coef=0.1,
          out_dir="ConverterWeights", epoch_hook=None, losses_path="losses.txt"):
    os.makedirs(out_dir, exist_ok=True)
    conv = converter
    conv.train()
    optimizer = torch.optim.AdamW(conv.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses, losses_per_residue, entropies = [], [], []
    best_epoch_loss = float("inf")

    for epoch in range(epochs):
        epoch_losses = []
        for rna_seqs, tags in dataloader:
            x, lengths, mask = prepare_batch(rna_seqs, device)

            # Sample one protein sequence per RNA
            probs = conv(x, src_key_padding_mask=mask, sm=True)  # (B, L, 20)
            dist = torch.distributions.Categorical(probs=probs)
            aa_indices = dist.sample()  # (B, L)
            token_mask = (~mask).float()

            final_seqs = dict(zip(tags, indices_to_strings(aa_indices.cpu(), lengths)))

            # Fold and score (lower is better for both RMSD/angle loss and negative TM-score)
            batch_loss = loss_fn(final_seqs).to(device)  # (B,)
            lengths_f = lengths.to(device).float()
            rewards = -batch_loss / lengths_f

            # Scale advantages to unit spread so the policy-gradient term doesn't shrink with the
            # reward's units (per-residue rewards are ~0.01) and get swamped by the entropy bonus
            advantage = leave_one_out_advantage(rewards)
            advantage = advantage / (advantage.std() + 1e-8) if advantage.numel() > 1 else advantage
            log_probs = (dist.log_prob(aa_indices) * token_mask).sum(dim=1)  # padding contributes nothing
            entropy = (dist.entropy() * token_mask).sum() / token_mask.sum()
            policy_loss = -(log_probs * advantage).mean() - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(conv.parameters(), 1.0)
            optimizer.step()

            loss = batch_loss.mean().item()
            per_residue = (batch_loss / lengths_f).mean().item()
            losses.append(loss)
            losses_per_residue.append(per_residue)
            entropies.append(entropy.item())
            epoch_losses.append(loss)
            print(f"\n\nCurrent Loss: {loss}")
            print(f"Average Loss per Residue: {per_residue}")
            print(f"Policy entropy: {entropy.item():.3f} (uniform = {math.log(20):.3f})")

        if epoch_hook is not None:
            epoch_hook(conv)

        epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        # Lower is better for every loss option (the TM-score loss is already negated)
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            print("New best epoch loss! Saving model...")
            torch.save(conv, f"{out_dir}/converter.pt")

        with open(losses_path, "w") as txt_file:
            txt_file.writelines(f"{l}\t{p}\t{h}\n" for l, p, h in zip(losses, losses_per_residue, entropies))
        torch.save(conv, f"{out_dir}/converter_epoch_{epoch}.pt")
        torch.save(conv.state_dict(), f"{out_dir}/converter_params_epoch_{epoch}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=80, help="skip RNAs longer than this")
    parser.add_argument("--num-rnas", type=int, default=1646, help="use the first N entries of RNA3DB")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--pp-dist", type=float, default=DEFAULT_PP_DIST,
                        help="P-P spacing used when scoring by RMSD/TM-score (unused for the default angle loss)")
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
    epoch_hook = None
    if args.fake_fold:
        from fake_fold import fake_mapping_report
        epoch_hook = lambda model: fake_mapping_report(model, device)
    losses_path = "losses_fake.txt" if args.fake_fold else "losses.txt"

    checkpoint = f"{out_dir}/converter.pt"
    if os.path.exists(checkpoint) and not args.fresh:
        print(f"Resuming from {checkpoint}")
        conv = load_converter(checkpoint, device, max_seq_len=args.max_seq_len)
    else:
        conv = Converter(max_seq_len=args.max_seq_len).to(device)

    try:
        print("Training...")
        train(SeqDataset(seqs), loss_fn, conv, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, entropy_coef=args.entropy_coef, out_dir=out_dir, losses_path=losses_path, epoch_hook=epoch_hook)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
