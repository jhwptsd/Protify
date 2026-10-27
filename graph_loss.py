"""Plot the training curve written by train_rl.py (losses.txt) or train_supervised.py (losses_sl.txt).

Each line of the file is "<batch loss>\t<loss per residue>"."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="losses.txt")
    parser.add_argument("--out", default="loss_pr.png")
    parser.add_argument("--column", type=int, choices=[0, 1], default=1,
                        help="0 = loss per batch, 1 = loss per residue")
    parser.add_argument("--window", type=int, default=1, help="moving-average window (in batches)")
    args = parser.parse_args()

    values = []
    with open(args.path) as file:
        for line in file:
            try:
                values.append(float(line.strip().split("\t")[args.column]))
            except (ValueError, IndexError):
                print(f"Skipping invalid line: {line.strip()}")

    if not values:
        raise SystemExit(f"No loss values found in {args.path}")

    w = max(1, min(args.window, len(values)))
    smoothed = [sum(values[i:i + w]) / w for i in range(len(values) - w + 1)]

    fig, ax = plt.subplots()
    ax.plot(range(w - 1, len(values)), smoothed)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss per residue" if args.column else "Loss")
    ax.grid()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
