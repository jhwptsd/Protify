"""A stand-in for AlphaFold, so the training loop can be tried without a GPU, ColabFold or RNA3DB.

Each amino acid is given a fixed (bond angle, torsion) pair and a protein "folds" into the chain those
angles describe. Each nucleotide's "true" geometry is the pair of one particular amino acid
(see RNA_TO_AA), so the best possible converter maps A, U, C, G to those four amino acids and reaches
zero loss. A converter that trains correctly should find this mapping; a random one scores badly.

This only checks that the training code works. It says nothing about whether the real approach does."""
import math

import torch
import torch.nn.functional as F

from Utils import AA_DICT, AngleLoss

AA_INDEX = {letter: i for i, letter in AA_DICT.items()}
RNA_TO_AA = {"A": 0, "U": 5, "C": 10, "G": 15}
MIN_RNA_LEN = 8


def _angle_table(seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = 1.4 + torch.rand(20, generator=g)  # interior bond angles, 1.4-2.4 rad
    phi = (torch.rand(20, generator=g) * 2 - 1) * math.pi  # torsions
    return theta, phi


THETA, PHI = _angle_table()


def build_trace(theta, phi):
    """Build an (N, 3) chain with unit bond lengths from per-point bond angles and torsions (NeRF).
    The first three points are fixed, so the first three entries don't affect the shape."""
    pts = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.5, 0.8, 0.0])]
    for k in range(3, len(theta)):
        a, b, c = pts[-3], pts[-2], pts[-1]
        bc = F.normalize(c - b, dim=0)
        normal = F.normalize(torch.cross(b - a, bc, dim=0), dim=0)
        m = torch.cross(normal, bc, dim=0)
        t, p = theta[k], phi[k]
        pts.append(c - torch.cos(t) * bc + torch.sin(t) * torch.cos(p) * m + torch.sin(t) * torch.sin(p) * normal)
    return torch.stack(pts[:len(theta)] if len(theta) < 3 else pts)


def fake_protein_trace(seq):
    idx = torch.tensor([AA_INDEX[c] for c in seq], dtype=torch.long)
    return build_trace(THETA[idx], PHI[idx])


def fake_rna_trace(seq):
    idx = torch.tensor([RNA_TO_AA[c] for c in seq if c in RNA_TO_AA], dtype=torch.long)
    return build_trace(THETA[idx], PHI[idx])


def synthetic_rna_dataset(n, max_len, seed=0):
    """{tag: random RNA sequence} with lengths between MIN_RNA_LEN and max_len."""
    g = torch.Generator().manual_seed(seed)
    seqs = {}
    for i in range(n):
        length = int(torch.randint(MIN_RNA_LEN, max(MIN_RNA_LEN, max_len) + 1, (1,), generator=g))
        seqs[f"fake{i}"] = "".join("AUCG"[j] for j in torch.randint(0, 4, (length,), generator=g).tolist())
    return seqs


class FakeFoldLoss:
    """Same interface as ProteinFoldLoss: (final_seqs, ref_tags=None) -> (B,) tensor of losses."""
    def __init__(self, rna_seqs):
        self.rna_seqs = rna_seqs
        self.angle_loss = AngleLoss()

    def __call__(self, final_seqs, ref_tags=None):
        ref_tags = ref_tags or {name: name for name in final_seqs}
        with torch.no_grad():
            losses = [self.angle_loss(fake_protein_trace(seq), fake_rna_trace(self.rna_seqs[ref_tags[name]]))
                      for name, seq in final_seqs.items()]
        return torch.stack(losses)


def fake_mapping_report(conv, device, length=24, seed=0):
    """Print what the converter currently does with each nucleotide in the fake task: the amino acid it
    gives that nucleotide on average (and with what probability), next to the amino acid that scores zero."""
    from Utils import encode_rna
    g = torch.Generator().manual_seed(seed)
    seq = "".join("AUCG"[j] for j in torch.randint(0, 4, (length,), generator=g).tolist())
    was_training = conv.training
    conv.eval()
    with torch.no_grad():
        probs = conv(encode_rna(seq).unsqueeze(0).to(device), sm=True)[0].cpu()  # (L, 20)
    conv.train(was_training)
    parts = []
    for nuc, target in RNA_TO_AA.items():
        rows = [i for i, c in enumerate(seq) if c == nuc and i >= 3]
        mean = probs[rows].mean(0)
        best = int(mean.argmax())
        parts.append(f"{nuc}: picks {AA_DICT[best]} (p={mean[best]:.2f}), ideal {AA_DICT[target]} (p={mean[target]:.2f})")
    print("Mapping check -> " + " | ".join(parts))
