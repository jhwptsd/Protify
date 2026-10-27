"""Tests for the Protify code (Converter, Utils, training helpers). Skipped without torch/biopython."""
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("Bio")

from Converter import Converter, PositionalEncoding, padding_mask_from_lengths
from Utils import (AngleLoss, RMSD, correct_protein_coords, encode_rna, kabsch_algorithm, parse_json,
                   parse_rna, rna_d0, tm_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def random_rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_positional_encoding_varies_by_position_not_batch():
    pe = PositionalEncoding(16, dropout=0.0, max_len=50)
    out = pe(torch.zeros(3, 10, 16))
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert torch.allclose(out[0], out[1])


def test_padding_mask_from_lengths():
    mask = padding_mask_from_lengths([2, 4])
    assert mask.tolist() == [[False, False, True, True], [False, False, False, False]]


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    model = Converter(max_seq_len=20).eval()
    short = encode_rna("GGUAC").unsqueeze(0)
    padded = torch.cat([short, torch.zeros(1, 3, 4)], dim=1)
    with torch.no_grad():
        alone = model(short, sm=True)
        masked = model(padded, src_key_padding_mask=padding_mask_from_lengths([5], 8), sm=True)
    assert torch.allclose(alone, masked[:, :5], atol=1e-5)


def test_converter_output_shapes():
    model = Converter(max_seq_len=20).eval()
    x = torch.stack([encode_rna("ACGUACGUAC"), encode_rna("UUUUUGGGGG")])
    assert model(x, sm=True).shape == (2, 10, 20)
    out = model(x)
    assert len(out) == 2 and len(out[0]) == 10 and all(0 <= i < 20 for i in out[0])


def test_converter_trains_through_softmax_output():
    model = Converter(max_seq_len=20)
    probs = model(encode_rna("ACGUACGU").unsqueeze(0), sm=True)
    probs.log().sum().backward()
    assert all(p.grad is not None for p in model.output_linear.parameters())


def test_encode_rna_drops_unknown_letters():
    assert encode_rna("ANU").tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]


def test_parse_json(tmp_path):
    data = {"c1": {"m1": {f"t{i}": {"length": 10 * i, "sequence": "A" * (10 * i)} for i in range(1, 5)}}}
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(data))
    seqs, comps, macros = parse_json(str(path), 1, 3, max_len=25)  # entries 1..3 are t2..t4; only t2 fits
    assert list(seqs) == ["t2"]
    assert len(comps) == len(macros) == len(seqs)


def test_kabsch_recovers_rigid_motion():
    torch.manual_seed(0)
    p = torch.randn(30, 3)
    q = p @ random_rotation().T + torch.randn(3)
    a, b = kabsch_algorithm(p, q)
    assert RMSD(a, b) < 1e-4


def test_kabsch_never_reflects():
    torch.manual_seed(0)
    p = torch.randn(30, 3)
    mirrored = p * torch.tensor([1.0, 1.0, -1.0])
    a, b = kabsch_algorithm(p, mirrored)
    assert RMSD(a, b) > 0.1  # a mirror image cannot be superimposed by a proper rotation


def test_rmsd_of_known_offset():
    p = torch.zeros(4, 3)
    q = torch.tensor([[3.0, 4.0, 0.0]] * 4)
    assert RMSD(p, q).item() == pytest.approx(5.0)


def test_tm_score_identical_is_minus_one():
    p = torch.randn(25, 3)
    assert tm_score(p, p, 25).item() == pytest.approx(-1.0)
    assert rna_d0(100) == pytest.approx(0.6 * math.sqrt(99.5) - 2.5)


def test_angle_loss_is_invariant_and_zero_for_identical():
    torch.manual_seed(0)
    p = torch.randn(12, 3)
    loss = AngleLoss()
    assert loss(p, p).item() == pytest.approx(0.0, abs=1e-6)
    moved = 4.0 * p @ random_rotation().T + torch.randn(3)
    assert loss(moved, p).item() == pytest.approx(0.0, abs=1e-4)
    assert loss(torch.randn(12, 3), p).item() > 0.01


def test_angle_loss_handles_short_chains():
    assert AngleLoss()(torch.randn(3, 3), torch.randn(3, 3)).item() >= 0


def test_correct_protein_coords_sets_step_length():
    pts = torch.randn(10, 3)
    out = correct_protein_coords(pts, 5.9)
    assert out.shape == pts.shape
    assert torch.allclose(torch.norm(out[1:] - out[:-1], dim=1), torch.full((9,), 5.9), atol=1e-4)


def test_parse_rna_reads_phosphorus_trace():
    path = os.path.join(ROOT, "Validation", "Actual", "R1107.pdb")
    pts, breaks = parse_rna(path, pdb=True, return_breaks=True)
    assert pts.shape[1] == 3 and len(pts) > 10
    assert torch.allclose(pts[0], torch.zeros(3))
    assert len(breaks) == len(pts) - 1


def test_leave_one_out_advantage():
    from train_rl import leave_one_out_advantage
    r = torch.tensor([1.0, 2.0, 6.0])
    # each reward minus the mean of the other two
    assert torch.allclose(leave_one_out_advantage(r), torch.tensor([1 - 4.0, 2 - 3.5, 6 - 1.5]))


def test_fake_fold_perfect_converter_scores_zero():
    from fake_fold import FakeFoldLoss, RNA_TO_AA
    from Utils import AA_DICT
    rna = {"r1": "ACGUACGUAC", "r2": "GGGUUUCCCAAA"}
    perfect = {k: "".join(AA_DICT[RNA_TO_AA[c]] for c in v) for k, v in rna.items()}
    losses = FakeFoldLoss(rna)(perfect)
    assert losses.shape == (2,) and torch.all(losses < 1e-4)


def test_fake_fold_wrong_converter_scores_worse():
    from fake_fold import FakeFoldLoss
    rna = {"r1": "ACGUACGUACGU"}
    assert FakeFoldLoss(rna)({"r1": "WWWWWWWWWWWW"}).item() > 0.05


def test_fake_fold_trace_has_unit_bonds():
    from fake_fold import fake_protein_trace
    pts = fake_protein_trace("ARNDCQEGHILK")
    assert torch.allclose(torch.norm(pts[1:] - pts[:-1], dim=1)[1:], torch.ones(len(pts) - 2), atol=1e-4)


def test_synthetic_dataset_lengths():
    from fake_fold import MIN_RNA_LEN, synthetic_rna_dataset
    data = synthetic_rna_dataset(20, 30, seed=1)
    assert len(data) == 20 and all(MIN_RNA_LEN <= len(s) <= 30 and set(s) <= set("AUCG") for s in data.values())
