import torch

from minselfflow.evaluate import EvalConfig, block_features, generate, linear_probe, val_loss
from minselfflow.model import DiT, DiTConfig


def test_linear_probe_separates_informative_layer_only():
    torch.manual_seed(0)
    y = torch.randint(0, 4, (2000,))
    informative = torch.nn.functional.one_hot(y, 16).float() * 3 + torch.randn(2000, 16)
    noise = torch.randn(2000, 16)
    x = torch.stack([noise, informative])
    acc = linear_probe(x[:, :1500], y[:1500], x[:, 1500:], y[1500:], epochs=50)
    assert acc[1] > 0.9, acc
    assert acc[0] < 0.4, acc


def test_eval_pieces_shapes():
    cfg = EvalConfig(sample_steps=2, batch=8, val_ts=(0.5,))
    model = DiT(DiTConfig(in_dim=12, num_tokens=256, dim=32, depth=2, heads=2)).eval()
    images = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8)
    labels = torch.randint(0, 10, (10,))

    samples = generate(model, 10, 2, cfg, torch.device("cpu"))
    assert samples.shape == (10, 3, 32, 32) and samples.dtype == torch.uint8

    vl = val_loss(model, images, labels, 2, cfg)
    assert set(vl) == {"t=0.5", "mean"} and vl["mean"] > 0

    feats = block_features(model, images, 2, cfg)
    assert feats.shape == (2, 10, 32)
