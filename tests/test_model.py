import torch

from minselfflow.model import DiT, DiTConfig, patchify, unpatchify


def test_patchify_layout_and_roundtrip():
    img = torch.randn(2, 3, 8, 8)
    tokens = patchify(img, 2)
    assert tokens.shape == (2, 16, 12)
    # token 1 is the patch at grid (row 0, col 1): pixels [0:2, 2:4], flattened as (py, px, ch)
    assert torch.equal(tokens[0, 1], img[0, :, 0:2, 2:4].permute(1, 2, 0).reshape(-1))
    assert torch.equal(unpatchify(tokens, 2), img)


def test_per_token_timestep_only_changes_that_token_without_attention():
    cfg = DiTConfig(in_dim=12, num_tokens=16, dim=32, depth=2, heads=2, num_classes=10)
    model = DiT(cfg)
    for p in model.parameters():  # break zero-init so outputs depend on inputs
        torch.nn.init.normal_(p, std=0.1)
    for block in model.blocks:  # remove token mixing: then tau_i can only affect token i
        block.attn.forward = lambda x: torch.zeros_like(x)
    x, y = torch.randn(1, 16, 12), torch.tensor([3])
    tau = torch.full((1, 16), 0.5)
    tau2 = tau.clone()
    tau2[0, 5] = 0.9
    diff = (model(x, tau, y) - model(x, tau2, y)).abs().sum(-1)[0]
    assert diff[5] > 0
    assert torch.all(diff[torch.arange(16) != 5] == 0)
