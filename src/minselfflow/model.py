"""Minimal DiT (Diffusion Transformer) with per-token timestep conditioning.

Data flow:
    image (B, 3, 32, 32) --patchify--> tokens (B, N=256, C=12)
    tokens --Linear--> (B, N, D) --DiT blocks--> (B, N, D) --Linear--> velocity (B, N, C)

Differences from the original DiT (Peebles & Xie, 2023), all for Self-Flow:
    * timestep is a vector tau of shape (B, N): each token is conditioned on its own noise level,
      so adaLN modulation is per token (B, N, D) instead of per image (B, D).
    * forward_features also returns every block's hidden states, for the representation loss.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def patchify(img: Tensor, p: int) -> Tensor:
    """Cut images into non-overlapping p x p patches and flatten each patch into a token.

    Args:
        img (Tensor): images, shape (B, 3, H, W).
        p (int): patch size; must divide H and W.

    Returns:
        Tensor: tokens, shape (B, N=(H/p)*(W/p), C=p*p*3), row-major over the patch grid.
    """
    B, ch, H, W = img.shape
    assert H % p == 0 and W % p == 0, (H, W, p)
    x = img.reshape(B, ch, H // p, p, W // p, p)  # split H and W into (grid, within-patch)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, gh, gw, p, p, ch)
    return x.reshape(B, (H // p) * (W // p), p * p * ch)


def unpatchify(x: Tensor, p: int, ch: int = 3) -> Tensor:
    """Inverse of patchify: tokens (B, N, p*p*ch) -> images (B, ch, H, W), assumes a square grid."""
    B, N, C = x.shape
    g = math.isqrt(N)
    assert g * g == N and C == p * p * ch, (N, C, p, ch)
    x = x.reshape(B, g, g, p, p, ch).permute(0, 5, 1, 3, 2, 4)  # (B, ch, gh, p, gw, p)
    return x.reshape(B, ch, g * p, g * p)


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal embedding of timesteps, like positional encoding in the original Transformer.

    Args:
        t (Tensor): timesteps in [0, 1], any shape S.
        dim (int): embedding size (even).

    Returns:
        Tensor: embeddings, shape (*S, dim).
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t[..., None] * 1000.0) * freqs  # scale [0,1] to [0,1000] like DDPM timesteps
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN: x * (1 + scale) + shift; all shapes (B, N, D)."""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention over tokens."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N, D) -> (B, N, D)."""
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, D/heads)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (per-token)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # Conditioning -> 6 modulation vectors: shift/scale/gate for attention and for MLP.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)  # "Zero": gates start at 0, so every block starts as identity
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x: tokens (B, N, D), c: per-token conditioning (B, N, D) -> (B, N, D)."""
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(self.norm1(x), sh1, sc1))
        return x + g2 * self.mlp(modulate(self.norm2(x), sh2, sc2))


@dataclass(frozen=True)
class DiTConfig:
    """DiT hyperparameters. Defaults: ~33M params, CIFAR-10 at patch size 2."""

    in_dim: int = 12  # C = p * p * 3
    num_tokens: int = 256  # N = (32 / 2) ** 2
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 10


class DiT(nn.Module):
    """Token-in, token-out velocity model: (x_tau, tau, y) -> v."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        self.x_embed = nn.Linear(cfg.in_dim, D)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_tokens, D) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.y_embed = nn.Embedding(cfg.num_classes + 1, D)  # last index = "no class", for CFG later
        self.blocks = nn.ModuleList([DiTBlock(D, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        self.final_out = nn.Linear(D, cfg.in_dim)
        for layer in (self.final_ada[1], self.final_out):  # model predicts v=0 at init
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward_features(self, x: Tensor, tau: Tensor, y: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Predict velocity and also return every block's output.

        Args:
            x (Tensor): noisy tokens, shape (B, N, C).
            tau (Tensor): per-token timesteps in [0, 1], shape (B, N).
            y (Tensor): class labels in [0, num_classes], shape (B,); num_classes = unconditional.

        Returns:
            tuple[Tensor, list[Tensor]]: velocity (B, N, C) and `depth` hidden states, each (B, N, D).
        """
        B, N, _ = x.shape
        assert tau.shape == (B, N), (tau.shape, x.shape)
        h = self.x_embed(x) + self.pos_embed
        c = self.t_embed(timestep_embedding(tau, 256)) + self.y_embed(y)[:, None]  # (B, N, D)
        feats = []
        for block in self.blocks:
            h = block(h, c)
            feats.append(h)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.final_out(modulate(self.final_norm(h), shift, scale)), feats

    def forward(self, x: Tensor, tau: Tensor, y: Tensor) -> Tensor:
        """Predict velocity (B, N, C); see forward_features."""
        return self.forward_features(x, tau, y)[0]
