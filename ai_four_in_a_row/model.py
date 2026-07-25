"""
AI-written neural network for Connect 4 — no pre-built layer classes.

Everything is built from first principles using raw nn.Parameter weight tensors
and explicit mathematical operations:

  Convolution:   F.conv2d(x, W, padding=p)
                   — slides kernel W over the input; each output pixel is the
                     dot product of W with the local input patch.
  Batch norm:    manual mean/variance → normalise → scale (γ) + shift (β)
  ReLU:          torch.clamp(x, min=0)  — zero out all negative activations
  Linear layer:  x @ W.T + b            — weighted sum of all inputs
  Tanh:          torch.tanh(x)          — squash value output to (−1, +1)
  Skip conn:     x = relu(conv(conv(x)) + x)  — residual learning

The only PyTorch primitives used are:
  nn.Module    — base class so autograd can track parameters
  nn.Parameter — marks a tensor as a trainable weight
  F.conv2d     — the underlying convolution math function (not a layer object)
  basic tensor ops: mean, var, sqrt, clamp, matmul, tanh, reshape
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS = 6
COLS = 7
IN_CHANNELS = 3   # board planes: current player, opponent, empty
HEAD_CHANNELS = 32


def _kaiming_uniform(shape: tuple, fan_in: int) -> torch.Tensor:
    """
    Kaiming (He) uniform initialisation.
    Keeps the variance of activations stable through ReLU layers.

    bound = sqrt(2 / fan_in)   (fan_in = number of input connections)
    weights are sampled from Uniform(−bound, +bound)
    """
    bound = math.sqrt(2.0 / fan_in)
    return torch.empty(shape).uniform_(-bound, bound)


class FourInARowNet(nn.Module):
    """
    Dual-head convolutional network.

    Input:  (batch, 3, 6, 7) board tensor
    Output: policy logits (batch, 7)  — one score per column
            value         (batch,)    — position evaluation in [−1, +1]

    Architecture
    ────────────
    Stem          3×3 conv  IN_CHANNELS → F filters
    Residual ×R   3×3 conv  F → F  (two convs + skip connection)
    Policy head   1×1 conv  F → 32, flatten, linear → 7
    Value head    1×1 conv  F → 32, flatten, linear → 64 → 1
    """

    def __init__(self, num_filters: int = 128, num_res_blocks: int = 6):
        super().__init__()
        F_ = num_filters
        self.num_filters = F_
        self.num_res_blocks = num_res_blocks

        # ── Stem convolution ────────────────────────────────────────────────
        # W shape: (out_channels, in_channels, kH, kW)
        # fan_in for a conv = in_channels × kH × kW
        self.stem_w = nn.Parameter(_kaiming_uniform((F_, IN_CHANNELS, 3, 3), IN_CHANNELS * 9))
        # Batch-norm learnable scale (gamma) and shift (beta) — one per channel
        self.stem_bn_g = nn.Parameter(torch.ones(F_))
        self.stem_bn_b = nn.Parameter(torch.zeros(F_))

        # ── Residual tower ──────────────────────────────────────────────────
        # Each block: conv1 → BN → ReLU → conv2 → BN → (+skip) → ReLU
        for i in range(num_res_blocks):
            setattr(self, f"res{i}_w1",    nn.Parameter(_kaiming_uniform((F_, F_, 3, 3), F_ * 9)))
            setattr(self, f"res{i}_bn1_g", nn.Parameter(torch.ones(F_)))
            setattr(self, f"res{i}_bn1_b", nn.Parameter(torch.zeros(F_)))
            setattr(self, f"res{i}_w2",    nn.Parameter(_kaiming_uniform((F_, F_, 3, 3), F_ * 9)))
            setattr(self, f"res{i}_bn2_g", nn.Parameter(torch.ones(F_)))
            setattr(self, f"res{i}_bn2_b", nn.Parameter(torch.zeros(F_)))

        # ── Policy head ──────────────────────────────────────────────────────
        # 1×1 conv compresses F channels → 32, then a single linear layer
        self.pol_w     = nn.Parameter(_kaiming_uniform((HEAD_CHANNELS, F_, 1, 1), F_))
        self.pol_bn_g  = nn.Parameter(torch.ones(HEAD_CHANNELS))
        self.pol_bn_b  = nn.Parameter(torch.zeros(HEAD_CHANNELS))
        pol_flat       = HEAD_CHANNELS * ROWS * COLS
        self.pol_fc_w  = nn.Parameter(_kaiming_uniform((COLS, pol_flat), pol_flat))
        self.pol_fc_b  = nn.Parameter(torch.zeros(COLS))

        # ── Value head ────────────────────────────────────────────────────────
        self.val_w     = nn.Parameter(_kaiming_uniform((HEAD_CHANNELS, F_, 1, 1), F_))
        self.val_bn_g  = nn.Parameter(torch.ones(HEAD_CHANNELS))
        self.val_bn_b  = nn.Parameter(torch.zeros(HEAD_CHANNELS))
        val_flat       = HEAD_CHANNELS * ROWS * COLS
        self.val_fc1_w = nn.Parameter(_kaiming_uniform((64, val_flat), val_flat))
        self.val_fc1_b = nn.Parameter(torch.zeros(64))
        self.val_fc2_w = nn.Parameter(_kaiming_uniform((1, 64), 64))
        self.val_fc2_b = nn.Parameter(torch.zeros(1))

    # ── Primitive operations ─────────────────────────────────────────────────

    @staticmethod
    def _batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                    eps: float = 1e-5) -> torch.Tensor:
        """
        Batch normalisation for a (batch, C, H, W) tensor.

        1. Per-channel mean  μ  = average over (batch, H, W)
        2. Per-channel variance σ² = average of (x − μ)² over (batch, H, W)
        3. Normalise: x̂ = (x − μ) / √(σ² + ε)
        4. Scale and shift: y = γ·x̂ + β   (γ, β are learned per channel)

        Normalising keeps activations centred and scaled so gradients don't
        vanish or explode as the network deepens.
        """
        mean = x.mean(dim=(0, 2, 3), keepdim=True)          # (1, C, 1, 1)
        var  = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

    @staticmethod
    def _relu(x: torch.Tensor) -> torch.Tensor:
        """
        Rectified Linear Unit: f(x) = max(0, x).
        Zeros out negative values, leaving positives unchanged.
        Introduces non-linearity so the network can learn complex patterns.
        """
        return torch.clamp(x, min=0.0)

    @staticmethod
    def _linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Fully-connected (dense) layer.

        Each output neuron computes a weighted sum of all inputs plus a bias:
            out_i = Σ_j (W_ij · x_j) + b_i

        In matrix form: out = x @ W^T + b
        x: (batch, in_features)
        W: (out_features, in_features)
        b: (out_features,)
        """
        return x @ W.t() + b

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 3, 6, 7)
        """
        # ── Stem ─────────────────────────────────────────────────────────────
        # F.conv2d slides stem_w (shape F×3×3×3) over the board.
        # padding=1 keeps spatial dimensions at 6×7.
        x = F.conv2d(x, self.stem_w, bias=None, padding=1)
        x = self._batch_norm(x, self.stem_bn_g, self.stem_bn_b)
        x = self._relu(x)

        # ── Residual blocks ──────────────────────────────────────────────────
        for i in range(self.num_res_blocks):
            skip = x   # save input so we can add it back (skip connection)

            # First convolution + normalisation + activation
            x = F.conv2d(x, getattr(self, f"res{i}_w1"), bias=None, padding=1)
            x = self._batch_norm(x, getattr(self, f"res{i}_bn1_g"), getattr(self, f"res{i}_bn1_b"))
            x = self._relu(x)

            # Second convolution + normalisation
            x = F.conv2d(x, getattr(self, f"res{i}_w2"), bias=None, padding=1)
            x = self._batch_norm(x, getattr(self, f"res{i}_bn2_g"), getattr(self, f"res{i}_bn2_b"))

            # Skip connection: add the block's input back before activation.
            # This lets gradients flow directly to earlier layers and helps
            # the block learn only the *residual* improvement over its input.
            x = self._relu(x + skip)

        # ── Policy head ──────────────────────────────────────────────────────
        p = F.conv2d(x, self.pol_w, bias=None)               # 1×1 conv → (B, 32, 6, 7)
        p = self._batch_norm(p, self.pol_bn_g, self.pol_bn_b)
        p = self._relu(p)
        p = p.reshape(p.size(0), -1)                          # flatten → (B, 32·6·7)
        policy = self._linear(p, self.pol_fc_w, self.pol_fc_b)  # (B, 7) raw logits

        # ── Value head ───────────────────────────────────────────────────────
        v = F.conv2d(x, self.val_w, bias=None)                # (B, 32, 6, 7)
        v = self._batch_norm(v, self.val_bn_g, self.val_bn_b)
        v = self._relu(v)
        v = v.reshape(v.size(0), -1)                          # (B, 32·6·7)
        v = self._relu(self._linear(v, self.val_fc1_w, self.val_fc1_b))  # (B, 64)
        v = self._linear(v, self.val_fc2_w, self.val_fc2_b)              # (B, 1)
        # tanh squashes the output to (−1, +1): −1 = losing, +1 = winning
        value = torch.tanh(v).squeeze(-1)                                 # (B,)

        return policy, value


def build_model(num_filters: int = 128, num_res_blocks: int = 6) -> FourInARowNet:
    return FourInARowNet(num_filters=num_filters, num_res_blocks=num_res_blocks)
