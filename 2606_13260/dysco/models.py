"""Neural and symbolic models used by DYSCO.

* :class:`RandomMixing` — the fixed nonlinear observation channel ``g`` that maps
  the low-dimensional latent state to high-dimensional observations.
* :class:`Encoder` — the learnable de-mixing function ``h`` (inverse of ``g``).
* :class:`SymbolicDynamics` — the learnable governing-equation model
  ``f_hat(x) = Theta @ Xi(x)`` with a polynomial library ``Xi``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RandomMixing(nn.Module):
    """Fixed, randomly-initialised 4-layer MLP mixing function ``g: R^d -> R^D``.

    Represents the nonlinear, injective observation channel of the paper. Its
    weights are frozen (``requires_grad_(False)``): it is part of the
    data-generating process, not something the model learns.
    """

    def __init__(self, d: int = 3, D: int = 32, hidden: int = 64, seed: int = 1):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        dims = [d, hidden, hidden, hidden, D]
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            with torch.no_grad():
                lin.weight.copy_(
                    torch.randn(dims[i + 1], dims[i], generator=gen)
                    / math.sqrt(dims[i]))
                lin.bias.copy_(0.1 * torch.randn(dims[i + 1], generator=gen))
            layers.append(lin)
        self.layers = nn.ModuleList(layers)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class Encoder(nn.Module):
    """Learnable de-mixing encoder ``h: R^D -> R^d`` (4-layer MLP, GELU)."""

    def __init__(self, D: int = 32, d: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def poly_library(x: torch.Tensor) -> torch.Tensor:
    """Degree-2 polynomial features for a 3-D state.

    ``x`` of shape ``(..., 3)`` maps to ``(..., 10)`` features
    ``[1, x1, x2, x3, x1^2, x1 x2, x1 x3, x2^2, x2 x3, x3^2]``.
    """
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    ones = torch.ones_like(x1)
    feats = [ones, x1, x2, x3,
             x1 * x1, x1 * x2, x1 * x3,
             x2 * x2, x2 * x3, x3 * x3]
    return torch.stack(feats, dim=-1)


#: Names of the degree-2 library terms, in order (handy for printing equations).
POLY_TERMS = ["1", "x1", "x2", "x3", "x1^2", "x1*x2", "x1*x3",
              "x2^2", "x2*x3", "x3^2"]
N_LIB = len(POLY_TERMS)


class SymbolicDynamics(nn.Module):
    """Learnable governing-equation model ``f_hat(x) = Theta @ Xi(x)``.

    ``Theta`` is a ``(d, n)`` coefficient matrix over the polynomial library
    ``Xi``; this structured parameterisation is what enables (approximate)
    symbolic recovery of the dynamics.
    """

    def __init__(self, d: int = 3, n: int = N_LIB):
        super().__init__()
        self.Theta = nn.Parameter(0.01 * torch.randn(d, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = poly_library(x)            # (..., n)
        return feats @ self.Theta.t()      # (..., d)


def rk4_step(fhat: nn.Module, x: torch.Tensor, dt: float) -> torch.Tensor:
    """Single RK4 integration step of the learned dynamics ``fhat``."""
    k1 = fhat(x)
    k2 = fhat(x + 0.5 * dt * k1)
    k3 = fhat(x + 0.5 * dt * k2)
    k4 = fhat(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(fhat: nn.Module, x: torch.Tensor, k: int, dt: float) -> torch.Tensor:
    """Integrate ``x`` forward ``k`` steps with the learned dynamics."""
    for _ in range(k):
        x = rk4_step(fhat, x, dt)
    return x
