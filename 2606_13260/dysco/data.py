"""Multi-view, noisy, high-dimensional observation generation.

Builds the data-generating process of the paper: a latent trajectory is mixed
through a fixed nonlinear channel ``g`` and then observed under ``V`` independent
noise realisations ("views"). Multi-view consistency across these independent
noise realisations is the denoising signal DYSCO exploits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DyscoConfig
from .models import RandomMixing
from .systems import simulate


@dataclass
class MultiViewData:
    """A generated dataset.

    Attributes
    ----------
    Y:
        Observations of shape ``(V, T, D)`` — ``V`` independent noisy views.
    x_norm:
        Ground-truth latent trajectory, per-dimension standardised, ``(T, d)``.
    mixing:
        The (frozen) :class:`RandomMixing` channel ``g`` used to generate them.
    """

    Y: torch.Tensor
    x_norm: torch.Tensor
    mixing: RandomMixing


def generate(cfg: DyscoConfig, V: int, system: str = "lorenz",
             device: str = "cpu") -> MultiViewData:
    """Generate ``V`` independent noisy views of a latent trajectory.

    Parameters
    ----------
    cfg:
        Experiment configuration (trajectory length, obs dim, noise, ...).
    V:
        Number of independent views to generate.
    system:
        Name of the ground-truth system (see :mod:`dysco.systems`).
    device:
        Torch device for the returned tensors.
    """
    d, D, dt = cfg.latent_dim, cfg.obs_dim, cfg.dt

    # ground-truth latent trajectory, standardised per dimension
    x_true = simulate(system, cfg.T, dt=dt, seed=cfg.seed).to(device)
    x_norm = (x_true - x_true.mean(0)) / (x_true.std(0) + 1e-8)

    # fixed nonlinear mixing -> clean high-dim signal (standardised)
    g = RandomMixing(d=d, D=D, hidden=cfg.mix_hidden, seed=cfg.seed + 7).to(device)
    with torch.no_grad():
        clean = g(x_norm)
        clean = (clean - clean.mean(0)) / (clean.std(0) + 1e-6)

    # V independent noise realisations of the SAME underlying signal
    views = []
    for v in range(V):
        gen = torch.Generator(device=device).manual_seed(1000 * cfg.seed + v)
        noise = cfg.noise * torch.randn(cfg.T, D, generator=gen, device=device)
        views.append(clean + noise)
    Y = torch.stack(views, dim=0)   # (V, T, D)

    return MultiViewData(Y=Y, x_norm=x_norm, mixing=g)
