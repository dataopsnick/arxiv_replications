"""Evaluation metrics: affine alignment + R^2 for trajectory and flow field.

Because identifiability holds only up to an affine indeterminacy (Theorem 1),
both the recovered latent trajectory and the recovered flow field are compared
to ground truth *after* fitting the optimal affine map.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .models import Encoder, SymbolicDynamics


def affine_fit(pred: torch.Tensor,
               target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares affine map ``L, b`` minimising ``||L pred + b - target||``.

    Parameters
    ----------
    pred, target:
        Tensors of shape ``(N, d)``.

    Returns
    -------
    aligned, L, b:
        The affinely-aligned ``pred`` ``(N, d)``, the linear map ``L`` ``(d, d)``
        and the offset ``b`` ``(d,)``.
    """
    N = pred.shape[0]
    P = torch.cat([pred, torch.ones(N, 1, device=pred.device)], dim=1)
    sol, *_ = torch.linalg.lstsq(P, target)
    aligned = P @ sol
    L = sol[:-1].t()
    b = sol[-1]
    return aligned, L, b


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination ``R^2`` between ``pred`` and ``target``."""
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(0, keepdim=True)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def evaluate(encoder: Encoder, dynamics: SymbolicDynamics,
             Y: torch.Tensor, x_norm: torch.Tensor, dt: float) -> dict:
    """Compute latent-trajectory and flow-field recovery metrics.

    Parameters
    ----------
    encoder, dynamics:
        The trained models.
    Y:
        Observations ``(V, T, D)``; view 0 is used for the read-out.
    x_norm:
        Ground-truth standardised latent trajectory ``(T, d)``.
    dt:
        Integration step (to convert the learned velocity to per-step units).

    Returns
    -------
    dict
        ``{"latentR2": float, "dynR2": float}``.
    """
    encoder.eval()
    dynamics.eval()
    with torch.no_grad():
        # latent trajectory recovery (affinely aligned)
        z = encoder(Y[0])                              # (T, d)
        aligned, L, _ = affine_fit(z, x_norm)
        latent_r2 = r2_score(aligned, x_norm)

        # flow-field recovery: compare true per-step velocity to the
        # affinely-conjugated learned flow field.
        true_vel = x_norm[1:] - x_norm[:-1]            # (T-1, d)
        learned_vel = dynamics(z[:-1]) @ L.t() * dt    # conjugate to x-coords
        dyn_r2 = r2_score(learned_vel, true_vel)

    return {"latentR2": latent_r2, "dynR2": dyn_r2}
