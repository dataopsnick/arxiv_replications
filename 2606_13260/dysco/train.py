"""The DYSCO multi-view temporal contrastive training loop.

This implements the core mechanism of the paper (the score of Eq. 6 trained with
an InfoNCE objective):

* sample an anchor observation at time ``t`` from view ``a`` and a positive at
  time ``t + k`` from an *independent* view ``b``;
* encode both, roll the anchor forward ``k`` steps with the learned symbolic
  dynamics, and require it to match the positive;
* contrast against the other ``B - 1`` positives in the batch as negatives.

With ``V > 1`` the anchor and positive carry *independent* observation noise, so
matching them forces the encoder to discard per-view noise and keep the shared
signal — the multi-view denoising effect. With ``V == 1`` the cross-view signal
collapses (at ``k = 0`` the positive is the identical observation), so single-view
training cannot exploit it. This asymmetry is exactly what the paper exploits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import DyscoConfig
from .data import generate, MultiViewData
from .evaluate import evaluate
from .models import Encoder, SymbolicDynamics, rollout


@dataclass
class ViewResult:
    """Result of training + evaluating a single view count.

    Attributes
    ----------
    V:
        Number of views used.
    latentR2:
        R^2 of the affinely-aligned recovered latent trajectory.
    dynR2:
        R^2 of the affinely-conjugated recovered flow field.
    final_loss:
        InfoNCE loss at the last logged step.
    history:
        List of ``(step, loss)`` tuples logged during training.
    encoder, dynamics:
        The trained models (kept on the training device).
    """

    V: int
    latentR2: float
    dynR2: float
    final_loss: float
    history: List[Tuple[int, float]] = field(default_factory=list)
    encoder: Encoder | None = None
    dynamics: SymbolicDynamics | None = None

    def to_dict(self) -> dict:
        return {
            "V": self.V,
            "latentR2": self.latentR2,
            "dynR2": self.dynR2,
            "final_loss": self.final_loss,
            "history": self.history,
        }


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_views(cfg: DyscoConfig, V: int, system: str = "lorenz",
                device: str = "cpu", data: MultiViewData | None = None,
                verbose: bool = True) -> ViewResult:
    """Train and evaluate a DYSCO model with ``V`` views.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    V:
        Number of independent views.
    system:
        Ground-truth system name.
    device:
        Torch device.
    data:
        Optional pre-generated :class:`~dysco.data.MultiViewData`; generated from
        ``cfg`` if not supplied.
    verbose:
        Print per-log-step loss if True.

    Returns
    -------
    ViewResult
    """
    _seed_everything(cfg.seed)
    if data is None:
        data = generate(cfg, V, system=system, device=device)
    Y, x_norm = data.Y, data.x_norm

    d = cfg.latent_dim
    enc = Encoder(D=cfg.obs_dim, d=d, hidden=cfg.enc_hidden).to(device)
    fhat = SymbolicDynamics(d=d).to(device)
    params = list(enc.parameters()) + list(fhat.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    T, kappa, tau, B, dt = cfg.T, cfg.kappa, cfg.temp, cfg.batch, cfg.dt
    steps = cfg.steps_for_views(V)

    history: List[Tuple[int, float]] = []
    for step in range(steps):
        opt.zero_grad()

        # Temporal horizon: k=0 is pure same-time cross-view consistency
        # (denoising); k>0 adds the dynamics roll-out. Mixing both teaches the
        # model to denoise across views AND obey the symbolic dynamics over time.
        k = int(torch.randint(0, kappa + 1, (1,)).item())
        t = torch.randint(0, T - max(k, 1), (B,), device=device)

        if V == 1:
            va = torch.zeros(B, dtype=torch.long, device=device)
            vb = torch.zeros(B, dtype=torch.long, device=device)
        else:
            va = torch.randint(0, V, (B,), device=device)
            offset = torch.randint(1, V, (B,), device=device)  # ensure b != a
            vb = (va + offset) % V

        y_a = Y[va, t]            # (B, D)
        y_b = Y[vb, t + k]        # (B, D)

        z_a = enc(y_a)
        z_b = enc(y_b)
        z_a_fwd = rollout(fhat, z_a, k, dt) if k > 0 else z_a

        # InfoNCE with negative-squared-Euclidean similarity / temperature
        diff = z_a_fwd.unsqueeze(1) - z_b.unsqueeze(0)     # (B, B, d)
        sim = -(diff ** 2).sum(-1) / tau                  # (B, B)
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(sim, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == steps - 1:
            history.append((step, float(loss.item())))
            if verbose:
                print(f"  [V={V}] step {step:5d}  loss {loss.item():.4f}",
                      flush=True)

    metrics = evaluate(enc, fhat, Y, x_norm, dt)
    return ViewResult(
        V=V,
        latentR2=metrics["latentR2"],
        dynR2=metrics["dynR2"],
        final_loss=history[-1][1] if history else float("nan"),
        history=history,
        encoder=enc,
        dynamics=fhat,
    )
