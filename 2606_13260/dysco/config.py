"""Configuration objects for DYSCO experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class DyscoConfig:
    """All hyperparameters for a single DYSCO training / evaluation run.

    A single :class:`DyscoConfig` describes the data-generating process (the
    latent system, the nonlinear mixing, the observation noise) and the
    training setup. It is shared across the view counts compared in an
    experiment; only the number of views ``V`` varies between configurations.

    Attributes
    ----------
    T:
        Length of the simulated latent trajectory (number of time steps).
    obs_dim:
        Dimensionality ``D`` of the high-dimensional observation channel.
    latent_dim:
        Dimensionality ``d`` of the latent state (3 for Lorenz).
    dt:
        Integration time step used for both the ground-truth simulation and the
        learned-dynamics RK4 roll-outs.
    noise:
        Standard deviation ``sigma`` of the additive Gaussian observation noise.
    steps:
        Base number of optimisation steps (for ``V == 1``). Larger view counts
        are given proportionally more steps; see ``fair_budget``.
    batch:
        Contrastive mini-batch size (also the number of InfoNCE negatives).
    lr:
        AdamW learning rate.
    weight_decay:
        AdamW weight decay.
    kappa:
        Maximum temporal integration horizon for the roll-out objective.
    temp:
        InfoNCE temperature (scales the negative-squared-distance similarity).
    seed:
        Random seed for data generation and model initialisation.
    enc_hidden:
        Encoder MLP hidden width.
    mix_hidden:
        Random-mixing MLP hidden width.
    poly_degree:
        Degree of the polynomial dynamics library (currently 2 is supported).
    grad_clip:
        Gradient-norm clipping value.
    fair_budget:
        If ``True``, scale ``steps`` with the number of views so larger-V models
        are not starved of optimisation budget (a fair ablation rather than a
        budget artifact).
    log_every:
        How often (in steps) to record the training loss.
    """

    # --- data-generating process ---
    T: int = 5000
    obs_dim: int = 32
    latent_dim: int = 3
    dt: float = 0.01
    noise: float = 1.5

    # --- optimisation ---
    steps: int = 6000
    batch: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    kappa: int = 4
    temp: float = 1.0
    seed: int = 0
    grad_clip: float = 5.0

    # --- architecture ---
    enc_hidden: int = 128
    mix_hidden: int = 64
    poly_degree: int = 2

    # --- bookkeeping ---
    fair_budget: bool = True
    log_every: int = 500

    def steps_for_views(self, V: int) -> int:
        """Number of optimisation steps to use for ``V`` views.

        With ``V`` views there are ``V*(V-1)`` ordered cross-view pairs, so a
        fixed step count lets larger-V models see each view-pair fewer times and
        underfit. When ``fair_budget`` is set, give more views proportionally
        more steps (capped at 2x) so the view sweep is a fair ablation.
        """
        if not self.fair_budget or V <= 1:
            return self.steps
        return int(self.steps * min(2.0, 1.0 + 0.25 * (V - 1)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DyscoConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})
