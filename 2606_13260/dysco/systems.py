"""Ground-truth latent dynamical systems.

The minimal reproduction uses the chaotic Lorenz system, the headline example in
the DYSCO paper (Table 2 / Fig. 3). New systems can be added as a vector field
``f(x)`` plus a simulator; the rest of the library is system-agnostic.
"""
from __future__ import annotations

import torch


def lorenz_f(x: torch.Tensor, sigma: float = 10.0, rho: float = 28.0,
             beta: float = 8.0 / 3.0) -> torch.Tensor:
    """Continuous Lorenz vector field.

    Parameters
    ----------
    x:
        States of shape ``(..., 3)``.

    Returns
    -------
    torch.Tensor
        Time derivative ``dx/dt`` of shape ``(..., 3)``.
    """
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return torch.stack([dx1, dx2, dx3], dim=-1)


def simulate_lorenz(T: int, dt: float = 0.01, x0: torch.Tensor | None = None,
                    seed: int = 0) -> torch.Tensor:
    """Euler-integrate a Lorenz trajectory.

    Parameters
    ----------
    T:
        Number of time steps.
    dt:
        Integration step.
    x0:
        Optional initial condition ``(3,)``; randomised near ``(1,1,1)`` if None.
    seed:
        Seed for the initial-condition jitter.

    Returns
    -------
    torch.Tensor
        Latent trajectory of shape ``(T, 3)``.
    """
    g = torch.Generator().manual_seed(seed)
    if x0 is None:
        x0 = torch.tensor([1.0, 1.0, 1.0]) + 0.01 * torch.randn(3, generator=g)
    xs = torch.empty(T, 3)
    x = x0.clone()
    for t in range(T):
        xs[t] = x
        x = x + dt * lorenz_f(x)
    return xs


#: Registry of available systems: name -> (simulator, latent_dim).
SYSTEMS = {
    "lorenz": (simulate_lorenz, 3),
}


def simulate(system: str, T: int, dt: float = 0.01, seed: int = 0) -> torch.Tensor:
    """Simulate a named ground-truth system, returning a ``(T, d)`` trajectory."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose from {list(SYSTEMS)}")
    sim, _ = SYSTEMS[system]
    return sim(T, dt=dt, seed=seed)
