"""Core physics model helpers for the standalone toy MCS/CSDA study.

This module is intentionally lightweight and SPINE-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# --- Physical constants / defaults ---
MUON_MASS_MEV = 105.658
LAR_DENSITY_G_CM3 = 1.396
LAR_X0_CM = 14.0  # Radiation length in liquid argon (approximate)


@dataclass(frozen=True)
class MuonLossModel:
    """Simple parametric stopping-power model in liquid argon.

    dE/dx is represented by a modestly energy-dependent function that captures
    a broad muon-like trend without aiming for full Bethe-Bloch precision.
    """

    a_mev_cm: float = 2.1
    b_log: float = 0.22
    e_scale_mev: float = 350.0

    def dedx_mev_per_cm(self, kinetic_energy_mev: np.ndarray | float) -> np.ndarray | float:
        """Return approximate stopping power (+dE/dx as positive quantity)."""
        t = np.clip(np.asarray(kinetic_energy_mev, dtype=float), 1e-6, None)
        # Gently increasing dE/dx with energy (toy model)
        return self.a_mev_cm * (1.0 + self.b_log * np.log1p(t / self.e_scale_mev))


class RangeTable:
    """Numerical T->R and R->T mapping for the toy energy-loss model."""

    def __init__(self, loss_model: MuonLossModel, t_max_mev: float = 4000.0, n_grid: int = 6000):
        self.loss_model = loss_model
        self.t_grid = np.linspace(0.0, t_max_mev, n_grid)
        dedx = np.clip(loss_model.dedx_mev_per_cm(self.t_grid), 1e-6, None)
        dr_dT = 1.0 / dedx
        self.r_grid = cumulative_trapezoid(dr_dT, self.t_grid, initial=0.0)

        self._range_of_t = interp1d(
            self.t_grid,
            self.r_grid,
            bounds_error=False,
            fill_value=(0.0, float(self.r_grid[-1])),
        )
        self._t_of_range = interp1d(
            self.r_grid,
            self.t_grid,
            bounds_error=False,
            fill_value=(0.0, float(self.t_grid[-1])),
        )

    @property
    def max_range_cm(self) -> float:
        return float(self.r_grid[-1])

    @property
    def max_energy_mev(self) -> float:
        return float(self.t_grid[-1])

    def range_cm(self, t0_mev: float) -> float:
        return float(self._range_of_t(float(np.clip(t0_mev, 0.0, self.max_energy_mev))))

    def kinetic_energy_from_range(self, range_cm: float) -> float:
        r = float(np.clip(range_cm, 0.0, self.max_range_cm))
        return float(self._t_of_range(r))



def momentum_mev_c(kinetic_energy_mev: np.ndarray | float, mass_mev: float = MUON_MASS_MEV):
    """Relativistic momentum from kinetic energy."""
    t = np.asarray(kinetic_energy_mev, dtype=float)
    return np.sqrt(np.clip(t * (t + 2.0 * mass_mev), 0.0, None))


def beta_from_momentum(p_mev_c: np.ndarray | float, mass_mev: float = MUON_MASS_MEV):
    p = np.asarray(p_mev_c, dtype=float)
    return p / np.sqrt(np.clip(p**2 + mass_mev**2, 1e-12, None))


def highland_theta0(
    p_mev_c: np.ndarray | float,
    segment_length_cm: float,
    z: float = 1.0,
    x0_cm: float = LAR_X0_CM,
    mass_mev: float = MUON_MASS_MEV,
):
    """Highland RMS scattering angle in radians."""
    p = np.asarray(p_mev_c, dtype=float)
    beta = np.clip(beta_from_momentum(p, mass_mev=mass_mev), 1e-5, None)
    x_x0 = np.clip(segment_length_cm / x0_cm, 1e-8, None)
    return (13.6 * z / (beta * np.clip(p, 1e-8, None))) * np.sqrt(x_x0) * (1.0 + 0.038 * np.log(x_x0))


def propagate_segment_energies(
    t0_mev: float,
    n_segments: int,
    segment_length_cm: float,
    loss_model: MuonLossModel,
):
    """Propagate kinetic energy segment-by-segment inside detector.

    Returns energies at segment boundaries of length n_segments+1.
    Values saturate at zero after stopping.
    """
    energies = np.zeros(n_segments + 1, dtype=float)
    energies[0] = max(float(t0_mev), 0.0)
    for i in range(n_segments):
        e = energies[i]
        if e <= 0.0:
            energies[i + 1] = 0.0
            continue
        de = float(loss_model.dedx_mev_per_cm(e) * segment_length_cm)
        energies[i + 1] = max(e - de, 0.0)
    return energies
