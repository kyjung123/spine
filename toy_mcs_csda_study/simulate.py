"""Toy segment-level muon simulation for MCS benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .models import (
        MUON_MASS_MEV,
        MuonLossModel,
        RangeTable,
        highland_theta0,
        momentum_mev_c,
        propagate_segment_energies,
    )
except ImportError:  # Script-mode fallback
    from models import (  # type: ignore
        MUON_MASS_MEV,
        MuonLossModel,
        RangeTable,
        highland_theta0,
        momentum_mev_c,
        propagate_segment_energies,
    )


@dataclass
class SimConfig:
    segment_length_cm: float = 10.0
    angular_resolution_rad: float = 0.0015
    seed: int = 12345


class ToyMuonSimulator:
    def __init__(self, loss_model: MuonLossModel, range_table: RangeTable, config: SimConfig):
        self.loss_model = loss_model
        self.range_table = range_table
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def simulate_event(self, t0_true_mev: float, f_contained: float, event_id: int = 0) -> dict:
        r_true = self.range_table.range_cm(t0_true_mev)
        l_in = float(np.clip(f_contained, 0.0, 1.0) * r_true)

        n_segments = max(3, int(np.floor(l_in / self.config.segment_length_cm)))
        l_in_eff = n_segments * self.config.segment_length_cm

        energies = propagate_segment_energies(
            t0_true_mev, n_segments, self.config.segment_length_cm, self.loss_model
        )
        p_edges = momentum_mev_c(energies, MUON_MASS_MEV)
        p_seg = np.sqrt(np.clip(p_edges[:-1] * p_edges[1:], 0.0, None))

        theta0_true = highland_theta0(
            p_seg,
            segment_length_cm=self.config.segment_length_cm,
            mass_mev=MUON_MASS_MEV,
        )
        sigma_obs = np.sqrt(theta0_true**2 + self.config.angular_resolution_rad**2)
        theta_obs = self.rng.normal(loc=0.0, scale=sigma_obs, size=n_segments)

        return {
            "event_id": event_id,
            "t0_true_mev": float(t0_true_mev),
            "f_contained": float(f_contained),
            "r_true_cm": float(r_true),
            "l_in_cm": float(l_in_eff),
            "n_segments": int(n_segments),
            "theta_obs_rad": theta_obs,
            "theta0_true_rad": theta0_true,
            "sigma_obs_rad": sigma_obs,
            "segment_length_cm": float(self.config.segment_length_cm),
            "angular_resolution_rad": float(self.config.angular_resolution_rad),
        }

    def simulate_grid_point(self, t0_true_mev: float, f_contained: float, n_events: int):
        return [self.simulate_event(t0_true_mev, f_contained, event_id=i) for i in range(n_events)]
