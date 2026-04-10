"""Reconstruction methods for the standalone toy MCS/CSDA study."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from .models import MUON_MASS_MEV, MuonLossModel, RangeTable, highland_theta0, momentum_mev_c, propagate_segment_energies
except ImportError:  # Script-mode fallback
    from models import (  # type: ignore
        MUON_MASS_MEV,
        MuonLossModel,
        RangeTable,
        highland_theta0,
        momentum_mev_c,
        propagate_segment_energies,
    )


def mcs_nll(
    t0_candidate_mev: float,
    theta_obs_rad: np.ndarray,
    segment_length_cm: float,
    loss_model: MuonLossModel,
    angular_resolution_rad: float,
) -> float:
    """Gaussian-angle negative log-likelihood with full normalization term."""
    if t0_candidate_mev <= 0.0:
        return np.inf

    n_segments = len(theta_obs_rad)
    energies = propagate_segment_energies(t0_candidate_mev, n_segments, segment_length_cm, loss_model)

    p_edges = momentum_mev_c(energies, MUON_MASS_MEV)
    # For MCS-only, keep hypotheses numerically evaluable even if energy is
    # exhausted early by flooring the momentum, so the CSDA-like term can carry
    # the path-length consistency information.
    p_seg = np.sqrt(np.clip(p_edges[:-1] * p_edges[1:], 30.0**2, None))
    theta0 = highland_theta0(p_seg, segment_length_cm=segment_length_cm, mass_mev=MUON_MASS_MEV)
    sigma = np.sqrt(theta0**2 + angular_resolution_rad**2)

    # Include normalization: 0.5*(theta/sigma)^2 + log(sigma)
    return float(np.sum(0.5 * (theta_obs_rad / sigma) ** 2 + np.log(sigma)))


def csda_like_penalty(
    t0_candidate_mev: float,
    l_in_cm: float,
    range_table: RangeTable,
    mode: str = "soft",
    soft_sigma_cm: float = 20.0,
    weight: float = 1.0,
) -> float:
    """Constraint from in-detector path length.

    - soft: quadratic penalty for deficit in total allowed range
    - hard: infinite penalty if total range is smaller than in-detector length
    """
    r_cand = range_table.range_cm(t0_candidate_mev)
    deficit = l_in_cm - r_cand
    if mode == "hard":
        return np.inf if deficit > 0.0 else 0.0
    if mode != "soft":
        raise ValueError(f"Unknown CSDA-like penalty mode: {mode}")
    if deficit <= 0.0:
        return 0.0
    return float(weight * 0.5 * (deficit / soft_sigma_cm) ** 2)


def fit_energy_mcs_only(event: dict, loss_model: MuonLossModel, t_bounds=(20.0, 3500.0)) -> dict:
    obj = lambda t: mcs_nll(
        t,
        theta_obs_rad=event["theta_obs_rad"],
        segment_length_cm=event["segment_length_cm"],
        loss_model=loss_model,
        angular_resolution_rad=event["angular_resolution_rad"],
    )
    res = minimize_scalar(obj, bounds=t_bounds, method="bounded")
    return {
        "method": "mcs_only",
        "t0_reco_mev": float(res.x),
        "success": bool(res.success and np.isfinite(res.fun)),
        "fun": float(res.fun),
    }


def fit_energy_mcs_csda_like(
    event: dict,
    loss_model: MuonLossModel,
    range_table: RangeTable,
    t_bounds=(20.0, 3500.0),
    penalty_mode: str = "soft",
    penalty_sigma_cm: float = 20.0,
    penalty_weight: float = 1.0,
) -> dict:
    def obj(t):
        base = mcs_nll(
            t,
            theta_obs_rad=event["theta_obs_rad"],
            segment_length_cm=event["segment_length_cm"],
            loss_model=loss_model,
            angular_resolution_rad=event["angular_resolution_rad"],
        )
        if not np.isfinite(base):
            return np.inf
        pen = csda_like_penalty(
            t,
            l_in_cm=event["l_in_cm"],
            range_table=range_table,
            mode=penalty_mode,
            soft_sigma_cm=penalty_sigma_cm,
            weight=penalty_weight,
        )
        return base + pen

    res = minimize_scalar(obj, bounds=t_bounds, method="bounded")
    return {
        "method": f"mcs_csda_like_{penalty_mode}",
        "t0_reco_mev": float(res.x),
        "success": bool(res.success and np.isfinite(res.fun)),
        "fun": float(res.fun),
    }


def estimate_energy_ideal_full_csda(event: dict, range_table: RangeTable) -> dict:
    """Optional baseline valid only for fully contained tracks in this toy."""
    if event["f_contained"] < 0.999:
        return {
            "method": "ideal_full_csda",
            "t0_reco_mev": np.nan,
            "success": False,
            "fun": np.nan,
        }
    t_est = range_table.kinetic_energy_from_range(event["l_in_cm"])
    return {
        "method": "ideal_full_csda",
        "t0_reco_mev": float(t_est),
        "success": True,
        "fun": 0.0,
    }
