#!/usr/bin/env python3
"""Standalone exiting-muon toy simulation (no SPINE imports).

Implements:
  - CSDA stepping/range in liquid argon (Bethe-Bloch based)
  - MCS-only likelihood fit
  - MCS+CSDA combined fit

This script intentionally does not import from `spine.*`, so you can run it
locally with only NumPy/Pandas/SciPy (and optionally matplotlib for plotting).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import fine_structure
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# -------------------------
# Constants (from SPINE)
# -------------------------
ELEC_MASS = 0.5109989461  # [MeV/c^2]
MUON_MASS = 105.658  # [MeV/c^2]

LAR_DENSITY = 1.396  # [g/cm^3]
LAR_Z = 18
LAR_A = 39.9481  # [g/mol]
LAR_MEE = 188.0e-6  # [MeV]
LAR_X0 = 14.0  # [cm]

LAR_a = 0.19559
LAR_k = 3.0
LAR_x0 = 0.2
LAR_x1 = 3.0
LAR_Cbar = 5.2146
LAR_delta0 = 0.0


def w_max(beta: float, gamma: float, mass: float) -> float:
    bg = beta * gamma
    return (2.0 * ELEC_MASS * bg**2) / (
        1.0 + 2.0 * gamma * ELEC_MASS / mass + (ELEC_MASS / mass) ** 2
    )


def delta_lar(bg: float) -> float:
    x = np.log10(bg)
    if x < LAR_x0:
        return LAR_delta0 * 10.0 ** (2.0 * (x - LAR_x0))
    if x < LAR_x1:
        return 2.0 * np.log(10.0) * x - LAR_Cbar + LAR_a * (LAR_x1 - x) ** LAR_k
    return 2.0 * np.log(10.0) * x - LAR_Cbar


def bethe_bloch_lar(T: float, mass: float, z: int = 1) -> float:
    # Bethe-Bloch constant [MeV*cm^2/mol]
    K = 0.307075

    gamma = 1.0 + T / mass
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    bg = beta * gamma

    F = -K * z**2 * (LAR_Z / LAR_A) * LAR_DENSITY / beta**2
    W = w_max(beta, gamma, mass)
    delta = delta_lar(bg)

    # (Same approximations as SPINE code)
    le_corr = 0.0
    del_dedx = (
        -K
        * fine_structure
        * (LAR_Z / LAR_A)
        / (4.0 * np.pi)
        * (np.log(2.0 * gamma) - (1.0 / 3.0) * np.log(2.0 * W / ELEC_MASS))
        * np.log(2.0 * W / ELEC_MASS) ** 2
    )
    spin_corr_muon = (1.0 / 8.0) * (W / gamma / mass) ** 2 * (mass == MUON_MASS)

    return (
        F
        * (
            0.5 * np.log((2.0 * ELEC_MASS * bg**2 * W) / LAR_MEE**2)
            - beta**2
            - 0.5 * delta
            + le_corr
            + spin_corr_muon
        )
        + del_dedx
    )


def step_energy_loss_lar(T0: float, mass: float, dx: float, num_steps: int) -> np.ndarray:
    assert T0 > 0.0
    out = [T0]
    Ti = T0
    step = 0
    while Ti > 0.0 and step < num_steps:
        step += 1
        Ti += dx * bethe_bloch_lar(Ti, mass)
        out.append(max(Ti, 0.0))
        if Ti <= 0.0:
            break
    return np.asarray(out, dtype=np.float64)


def csda_range_lar(T0: float, mass: float, z: int = 1) -> float:
    if T0 <= 0.0:
        return 0.0

    def inv_bethe(T: float) -> float:
        return -1.0 / bethe_bloch_lar(T, mass, z)

    return quad(inv_bethe, 0.0, T0, epsrel=1e-3, epsabs=1e-3)[0]


def highland(p: np.ndarray, mass: float, dx: float, z: int = 1) -> np.ndarray:
    p = np.maximum(p, 1e-6)
    beta = p / np.sqrt(p**2 + mass**2)
    beta = np.maximum(beta, 1e-6)
    pref = 13.6 * z / (beta * p)
    return pref * (
        np.sqrt(dx / LAR_X0) * (1.0 + 0.038 * np.log(z**2 * dx / LAR_X0 / beta**2))
    )


def mcs_nll(T0: float, theta: np.ndarray, mass: float, dx: float, res_a: float, res_b: float) -> float:
    n_steps = len(theta)
    ke = step_energy_loss_lar(T0, mass, dx, num_steps=n_steps)
    if len(ke) < n_steps + 1:
        return np.inf

    mom = np.sqrt(ke**2 + 2.0 * mass * ke)
    mom_steps = np.sqrt(mom[1:] * mom[:-1])
    theta0 = highland(mom_steps, mass, dx)
    res = res_a / dx**res_b
    scale = np.sqrt(theta0**2 + res**2)
    return np.sum(0.5 * (theta / scale) ** 2 + 2.0 * np.log(scale))


def mcs_fit(theta: np.ndarray, mass: float, dx: float, lower: float = 10.0, upper: float = 100000.0) -> float:
    res = minimize_scalar(
        lambda x: mcs_nll(float(x), theta, mass, dx, 0.25, 1.25),
        bounds=(lower, upper),
        method="bounded",
    )
    return float(res.x)


def mcs_csda_fit(
    theta: np.ndarray,
    mass: float,
    dx: float,
    csda_ke: float,
    csda_ke_frac: float = 0.3,
    csda_weight: float = 2.0,
    lower: float = 10.0,
    upper: float = 100000.0,
) -> float:
    sigma = max(1.0, csda_ke_frac * csda_ke)

    def obj(x: float) -> float:
        prior = 0.5 * csda_weight * ((x - csda_ke) / sigma) ** 2
        return mcs_nll(float(x), theta, mass, dx, 0.25, 1.25) + prior

    res = minimize_scalar(obj, bounds=(lower, upper), method="bounded")
    return float(res.x)


def make_csda_interpolator(grid_max_mev: float = 5000.0, num_points: int = 3000):
    """Build a fast range->KE interpolator once."""
    Ts = np.linspace(1.0, grid_max_mev, num_points)
    Rs = np.array([csda_range_lar(T, MUON_MASS) for T in Ts])

    def _interp(range_cm: float) -> float:
        return float(np.interp(range_cm, Rs, Ts, left=Ts[0], right=Ts[-1]))

    return _interp


@dataclass
class ToyConfig:
    energies: tuple[float, ...] = (200.0, 400.0, 600.0, 800.0, 1000.0)
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    n_toys: int = 300
    dx_cm: float = 5.0
    length_smear_cm: float = 3.0
    seed: int = 20260409


def run_toy(cfg: ToyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    csda_ke_from_range = make_csda_interpolator()
    rows = []
    for T0 in cfg.energies:
        for frac in cfg.fractions:
            for _ in range(cfg.n_toys):
                R_true = csda_range_lar(T0, MUON_MASS)
                L_cont = frac * R_true
                L_reco = max(0.0, L_cont + rng.normal(0.0, cfg.length_smear_cm))
                n_steps = int(np.floor(L_cont / cfg.dx_cm))
                if n_steps < 2:
                    continue

                ke = step_energy_loss_lar(T0, MUON_MASS, cfg.dx_cm, n_steps)
                if len(ke) < n_steps + 1:
                    continue

                p = np.sqrt(ke**2 + 2.0 * MUON_MASS * ke)
                p_steps = np.sqrt(p[:-1] * p[1:])
                if np.any(p_steps <= 0.0):
                    continue
                theta0 = highland(p_steps, MUON_MASS, cfg.dx_cm)
                res = 0.25 / cfg.dx_cm**1.25
                theta = rng.rayleigh(np.sqrt(theta0**2 + res**2))

                csda_ke = csda_ke_from_range(L_reco)
                ke_mcs = mcs_fit(theta, MUON_MASS, cfg.dx_cm)  # no truth-based upper cap
                ke_comb = mcs_csda_fit(theta, MUON_MASS, cfg.dx_cm, csda_ke)

                rows.append((T0, frac, L_cont, L_reco, ke_mcs, ke_comb, csda_ke))

    df = pd.DataFrame(
        rows,
        columns=[
            "T0_true",
            "frac_contained",
            "L_cont_true_cm",
            "L_cont_reco_cm",
            "mcs_ke",
            "comb_ke",
            "csda_ke_proxy",
        ],
    )
    df["mcs_err"] = df["mcs_ke"] - df["T0_true"]
    df["comb_err"] = df["comb_ke"] - df["T0_true"]

    def robust_sigma(x: np.ndarray) -> float:
        q75, q25 = np.percentile(x, [75, 25])
        return (q75 - q25) / 1.349

    summary_rows = []
    for T0 in cfg.energies:
        for frac in cfg.fractions:
            d = df[(df.T0_true == T0) & (df.frac_contained == frac)]
            summary_rows.append(
                {
                    "T0_true": T0,
                    "frac_contained": frac,
                    "N": len(d),
                    "mcs_medae": float(np.median(np.abs(d["mcs_err"]))),
                    "comb_medae": float(np.median(np.abs(d["comb_err"]))),
                    "mcs_robust_sigma": robust_sigma(d["mcs_err"].to_numpy()),
                    "comb_robust_sigma": robust_sigma(d["comb_err"].to_numpy()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["robust_sigma_improve_pct"] = 100.0 * (
        summary["mcs_robust_sigma"] - summary["comb_robust_sigma"]
    ) / summary["mcs_robust_sigma"]

    return df, summary


def maybe_plot(summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not found; skipping plot generation.")
        return

    energies = sorted(summary["T0_true"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for T0 in energies:
        d = summary[summary["T0_true"] == T0]
        ax[0].plot(d["frac_contained"], d["mcs_robust_sigma"], marker="o", label=f"MCS {int(T0)}")
        ax[0].plot(
            d["frac_contained"],
            d["comb_robust_sigma"],
            marker="s",
            linestyle="--",
            label=f"Comb {int(T0)}",
        )
        ax[1].plot(d["frac_contained"], d["robust_sigma_improve_pct"], marker="o", label=f"{int(T0)}")

    ax[0].set_title("Robust sigma vs contained fraction")
    ax[0].set_xlabel("Contained fraction")
    ax[0].set_ylabel("Robust sigma [MeV]")
    ax[0].legend(ncol=2, fontsize=8)

    ax[1].axhline(0.0, color="k", lw=1)
    ax[1].set_title("Improvement of MCS+CSDA")
    ax[1].set_xlabel("Contained fraction")
    ax[1].set_ylabel("Robust sigma improvement [%]")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "standalone_resolution_plots.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="notebooks/artifacts_standalone")
    parser.add_argument("--n-toys", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ToyConfig(n_toys=args.n_toys)
    toys, summary = run_toy(cfg)
    toys.to_csv(out_dir / "standalone_exiting_muon_toys.csv", index=False)
    summary.to_csv(out_dir / "standalone_exiting_muon_summary.csv", index=False)
    maybe_plot(summary, out_dir)

    print("Wrote:")
    print(out_dir / "standalone_exiting_muon_toys.csv")
    print(out_dir / "standalone_exiting_muon_summary.csv")

    # Robust/global quick summary
    m_medae = np.median(np.abs(toys["mcs_err"]))
    c_medae = np.median(np.abs(toys["comb_err"]))
    print(f"Global Median|err|: MCS={m_medae:.2f} MeV, MCS+CSDA={c_medae:.2f} MeV")


if __name__ == "__main__":
    main()
