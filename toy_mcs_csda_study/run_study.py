"""Run the toy MCS vs MCS+CSDA-like benchmark grid.

This module can be used both as:
- a CLI script (`python run_study.py`)
- an importable API from notebooks (`run_study(...)`)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .models import MuonLossModel, RangeTable
    from .reconstruct import (
        csda_like_penalty,
        estimate_energy_ideal_full_csda,
        fit_energy_mcs_csda_like,
        fit_energy_mcs_only,
        mcs_nll,
    )
    from .simulate import SimConfig, ToyMuonSimulator
except ImportError:  # Script-mode fallback
    from models import MuonLossModel, RangeTable  # type: ignore
    from reconstruct import (  # type: ignore
        csda_like_penalty,
        estimate_energy_ideal_full_csda,
        fit_energy_mcs_csda_like,
        fit_energy_mcs_only,
        mcs_nll,
    )
    from simulate import SimConfig, ToyMuonSimulator  # type: ignore


def compute_metrics(df: pd.DataFrame, catastrophic_frac_threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    grp = df.groupby(["method", "t0_true_mev", "f_contained"], as_index=False)
    for (method, t_true, f_cont), sub in grp:
        valid = sub[np.isfinite(sub["t0_reco_mev"])].copy()
        if len(valid) == 0:
            continue
        err = valid["t0_reco_mev"].values - t_true
        frac = err / t_true
        rmse = float(np.sqrt(np.mean(err**2)))
        rows.append(
            {
                "method": method,
                "t0_true_mev": float(t_true),
                "f_contained": float(f_cont),
                "n_events": int(len(sub)),
                "n_valid": int(len(valid)),
                "bias_mev": float(np.mean(err)),
                "fractional_bias": float(np.mean(frac)),
                "resolution_mev": float(np.std(err, ddof=1) if len(err) > 1 else 0.0),
                "fractional_resolution": float(np.std(frac, ddof=1) if len(frac) > 1 else 0.0),
                "rmse_mev": rmse,
                "catastrophic_frac": float(np.mean(np.abs(frac) > catastrophic_frac_threshold)),
                "fit_fail_frac": float(1.0 - (len(valid) / len(sub))),
            }
        )
    return pd.DataFrame(rows)


def compute_method_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Directly compare MCS-only and MCS+CSDA-like summary metrics."""
    cols = [
        "t0_true_mev",
        "f_contained",
        "bias_mev",
        "resolution_mev",
        "rmse_mev",
        "catastrophic_frac",
    ]
    left = metrics_df[metrics_df["method"] == "mcs_only"][cols].copy()
    right = metrics_df[metrics_df["method"] == "mcs_csda_like_soft"][cols].copy()
    merged = left.merge(
        right,
        on=["t0_true_mev", "f_contained"],
        suffixes=("_mcs_only", "_mcs_csda_like_soft"),
        how="inner",
    )
    if merged.empty:
        return merged

    merged["delta_rmse_mev"] = merged["rmse_mev_mcs_csda_like_soft"] - merged["rmse_mev_mcs_only"]
    merged["frac_rmse_improvement"] = (
        merged["rmse_mev_mcs_only"] - merged["rmse_mev_mcs_csda_like_soft"]
    ) / merged["rmse_mev_mcs_only"]
    merged["delta_resolution_mev"] = (
        merged["resolution_mev_mcs_csda_like_soft"] - merged["resolution_mev_mcs_only"]
    )
    merged["delta_bias_mev"] = merged["bias_mev_mcs_csda_like_soft"] - merged["bias_mev_mcs_only"]
    return merged


def run_study(
    true_energies=None,
    contained_fracs=None,
    n_events_per_grid: int | None = None,
    output_dir: Path | None = None,
):
    """Run the benchmark scan and return the output DataFrames."""
    if true_energies is None:
        true_energies = [200.0, 400.0, 800.0, 1200.0, 2000.0]
    if contained_fracs is None:
        contained_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    if n_events_per_grid is None:
        n_events_per_grid = int(os.environ.get("TOY_N_EVENTS", "250"))

    base_dir = Path(__file__).resolve().parent
    out_dir = output_dir if output_dir is not None else (base_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_model = MuonLossModel()
    range_table = RangeTable(loss_model=loss_model, t_max_mev=4500.0, n_grid=7000)
    sim = ToyMuonSimulator(loss_model, range_table, SimConfig(segment_length_cm=10.0, angular_resolution_rad=0.0015, seed=12345))

    event_rows = []
    reco_rows = []
    like_rows = []

    for t_true in true_energies:
        for f_cont in contained_fracs:
            events = sim.simulate_grid_point(t_true, f_cont, n_events=n_events_per_grid)
            for ev in events:
                event_rows.append(
                    {
                        "event_id": ev["event_id"],
                        "t0_true_mev": ev["t0_true_mev"],
                        "f_contained": ev["f_contained"],
                        "r_true_cm": ev["r_true_cm"],
                        "l_in_cm": ev["l_in_cm"],
                        "n_segments": ev["n_segments"],
                    }
                )

                r1 = fit_energy_mcs_only(ev, loss_model)
                r2 = fit_energy_mcs_csda_like(
                    ev,
                    loss_model,
                    range_table,
                    penalty_mode="soft",
                    penalty_sigma_cm=8.0,
                    penalty_weight=2.0,
                )
                r3 = estimate_energy_ideal_full_csda(ev, range_table)

                for rr in (r1, r2, r3):
                    reco_rows.append(
                        {
                            "event_id": ev["event_id"],
                            "t0_true_mev": ev["t0_true_mev"],
                            "f_contained": ev["f_contained"],
                            "l_in_cm": ev["l_in_cm"],
                            "method": rr["method"],
                            "t0_reco_mev": rr["t0_reco_mev"],
                            "success": rr["success"],
                            "objective": rr["fun"],
                        }
                    )

            # Save a few example likelihood curves per grid point
            for ev_ex in events[:2]:
                grid = np.linspace(50.0, 3000.0, 220)
                for t in grid:
                    l_mcs = mcs_nll(t, ev_ex["theta_obs_rad"], ev_ex["segment_length_cm"], loss_model, ev_ex["angular_resolution_rad"])
                    l_soft = l_mcs + csda_like_penalty(
                        t,
                        l_in_cm=ev_ex["l_in_cm"],
                        range_table=range_table,
                        mode="soft",
                        soft_sigma_cm=8.0,
                        weight=2.0,
                    )
                    like_rows.append(
                        {
                            "t0_true_mev": ev_ex["t0_true_mev"],
                            "f_contained": ev_ex["f_contained"],
                            "event_id": ev_ex["event_id"],
                            "t_scan_mev": t,
                            "nll_mcs_only": l_mcs,
                            "nll_mcs_csda_like_soft": l_soft,
                        }
                    )

    event_df = pd.DataFrame(event_rows)
    reco_df = pd.DataFrame(reco_rows)
    metrics_df = compute_metrics(reco_df)

    compare_df = compute_method_comparison(metrics_df)

    event_df.to_csv(out_dir / "events.csv", index=False)
    reco_df.to_csv(out_dir / "reco_event_level.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(like_rows).to_csv(out_dir / "likelihood_curves.csv", index=False)
    compare_df.to_csv(out_dir / "method_comparison_summary.csv", index=False)

    like_df = pd.DataFrame(like_rows)
    return event_df, reco_df, metrics_df, like_df, compare_df, out_dir


def main():
    _, _, _, _, _, out_dir = run_study()
    print("Saved:")
    print(out_dir / "events.csv")
    print(out_dir / "reco_event_level.csv")
    print(out_dir / "metrics_summary.csv")
    print(out_dir / "likelihood_curves.csv")
    print(out_dir / "method_comparison_summary.csv")


if __name__ == "__main__":
    main()
