"""Plot helpers for the toy MCS/CSDA benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_metric_vs_energy(metrics: pd.DataFrame, metric: str, f_contained: float, out_dir: Path):
    _ensure_out(out_dir)
    sub = metrics[np.isclose(metrics["f_contained"], f_contained)]
    if sub.empty:
        return
    plt.figure(figsize=(7, 5))
    for method in sorted(sub["method"].unique()):
        d = sub[sub["method"] == method].sort_values("t0_true_mev")
        plt.plot(d["t0_true_mev"], d[metric], marker="o", label=method)
    plt.xlabel("True initial kinetic energy [MeV]")
    plt.ylabel(metric)
    plt.title(f"{metric} vs true energy (f_contained={f_contained:.1f})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_vs_energy_f{f_contained:.1f}.png", dpi=140)
    plt.close()


def plot_metric_vs_fraction(metrics: pd.DataFrame, metric: str, t0_true_mev: float, out_dir: Path):
    _ensure_out(out_dir)
    sub = metrics[np.isclose(metrics["t0_true_mev"], t0_true_mev)]
    if sub.empty:
        return
    plt.figure(figsize=(7, 5))
    for method in sorted(sub["method"].unique()):
        d = sub[sub["method"] == method].sort_values("f_contained")
        plt.plot(d["f_contained"], d[metric], marker="o", label=method)
    plt.xlabel("Contained fraction")
    plt.ylabel(metric)
    plt.title(f"{metric} vs contained fraction (T0={t0_true_mev:.0f} MeV)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_vs_fraction_t{int(t0_true_mev)}.png", dpi=140)
    plt.close()


def plot_reco_scatter(reco: pd.DataFrame, method: str, out_dir: Path):
    _ensure_out(out_dir)
    sub = reco[(reco["method"] == method) & np.isfinite(reco["t0_reco_mev"])]
    if sub.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(sub["t0_true_mev"], sub["t0_reco_mev"], s=8, alpha=0.35)
    lo = min(sub["t0_true_mev"].min(), sub["t0_reco_mev"].min())
    hi = max(sub["t0_true_mev"].max(), sub["t0_reco_mev"].max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True T0 [MeV]")
    plt.ylabel("Reco T0 [MeV]")
    plt.title(f"Reco vs True ({method})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_reco_vs_true_{method}.png", dpi=140)
    plt.close()


def plot_likelihood_examples(like: pd.DataFrame, out_dir: Path):
    _ensure_out(out_dir)
    # Select a few unique events
    keys = like[["t0_true_mev", "f_contained", "event_id"]].drop_duplicates().head(6)
    for _, row in keys.iterrows():
        sub = like[
            (like["t0_true_mev"] == row["t0_true_mev"])
            & (like["f_contained"] == row["f_contained"])
            & (like["event_id"] == row["event_id"])
        ].sort_values("t_scan_mev")
        if sub.empty:
            continue
        y1 = sub["nll_mcs_only"].values
        y2 = sub["nll_mcs_csda_like_soft"].values
        y1 = y1 - np.nanmin(y1)
        y2 = y2 - np.nanmin(y2)
        plt.figure(figsize=(7, 5))
        plt.plot(sub["t_scan_mev"], y1, label="MCS-only")
        plt.plot(sub["t_scan_mev"], y2, label="MCS+CSDA-like")
        plt.axvline(row["t0_true_mev"], color="k", linestyle="--", lw=1, label="True T0")
        plt.xlabel("Candidate T0 [MeV]")
        plt.ylabel("Shifted NLL")
        plt.title(f"Likelihood example: T0={row['t0_true_mev']:.0f} MeV, f={row['f_contained']:.1f}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        name = f"likelihood_example_t{int(row['t0_true_mev'])}_f{row['f_contained']:.1f}_e{int(row['event_id'])}.png"
        plt.savefig(out_dir / name, dpi=140)
        plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"

    metrics = pd.read_csv(out_dir / "metrics_summary.csv")
    reco = pd.read_csv(out_dir / "reco_event_level.csv")
    like = pd.read_csv(out_dir / "likelihood_curves.csv")

    for f in sorted(metrics["f_contained"].unique()):
        plot_metric_vs_energy(metrics, "bias_mev", float(f), out_dir)
        plot_metric_vs_energy(metrics, "resolution_mev", float(f), out_dir)

    for t in sorted(metrics["t0_true_mev"].unique()):
        plot_metric_vs_fraction(metrics, "bias_mev", float(t), out_dir)
        plot_metric_vs_fraction(metrics, "resolution_mev", float(t), out_dir)

    for method in ["mcs_only", "mcs_csda_like_soft"]:
        plot_reco_scatter(reco, method, out_dir)

    plot_likelihood_examples(like, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
