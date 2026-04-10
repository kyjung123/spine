"""Notebook-friendly helpers for interactive exploration."""

from __future__ import annotations

import numpy as np


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover - convenience path
        raise RuntimeError(
            "matplotlib is required for notebook plotting. "
            "Install it in your notebook environment (e.g. `pip install matplotlib`)."
        ) from exc


def plot_metric_vs_energy_inline(metrics_df, metric="bias_mev", f_contained=0.6):
    require_matplotlib()
    import matplotlib.pyplot as plt

    sub = metrics_df[np.isclose(metrics_df["f_contained"], f_contained)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted(sub["method"].unique()):
        d = sub[sub["method"] == method].sort_values("t0_true_mev")
        ax.plot(d["t0_true_mev"], d[metric], marker="o", label=method)
    ax.set_xlabel("True initial kinetic energy [MeV]")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs energy (f_contained={f_contained:.1f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_metric_vs_fraction_inline(metrics_df, metric="resolution_mev", t0_true_mev=800.0):
    require_matplotlib()
    import matplotlib.pyplot as plt

    sub = metrics_df[np.isclose(metrics_df["t0_true_mev"], t0_true_mev)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted(sub["method"].unique()):
        d = sub[sub["method"] == method].sort_values("f_contained")
        ax.plot(d["f_contained"], d[metric], marker="o", label=method)
    ax.set_xlabel("Contained fraction")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs contained fraction (T0={t0_true_mev:.0f} MeV)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_reco_scatter_inline(reco_df, method="mcs_only"):
    require_matplotlib()
    import matplotlib.pyplot as plt

    sub = reco_df[(reco_df["method"] == method) & np.isfinite(reco_df["t0_reco_mev"])]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["t0_true_mev"], sub["t0_reco_mev"], s=8, alpha=0.35)
    lo = min(sub["t0_true_mev"].min(), sub["t0_reco_mev"].min())
    hi = max(sub["t0_true_mev"].max(), sub["t0_reco_mev"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True T0 [MeV]")
    ax.set_ylabel("Reco T0 [MeV]")
    ax.set_title(f"Reco vs True ({method})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax
