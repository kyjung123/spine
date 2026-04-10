# Toy MCS vs MCS+CSDA-like study (standalone)

This package is a **local, inspectable, SPINE-independent** toy simulation and benchmark for muon kinetic-energy reconstruction in liquid argon.

## What is borrowed conceptually from SPINE

From SPINE (`src/spine/utils/mcs.py`, `src/spine/post/reco/mcs.py`, `src/spine/utils/energy_loss.py`, and related modules), this toy borrows these ideas:

1. **Energy-loss-aware MCS likelihood**: the fit parameter is initial kinetic energy, and momentum/scattering width changes along the track as energy is depleted.
2. **Segment-level Highland model**: one scattering width per segment derived from local momentum.
3. **Likelihood with normalization term**: not only residual term, but also `log(sigma)` term in NLL.
4. **Track segmentation mindset**: fixed segment length along track.

## What is intentionally simplified here

1. **Energy loss model**: uses a simple parametric `dE/dx(T)` form, not full Bethe-Bloch with all corrections.
2. **Range model**: built numerically from that toy `dE/dx`, not from tabulated PSTAR/CSDA tables.
3. **Scattering observable**: sampled as 1D Gaussian angular deflections with optional additive detector angular resolution.
4. **Geometry/detector**: no voxelization, no full tracking, no endpoint finding, no 3D topology.
5. **CSDA usage for exiting tracks**: no full-contained CSDA estimate unless `f_contained=1`; otherwise only a lower-bound-like constraint on candidate range.

## Operational definitions

- `T0_true`: true initial kinetic energy (MeV).
- `R_true`: total muon range predicted by toy energy-loss model.
- `f_contained`: fraction of total range inside detector.
- `L_in = f_contained * R_true`: in-detector path length used to generate segments.
- Segment length defaults to 10 cm.
- Angles are **projected 1D deflections** (toy), not full 3D angle-pair treatment.

## Reconstruction methods

1. **MCS-only**:
   - Minimize MCS negative log-likelihood over candidate `T0`.
   - Propagate candidate energy down segment-by-segment.
   - Keep hypotheses numerically evaluable even if energy is exhausted early (momentum floor), so MCS-only remains weakly constrained at short containment.

2. **MCS+CSDA-like constraint** (default soft):
   - Same MCS NLL as above.
   - Add penalty if candidate total range is smaller than observed `L_in`:
     - Soft mode (default): quadratic deficit penalty.
     - Hard mode (optional): infeasible if range deficit > 0.

3. **Optional ideal full-CSDA baseline**:
   - Used only for `f_contained ~ 1.0` as a sanity reference.

## Files

- `models.py`: toy physics models (dE/dx, range table, momentum, Highland).
- `simulate.py`: segment-level event generator.
- `reconstruct.py`: MCS-only and MCS+CSDA-like fitting.
- `run_study.py`: grid scan, event-level outputs, metric summaries.
- `plot_results.py`: standard plots for method comparison.
- `notebook_helpers.py`: inline plotting helpers for Jupyter workflows.
- `Toy_MCS_CSDA_Study.ipynb`: ready-to-run notebook for scan + inline plots.
- `outputs/`: CSV and figures.

## Run

```bash
cd toy_mcs_csda_study
python run_study.py
python plot_results.py
```

Optional:

```bash
TOY_N_EVENTS=500 python run_study.py
```

## Jupyter notebook usage

You can run the full study interactively in notebook form:

```bash
cd toy_mcs_csda_study
jupyter notebook Toy_MCS_CSDA_Study.ipynb
```

The notebook uses:
- `run_study.run_study(...)` as an importable API
- `notebook_helpers.py` for inline plots (bias/resolution vs energy/fraction, scatter)

If matplotlib is missing in your notebook environment, install it there:

```bash
pip install matplotlib
```

Default grid:

- `T0_true` in `[200, 400, 800, 1200, 2000]` MeV
- `f_contained` in `[0.2, 0.4, 0.6, 0.8, 1.0]`
- `TOY_N_EVENTS` per grid point (default 250)

## Output products

- `outputs/events.csv`
- `outputs/reco_event_level.csv`
- `outputs/metrics_summary.csv`
- `outputs/likelihood_curves.csv`
- `outputs/method_comparison_summary.csv` (direct `mcs_only` vs `mcs_csda_like_soft`)
- Standard PNG summaries:
  - bias vs true energy (fixed contained fraction)
  - resolution vs true energy (fixed contained fraction)
  - bias vs contained fraction (fixed energy)
  - resolution vs contained fraction (fixed energy)
  - reco-vs-true scatter
  - example likelihood curves

## Notes for extension

- Replace `MuonLossModel` with tabulated stopping power or Bethe-Bloch.
- Move from 1D angles to 2D projected likelihood or full 3D treatment.
- Introduce segment-length optimization and/or heterogeneous angular resolution.
- Add priors or PID-hypothesis switching.
