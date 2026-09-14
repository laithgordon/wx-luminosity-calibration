# Reproducing the WarpX paper numbers

`reproduce_paper_numbers.py` regenerates every WarpX number reported in the paper from the data committed in this
repository. It writes `paper_numbers.json` (full precision) and `paper_numbers.md` (tables for comparison with the
manuscript). Both files are committed.

## How to run

```
git clone https://github.com/laithgordon/wx-luminosity-calibration.git
cd wx-luminosity-calibration
python reproduce_paper_numbers.py          # about 2 s; needs Python >= 3.10 and numpy
```

Compare `paper_numbers.md` with the manuscript, or run `git diff` after the script to confirm the committed outputs
are reproduced unchanged.

It reads only files under `data/`:
- `results.csv` and `joblist.csv`;
- `R1_table.npz`;
- `gp_exports/gp_requirements_for_wx.csv` and `gp_exports/gp_luminosity_for_wx.csv`, for the GUINEA-PIG++ κ;
- the `slice_widths_hw010_*` and `slice_fiterr_hw010_*` caches.

It uses no network access, no environment variables, and no paths outside the repository.

## What it covers

| section of the output | content |
|---|---|
| `luminosity_dataset` | For each configuration (`nominal`, `conservative`, `frozen_extension`, `extrapolated_extension`) and emittance: mean L, std, se and seed count; the seeds; n_x, n_y, n_z, n_t and n_m as run; the cut multipliers. Each row also gives the selection rule and the runs per phase. |
| `derived_luminosity` | Conservative/nominal per emittance; H_D = L/L_geom per configuration and emittance, with L_geom and its parameters; the conservative gain from 20 to 2 nm; the relative seed scatter per point. |
| `requirements` | n_m^req and n_y^req per emittance: uncertainty on ln n^req, Monte Carlo percentiles, L_inf, bracketing rungs, and whether the point enters the fits. |
| `fits` | The n_m fit (s, C_m); the free and exponent-constrained n_y fits (q, C_y). Each has uncertainty, χ², ndf and its excluded points, plus the same fit with no exclusions. The conservative loci are given as their frozen constants. |
| `kappa` | κ per emittance with uncertainty, for WarpX and for GUINEA-PIG++ (from `data/gp_exports/gp_requirements_for_wx.csv`, `sigma_log` column); for each, the weighted mean of ln κ with its uncertainty and the slope of ln κ against ln D_y with χ² and ndf; κ_WarpX/κ_GP with uncertainty and the agreement in σ. R_1 comes from `data/R1_table.npz` for both codes. |
| `recommendation_table` | n_x, n_y, n_z and n_m from the frozen loci for 0.5–100 nm, next to the settings actually run. |
| `disruption_parameter` | D_y per emittance, the expression, and its parameters. |
| `core_width` | Minimum core width / σ_y* per emittance with uncertainty, the first-waist prediction 1/R_1, their ratio, seed count, and the n_y and n_m run. |
| `solver_and_deposition_variants` | 2D-slice solver, and first-order (CIC) deposition with the 3D and 2D-slice solvers, beside the 3D third-order reference at the same settings. Each has L, seed count, the n_m run and the ratio to the reference. |
| `ps1_reference` | Nominal 20 nm: mean, std, se, seed count, and the ratio to the published 1.35×10³⁴ cm⁻² s⁻¹. |

### Conventions
- **Uncertainty definitions.** Every uncertainty in the JSON carries a `definition` saying what it is: a sample standard deviation over seeds (ddof = 1), a standard error, a Monte Carlo interval half-width on ln n^req, or a fit parameter uncertainty.
- **Same-seed repeats.** Runs repeated at an identical configuration and seed are averaged into one sample before seed statistics; each row states how many were.
- **Exclusions.** Points left out of a fit are listed with the reason:
  - n_m^req at 0.5 and 1 nm, and n_y^req at 0.5 nm: anomalous, shown in the figures but not fitted.
  - Superseded pinched-core runs: off the locus rule.
- **Frozen loci.** The conservative loci are reported as the frozen constants that defined the production runs: n_m^cons = 0.440 D_y^3.269, and n_y^cons = smallest power of two ≥ 38.1 D_y^0.450260. They are not recomputed from the current fit. The fits they were cut from are recomputed alongside for comparison.

## What it deliberately does not cover

- **GUINEA-PIG++ numbers**, including the IPC background table and the n_x convergence study. These are reproduced from the [gp-luminosity-calibration](https://github.com/laithgordon/gp-luminosity-calibration) repository.
- **WarpX/GP++ luminosity ratios.** Each repository's script reproduces its own luminosities, and the ratio follows from the two outputs. The κ comparison is included, from the GP++ export committed here.
- **Material behind repository-only figures and not quoted in the paper**: the coarse-grid control, the n_x–n_m coupling test, the deposition-error correlation study, and the pinch evolution and histograms.
- **Intermediate products**: per-run luminosities, per-step width caches, ladder rungs other than those that define n^req, and settings superseded by the production runs.

## Verification

At commit `858998b` (the code and data this document describes), on a fresh clone with Python 3.13.15 and numpy 2.4.6, all of the following were confirmed.

- **Determinism.** Two consecutive runs give byte-identical `paper_numbers.json` and `paper_numbers.md`, identical to the committed files. A different numpy version could change the last digits of floating-point sums.
- **Internal consistency (611 checks).** Every derived quantity was recomputed from the rows of the same JSON file with an independent implementation. The GP++ κ was recomputed from `data/gp_exports/gp_requirements_for_wx.csv`. The checks cover:
  - standard errors, ratios, H_D, L_geom, D_y, the gain and the scatter;
  - all fits, their χ² and ndf;
  - κ for both codes, the means, slopes, their ratio and the agreement;
  - the recommendation table from the frozen constants;
  - the core-width ratios and the PS1 ratio.

  All agree to floating-point precision. Every uncertainty in the file carries a definition.
- **Agreement with the figure code (292 checks).** The luminosity, fit, κ (both codes) and pinched-core values equal those drawn by `plot_comparison.py`, `plot_deposition.py`, `paper_figures.py` and `plot_core_width.py`. The figure code reads the same committed core-width caches.
- **Fails loudly.** Each of the following stops the script with an error naming what is missing or wrong:
  - deleting a core-width cache;
  - deleting `R1_table.npz`;
  - removing runs from `results.csv`;
  - removing an emittance from the data;
  - deleting the GP++ requirements export;
  - substituting its earlier version, which has a `sigma_tot` column instead of `sigma_log`.

## Limits

- **n_t and cut multipliers** are not stored per run. They follow from the deck, `inputs/input_calib_C3_250.txt`: n_t = n_z, and the domain spans ±16 σ_x, ±16 σ_y, ±8 σ_z in every configuration.
- **Variant emittances.** The solver and deposition variants are reported at all eight emittances run. The emittances the manuscript quotes were not singled out.
- **Rounding.** `paper_numbers.md` rounds to the precision of the values quoted in the manuscript: luminosities, ratios and H_D to two decimals; exponents and κ to three. The table layout and rounding were not compared line by line with the manuscript source. The JSON is authoritative.
- **`paper_numbers.py`** is the earlier summary printer. Its `--json` option writes a different layout, so do not point it at `paper_numbers.json`.
