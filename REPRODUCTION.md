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

Compare `paper_numbers.md` with the manuscript, or run `git diff` after the script to compare with the committed outputs.

The committed `paper_numbers.json` is reproduced to floating-point last-digit precision: a few values can differ in their final digit between platforms, and no quoted value depends on those digits.

It reads only files under `data/`:
- `results.csv` and `joblist.csv`;
- `R1_table.npz`;
- `gp_exports/gp_requirements_for_wx.csv` and `gp_exports/gp_luminosity_for_wx.csv`, for the GUINEA-PIG++ κ and luminosities;
- `nm_vs_nx_ladders.csv`, for the n_m ladders at fixed n_x (through `nx_ladders.py`);
- the `slice_widths_hw010_*` and `slice_fiterr_hw010_*` caches, and their estimator variations
  `slice_widths_hw<HW>_thr<threshold>_nb<bins>_*` and `slice_fiterr_hw<HW>_thr<threshold>_nb<bins>_*` (through `core_width_syst.py`).

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
| `core_width_systematic` | The core width under each estimator variation (slice half-width 0.05 and 0.20 σ_z, fit threshold 0.10 and 0.30 of the peak, bin width ×0.5 and ×2), with each deviation and its seed-paired se; the systematic per emittance (maximum and RMS deviation) and the total se ⊕ systematic; the weighted fit of ln(σ_y*/σ_min) = a + q ln D_y over 0.5–20 nm and 1–20 nm (q, Λ = e^a, χ², ndf, p, pulls against q_p with and without its uncertainty); the same fit on each variation's points; the slope of ln(extracted/predicted) vs ln D_y; R and R_1 at 0.5 and 1 nm for every estimator. |
| `solver_and_deposition_variants` | 2D-slice solver, and first-order (CIC) deposition with the 3D and 2D-slice solvers, beside the 3D third-order reference at the same settings. Each has L, seed count, the n_m run and the ratio to the reference. |
| `ps1_reference` | Nominal 20 nm: mean, std, se, seed count, and the ratio to the published 1.35×10³⁴ cm⁻² s⁻¹. |
| `wx_over_gp_luminosity` | L_WarpX/L_GP++ per emittance with se, each simulator at its recommended configuration (WarpX `conservative` and `extrapolated_extension`; GUINEA-PIG++ `conservative` and `frozen_extension` from `data/gp_exports/gp_luminosity_for_wx.csv`); its range over 8–100 nm and its value at 1 nm; and, separately, the ratio with both simulators frozen at their 20 nm configurations over 40–100 nm. |
| `nx_ladders_20nm` | Per n_x (64–8192) at 20 nm: the plateau L_inf where one exists and its ratio to n_x = 512, the top rung and its ratio, the side of approach, n_m^req or its lower bound against eq:law_nm, and the spread of L_inf over n_x = 256–1024. |

### Conventions
- **Uncertainty definitions.** Every uncertainty in the JSON carries a `definition` saying what it is: a sample standard deviation over seeds (ddof = 1), a standard error, a Monte Carlo interval half-width on ln n^req, or a fit parameter uncertainty.
- **Same-seed repeats.** Runs repeated at an identical configuration and seed are averaged into one sample before seed statistics; each row states how many were.
- **Exclusions.** Points left out of a fit are listed with the reason:
  - n_m^req at 0.5 and 1 nm, and n_y^req at 0.5 nm: anomalous, shown in the figures but not fitted.
  - Pinched-core runs off the locus rule.
- **Frozen loci.** The conservative loci are reported as the frozen constants that defined the production runs: n_m^cons = 0.440 D_y^3.269, and n_y^cons = smallest power of two ≥ 38.1 D_y^0.450260. They are not recomputed from the current fit. The fits they were cut from are recomputed alongside for comparison.

### Core-width caches

The core-width caches are committed, so the script needs no dumps. To regenerate any that are missing from the
per-step particle dumps (one read of each dump writes every estimator setting):

```
WX_RUN_ROOT=/path/to/run/directories python core_width_syst.py --build --workers 16
```

`python core_width_syst.py` without `--build` prints the systematic and fit tables from the committed caches, and
`WX_CACHED_ONLY=1 python make_figures.py` redraws `plots/WX_core_width_vs_Dy` from the same numbers.

## What it deliberately does not cover

- **GUINEA-PIG++ numbers**, including the IPC background table and the n_x convergence study. These are reproduced from the [gp-luminosity-calibration](https://github.com/laithgordon/gp-luminosity-calibration) repository.
- **Material behind repository-only figures and not quoted in the paper**: the coarse-grid control, the n_x–n_m coupling test, the deposition-error correlation study, and the pinch evolution and histograms.
- **Intermediate products**: per-run luminosities, per-step width caches, ladder rungs other than those that define n^req, and settings other than those of the production runs.

## Verification

At the commit containing this document, on a fresh clone with Python 3.13.15 and numpy 2.4.6, all of the following were confirmed.

- **Determinism.** Two consecutive runs give byte-identical `paper_numbers.json` and `paper_numbers.md`.
- **Internal consistency (1021 checks).** Every derived quantity was recomputed from the rows of the same JSON file with an independent implementation. The GP++ κ was recomputed from `data/gp_exports/gp_requirements_for_wx.csv`. The checks cover:
  - standard errors, ratios, H_D, L_geom, D_y, the gain and the scatter;
  - all fits, their χ² and ndf;
  - κ for both codes, the means, slopes, their ratio and the agreement;
  - the recommendation table from the frozen constants;
  - the core-width ratios and the PS1 ratio.

  All agree to floating-point precision. Every uncertainty in the file carries a definition.
- **Agreement with the figure code (438 checks).** The luminosity, fit, κ (both codes) and pinched-core values equal those drawn by `plot_comparison.py`, `plot_deposition.py`, `paper_figures.py` and `plot_core_width.py`. The figure code reads the same committed core-width caches.
- **Fails loudly.** Each of the following stops the script with an error naming what is missing or wrong:
  - deleting a core-width cache, default or variation;
  - deleting `R1_table.npz`;
  - removing runs from `results.csv`;
  - removing an emittance from the data;
  - deleting the GP++ requirements export;
  - substituting a GP++ requirements export without a `sigma_log` column.

## Limits

- **n_t and cut multipliers** are not stored per run. They follow from the deck, `inputs/input_calib_C3_250.txt`: n_t = n_z, and the domain spans ±16 σ_x, ±16 σ_y, ±8 σ_z in every configuration.
- **Variant emittances.** The solver and deposition variants are reported at all eight emittances run. The emittances the manuscript quotes were not singled out.
- **Rounding.** `paper_numbers.md` rounds to the precision of the values quoted in the manuscript: luminosities, ratios and H_D to two decimals; exponents and κ to three. The table layout and rounding were not compared line by line with the manuscript source. The JSON is authoritative.
- **`paper_numbers.py`** is a separate summary printer. Its `--json` option writes a different layout, so do not point it at `paper_numbers.json`.
