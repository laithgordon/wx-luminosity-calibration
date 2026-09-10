# wx-luminosity-calibration

Analysis code, datasets and paper figures for the WarpX beam–beam luminosity calibration study
(C³-250 collider configuration): how many macroparticles (`n_m`) and vertical grid cells (`n_y`)
WarpX needs for a converged luminosity as the vertical emittance is lowered from 20 nm to 0.5 nm,
the pinch physics behind the requirement, and the comparison with GUINEA-PIG++ at converged settings.

The GUINEA-PIG++ side of the same study, including the scripts that produce the `data/gp/` exports
used here, is at <https://github.com/laithgordon/gp-luminosity-calibration>.

Layout follows that repository: scripts at the top level, `data/` and `plots/`.

## Contents

| script | role |
|---|---|
| `wxcal.py`, `collect.py` | run bookkeeping and the extraction that populates `data/results.csv`: `collect.py` integrates each run's `DifferentialLuminosity` spectrum (per-crossing luminosity in m⁻²) and converts it to 10³⁴ cm⁻² s⁻¹ with `1e-4 × n_b × f_rep` (133 bunches × 120 Hz); `wxcal.py` holds the beam constants and the run-label conventions. `data/joblist.csv` is the run register they read and write |
| `nmreq.py` | the calibration machinery: disruption parameter `D_y(ε_y)`, ladders, plateau rule, ±5 % bracket estimator, Monte-Carlo errors, weighted power-law fits, the first-waist envelope compression `R_1(D_y)`, κ, and the readers of the GP++ exports |
| `richardson.py` | Richardson-extrapolation overlays |
| `paper_figures.py` | figure style and the four calibration figures plus `kappa_vs_Dy` |
| `plot_comparison.py`, `plot_extension.py`, `plot_deposition.py` | tuned-vs-untuned and WarpX-vs-GP++ luminosity comparisons, the 40–100 nm extension, and the deposition-order test |
| `plot_coarse.py`, `plot_nx_nm_coupling.py` | coarse-grid control and the `n_x`–`n_m` coupling test |
| `plot_core_width.py`, `plot_slice_pinch.py`, `plot_pinch_evolution.py` | the pinched-core measurements from per-step particle dumps; `plot_core_width.py` also builds and caches `data/slice_widths_*.csv`, `data/slice_fiterr_*.csv` and `data/R1_table.npz` |
| `deposition_error.py`, `depo_corr_stat.py`, `plot_depo_correlation.py` | per-pass deposition-error measurement at the waists and its seed statistics (`data/depo_error.json`, `data/depo_corr_stat.json`) |
| `submit.py`, `add_jobs.py` | job submission: `add_jobs.py` appends runs to `data/joblist.csv`; `submit.py` renders `inputs/template.sbatch` per pending row (grid, `n_m`, ε_y, seed on the WarpX command line, `--extra` for variant flags) and submits it |
| `make_figures.py` | regenerates every figure in `plots/` |

## Figures (`plots/`)

| figure | content |
|---|---|
| `WX_L_vs_nm`, `WX_nm_req_calibration` | luminosity vs macroparticle count per ε_y, and the requirement `n_m^req/(n_x n_y n_z)` vs `D_y` with its power-law fit and conservative locus |
| `WX_L_vs_ny`, `WX_ny_req_calibration` | luminosity vs vertical cell count, and `n_y^req` vs `D_y` |
| `kappa_vs_Dy` | cells per pinched vertical σ for both codes, each at its own box cut |
| `nx_nm_coupling` | luminosity vs `n_m` at eight `n_x` values (the test that fixed the nominal grid) |
| `WX_coarse_grid_L_vs_nm` | a deliberately under-resolved grid: seed scatter collapses while L converges to the wrong value |
| `WX_comparison_L_vs_ey`, `WX_comparison_HD_vs_ey`, `WX_comparison_ratio_vs_ey` | L and `H_D = L/L_geom` for untuned and tuned WarpX and GP++, and the tuned/untuned ratios |
| `WX_extension_L_vs_ey`, `WX_code_ratio_vs_ey` | the tuned sets extended to 40–100 nm, and WarpX/GP++ from 1 to 100 nm |
| `WX_deposition_L_vs_ey` | 1st-order (CIC) vs 3rd-order deposition, 2D-slice vs 3D solver, against GP++ |
| `WX_core_width_vs_Dy` | minimum Gaussian-core width of the central slice vs `D_y` with the envelope prediction `1/R_1(D_y)` |
| `WX_pinch_evolution`, `WX_pinch_histogram` | vertical width through the crossing (whole beam, central and off-centre slices) and the slice y-distribution at the pinch |
| `WX_depo_correlation`, `WX_depo_error_passes` | deposition-error correlation between successive waists, and the per-pass residuals |

## Inputs and job submission (`inputs/`)

| file | content |
|---|---|
| `input_calib_C3_250.txt` | the default WarpX deck used for every calibration run: C³-250 beams, 3D integrated-Green-function Poisson solver, 3rd-order (cubic B-spline) deposition, Vay pusher, quantum-synchrotron beamstrahlung on, per-crossing luminosity from the `DifferentialLuminosity` diagnostic. Grid (`nx, ny, nz`), `nmacropart`, `emity`/`sigmay` and the seed are overridden per run on the command line, so one deck serves the whole scan |
| `input_calib_C3_250_2dslice.txt` | variant: `warpx.use_2d_slices_fft_solver = 1` (2D-slice solver, the GP++ field model; phase PS) |
| `input_calib_C3_250_cic.txt` | variant: `algo.particle_shape = 1` (first-order CIC deposition, 3D solver; phase PC3) |
| `input_calib_C3_250_2dslice_cic.txt` | variant: both (phase PC2). WarpX applies one shape order to all axes, so this is (1,1,1) against GP++'s (1,1,0) |
| `template.sbatch` | the Slurm template `submit.py` renders: whole GPU nodes on Perlmutter (4 ranks per node, `--gpu-bind=none`), the WarpX executable and module environment, the command-line overrides, and a watchdog that cancels a run whose solver has aborted. Site-specific paths (executable, account, CUDA libraries) are the ones used for the study and will need changing elsewhere |

In the study the three variants were run by passing the changed lines through `submit.py --extra "..."` rather than
as separate decks; they are written out in full here so each configuration is self-contained.

## Datasets (`data/`)

| file | content |
|---|---|
| `results.csv` | one row per WarpX run: ε_y, σ_y, grid (`nx, ny, nz`), `nm`, seed, luminosity `L` in 10³⁴ cm⁻² s⁻¹, and `solver` (`3d` calibration; `2d` 2D-slice solver; `3d_cic`/`2d_cic` first-order deposition) |
| `joblist.csv` | the run register (label, phase, grid, `n_m`, seed, node/walltime, status); read by `plot_core_width.py` to enumerate runs |
| `slice_widths_hw010_*.csv`, `slice_fiterr_hw010_*.csv` | per-step central-slice widths (rms, IQR, Gaussian core) and the core-fit errors, cached from the particle dumps (phases PP and PQ) |
| `R1_table.npz` | first-waist envelope compression `R_1(D_y)` on a `D_y` grid |
| `depo_error.json`, `depo_corr_stat.json` | per-pass deposition residuals and their correlations |
| `gp/*.csv` | the GUINEA-PIG++ exports: `n_y^req`/κ tables and fitted constants (`gp_*_export.csv`) and the luminosity sets (`lumi_nominal_vs_calibrated.csv`, `lumi_tuned_vs_frozen_highemit.csv`, `lumi_ee` in 10³⁴ cm⁻² s⁻¹, raw per-crossing value in `lumi_ee_m2`) |

## Regenerating

```
python make_figures.py            # every figure that needs only data/
python make_figures.py --dumps    # also WX_pinch_evolution, WX_pinch_histogram, WX_depo_error_passes
```

The three dump-based figures, and rebuilding the `slice_*` caches, `depo_*.json` and `results.csv`, need the raw WarpX
outputs (per-step openPMD particle dumps and `DifferentialLuminosity` spectra, stored externally); `wxcal.RUN_ROOT`
points at them. Everything else regenerates from `data/` alone.

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`; `h5py` for the dump-based scripts. WarpX 26.06 produced the runs.
