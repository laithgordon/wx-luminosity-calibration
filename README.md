# wx-luminosity-calibration

Analysis code, datasets and figures for the WarpX beam–beam luminosity study of the C³-250 collider configuration:
how many macroparticles (`n_m`) and vertical grid cells (`n_y`) WarpX needs for a converged luminosity as the
vertical emittance is lowered from 20 nm to 0.5 nm, the pinch physics behind the requirement, and the comparison with
GUINEA-PIG++ at converged settings.

The GUINEA-PIG++ side of the study is at <https://github.com/laithgordon/gp-luminosity-calibration>. Its two exported
CSVs in `data/gp_exports/` are the only GP inputs used here.

## Reproducing the results

Requirements: Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`; `h5py` only for the dump-based figures.
WarpX 26.06 produced the runs.

From a fresh clone, with no configuration, in this order:

```
python make_figures.py      # every figure in plots/ that needs only data/
python paper_numbers.py     # every quoted WarpX number, printed (add --json FILE to save them)
```

Both read only `data/`. `paper_numbers.py` takes each number from the same code path that draws the corresponding
figure and never writes to `plots/`.

### Paper numbers

```
python reproduce_paper_numbers.py
```

regenerates every WarpX number the paper reports and writes `paper_numbers.json` and `paper_numbers.md`, which are
committed. The output corresponds to the numbers in the paper: a reviewer can compare `paper_numbers.md` with the
manuscript without reading the analysis code. It covers:
- the luminosity of each configuration and emittance, with the grid actually run and derived ratios and H_D;
- the n_y^req and n_m^req extractions and the fits to them;
- κ, the recommendation table from the frozen locus, and D_y;
- the pinched-core widths, the solver and deposition variants, and the PS1 reference.

The JSON holds full precision, the definition of every uncertainty, and the data file and selection rule behind every
row; the Markdown rounds as the paper does. The script reads only committed files under `data/` and needs only
`numpy`. Its one resampling step uses a fixed seed, so repeated runs give identical files. It stops with an error if an
expected input or data point is missing.

Three repository-only figures also need the raw per-step particle dumps, which are not in the repository:

```
WX_RUN_ROOT=/path/to/run/directories python make_figures.py --dumps
```

The upstream steps are also here. `python collect.py` rebuilds `data/results.csv` and `data/joblist.csv` from raw
output; `python depo_corr_stat.py` rebuilds `data/depo_corr_stat.json` from the dumps. Both need `WX_RUN_ROOT`.
`add_jobs.py` then `submit.py` submit new runs, with the configuration below.

## Figures (`plots/`)

**In the manuscript**

| figure | content |
|---|---|
| `WX_L_vs_nm` | luminosity vs macroparticle count per ε_y (the macroparticle tuning ladders) |
| `WX_L_vs_ny` | luminosity vs vertical cell count per ε_y (the vertical-resolution tuning ladders) |
| `WX_nm_req_convergence` | the requirement `n_m^req/(n_x n_y n_z)` vs `D_y` with its power-law fit and the frozen production locus |
| `WX_ny_req_convergence` | `n_y^req` vs `D_y` |
| `kappa_vs_Dy` | cells per pinched vertical σ for both codes, each at its own box cut |
| `WX_comparison_L_vs_ey` | L vs ε_y for untuned and tuned WarpX and GP++, with the geometric luminosity |
| `WX_code_ratio_vs_ey` | WarpX/GP++ tuned luminosity from 1 to 100 nm |
| `WX_core_width_vs_Dy` | minimum Gaussian-core width of the central slice vs `D_y`, and measured against the envelope prediction `1/R_1(D_y)` |

**Repository only**

| figure | content |
|---|---|
| `WX_comparison_HD_vs_ey` | `H_D = L/L_geom` for untuned and tuned WarpX and GP++ |
| `nx_nm_coupling` | luminosity vs `n_m` at eight `n_x` values |
| `WX_coarse_grid_L_vs_nm` | a deliberately under-resolved grid: seed scatter collapses while L converges to the wrong value |
| `WX_comparison_ratio_vs_ey` | tuned/untuned luminosity ratio per code |
| `WX_extension_L_vs_ey` | the tuned sets extended to 40–100 nm |
| `WX_deposition_L_vs_ey` | 1st-order (CIC) vs 3rd-order deposition, 2D-slice vs 3D solver, against GP++ |
| `WX_depo_correlation` | deposition-error correlation between successive waists |
| `WX_pinch_evolution`, `WX_pinch_histogram` | vertical width through the crossing and the slice y-distribution at the pinch (dumps) |
| `WX_depo_error_passes` | per-pass deposition residuals at the waists (dumps) |

## Configuration

Every machine-specific location is set in one place, `wxcal.py`, from an environment variable. None is needed to
regenerate figures or numbers from `data/`.

| variable | used by | default | value in the study (NERSC Perlmutter) |
|---|---|---|---|
| `WX_RUN_ROOT` | `collect.py`, `submit.py`, `depo_corr_stat.py`, dump-based figures | `runs/` | `/pscratch/sd/l/laithg/simulations/wx_calibration` |
| `WX_WARPX_EXE` | `submit.py` | `warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.QED` | `/pscratch/sd/l/laithg/WarpX/build/bin/warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.QED` |
| `WX_QED_TABLE_DIR` | `submit.py` | `qed_tables/` | `/global/cfs/cdirs/m4272/aforment/qed_tables` |
| `WX_SITE_ENV` | `submit.py` | `inputs/site_env_perlmutter.sh` | (default) |

`inputs/site_env_perlmutter.sh` is the module and library environment of the study's WarpX build on Perlmutter;
replace it on another machine. The QED lookup-table paths are passed to WarpX on the command line at submission, so
the decks carry only the table file names.

## Frozen production locus

The production runs were defined by the conservative locus

    n_m^cons = 0.440 D_y^3.269            n_y^cons = smallest sampled power of two >= 38.1 D_y^0.450260

These constants are fixed in `nmreq.py` (`LOCUS_*`, `n_m_cons`, `n_y_cons`), frozen at the time the runs were
submitted. Downstream code uses them and never re-derives the locus from the current ladder fit, which drifts as rungs
are added. `NM_CONS` / `NY_CONS` in the same file are the older frozen tables that define the tuning-ladder runs, not
this locus.

## Scripts

| script | role |
|---|---|
| `make_figures.py` | regenerates every figure in `plots/` |
| `paper_numbers.py` | computes every quoted WarpX number |
| `wxcal.py`, `collect.py` | run bookkeeping, the site configuration, and the extraction that populates `data/results.csv`: `collect.py` integrates each run's `DifferentialLuminosity` spectrum (per-crossing luminosity in m⁻²) and converts it to 10³⁴ cm⁻² s⁻¹ with `1e-4 × n_b × f_rep` (133 bunches × 120 Hz) |
| `nmreq.py` | the convergence machinery: disruption parameter `D_y(ε_y)`, ladders, plateau rule, ±5 % bracket estimator, Monte-Carlo errors, weighted power-law fits, the first-waist envelope compression `R_1(D_y)`, κ, the frozen production locus, and the readers of the GP++ exports |
| `richardson.py` | Richardson-extrapolation overlays |
| `paper_figures.py` | figure style, the ladder and convergence figures, and `kappa_vs_Dy` |
| `plot_comparison.py`, `plot_extension.py`, `plot_deposition.py` | tuned-vs-untuned and WarpX-vs-GP++ luminosity comparisons, the 40–100 nm extension, and the deposition-order test |
| `plot_coarse.py`, `plot_nx_nm_coupling.py` | coarse-grid control and the `n_x`–`n_m` coupling test |
| `plot_core_width.py`, `plot_slice_pinch.py`, `plot_pinch_evolution.py` | the pinched-core measurements from per-step particle dumps; `plot_core_width.py` also builds the `data/slice_*` caches |
| `deposition_error.py`, `depo_corr_stat.py`, `plot_depo_correlation.py` | per-pass deposition error at the waists and its seed statistics |
| `add_jobs.py`, `submit.py` | job submission: `add_jobs.py` appends runs to `data/joblist.csv`; `submit.py` renders `inputs/template.sbatch` per pending row and submits it |

## Inputs (`inputs/`)

| file | content |
|---|---|
| `input_calib_C3_250.txt` | the default WarpX deck: C³-250 beams, 3D integrated-Green-function Poisson solver, 3rd-order deposition, Vay pusher, quantum-synchrotron beamstrahlung, per-crossing luminosity from the `DifferentialLuminosity` diagnostic. Grid, `nmacropart`, `emity`/`sigmay`, seed and QED table paths are set on the command line |
| `input_calib_C3_250_2dslice.txt` | variant: 2D-slice solver |
| `input_calib_C3_250_cic.txt` | variant: first-order (CIC) deposition, 3D solver |
| `input_calib_C3_250_2dslice_cic.txt` | variant: both |
| `template.sbatch`, `site_env_perlmutter.sh` | the Slurm template `submit.py` renders (whole GPU nodes, 4 ranks per node, a watchdog that cancels a run whose solver has aborted) and the site environment it sources |

## Datasets (`data/`)

| file | read by |
|---|---|
| `results.csv` — one row per WarpX run: ε_y, σ_y, grid (`nx, ny, nz`), `nm`, seed, luminosity `L` in 10³⁴ cm⁻² s⁻¹, and `solver` (`3d`; `2d` 2D-slice; `3d_cic`/`2d_cic` first-order deposition) | every luminosity figure, `paper_numbers.py` |
| `joblist.csv` — the run register (label, phase, grid, `n_m`, seed, node/walltime, status) | `plot_core_width.py`, `collect.py`, `submit.py` |
| `slice_widths_hw010_*.csv`, `slice_fiterr_hw010_*.csv` — per-step central-slice widths and core-fit errors cached from the particle dumps | `plot_core_width.py`, `depo_corr_stat.py`, `deposition_error.py` |
| `R1_table.npz` — first-waist envelope compression `R_1(D_y)` | `nmreq.py`, `plot_core_width.py` |
| `depo_corr_stat.json` — per-seed deposition-error correlations | `plot_depo_correlation.py` |
| `gp_exports/gp_luminosity_for_wx.csv` — GUINEA-PIG++ luminosity per emittance in blocks `nominal`, `conservative`, `frozen_extension` | `plot_comparison.py`, `plot_extension.py`, `plot_deposition.py`, `paper_figures.py` |
| `gp_exports/gp_requirements_for_wx.csv` — GUINEA-PIG++ tuning-ladder requirements, with the published fit constants in its header | `paper_figures.py` |
