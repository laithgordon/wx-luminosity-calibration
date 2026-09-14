#!/usr/bin/env python3
"""Shared Stage-A machinery: nominal grid per e_y, D_y(e_y), the n_m ladders from
results.csv, and the Richardson-fitted n_m^req per e_y (methods §2-§5).

Nominal grid: n_x = 512, n_z = 128, n_y = n_y^theory(e_y) = pow2[256 (8/e_y)^1/4]
(log-2 half-up rounding, same as wxcal.ny_theory): 256 for e_y >= 4 nm,
512 for e_y <= 2 nm.
"""
import csv, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import richardson as R

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
RESULTS = ROOT / "data" / "results.csv"
EYS = [0.5, 1, 2, 4, 8, 12, 16, 20]
NX, NZ = 512, 128

# ---------------------------------------------------------------------------
# Pinch / vertical-compression theory (Theory_Prediction.txt, sec:ny_derivation)
#   R(D_y) = sigma_y^* / sigma_y^min = LAMBDA * D_y^Q_PRED      (eq:R_predicted)
#   n_y^req = 2 c_y kappa R(D_y) = C_y D_y^q, C_y = 2 c_y kappa LAMBDA (eq:ny_law)
#   n_m^req = C_m D_y^(s-q) n_x n_y n_z                          (eq:law_nm)
# Q_PRED is the canonical predicted exponent; LAMBDA is the normalisation of the
# power-law fit to R(D_y) over the ten scan points (see R_of_D / _R_TABLE).
# ---------------------------------------------------------------------------
Q_PRED = 0.379                  # predicted pinch exponent q (supersedes the old 1/2)
LAMBDA = 1.217                  # R(D_y) = LAMBDA D_y^q normalisation
# --- first-waist pinch model (supersedes the global-minimum R for all predictions) ---
# R_1(D_y) is the FIRST local minimum of the envelope beta(s), not the global one: beyond
# D* ~ 18 the global minimum is carried by later betatron oscillations that the real
# (phase-mixing, filamenting) beam never reaches. Fit over all ten D_Y_SCAN points:
LAMBDA1 = 1.856                 # ± SLAMBDA1, R_1(D_y) ≈ LAMBDA1 D_y^Q_P (rms resid 0.6 %)
SLAMBDA1 = 0.023
Q_P = 0.2153                    # ± SQ_P, pinch exponent of R_1 — the ONLY q that feeds s
SQ_P = 0.0032
Q_N = Q_P + 0.25                # = 0.4653, vertical-resolution exponent (per-pass error
                                # accumulation adds D^(1/4): n_y^req ∝ R_1(D_y) D_y^(1/4))


def s_from_bpercell(b):
    """s = b_percell + Q_P. The ONLY sanctioned way to form s: Q_P (pinch exponent), never
    Q_N (resolution exponent) — two different numbers that both read as "q"."""
    assert abs(Q_P - (Q_N - 0.25)) < 1e-12 and Q_P < 0.3, "s must be built from Q_P, not Q_N"
    return b + Q_P


_R1_CACHE = None


def R1_of_D(D):
    """First-oscillation compression R_1(D_y): earliest local minimum of beta(s) through the
    Gaussian pinch field. Served from Data/R1_table.npz (built by plot_core_width.R1_table)."""
    global _R1_CACHE
    if _R1_CACHE is None:
        import numpy as _np
        f = ROOT / "data" / "R1_table.npz"
        if not f.exists():
            from plot_core_width import R1_table as _rt
            _rt()
        d = _np.load(f)
        _R1_CACHE = (d["D"], d["R1"])
    return float(np.interp(D, _R1_CACHE[0], _R1_CACHE[1]))

BETA_STAR_OVER_SIGZ = 1.2       # beta_y^*/sigma_z for C3-250; sets alpha and R(D_y)
# --- grid cut multipliers, PER CODE (cut = c * sigma^*, cell width = 2 c sigma^*/n) ---
# WarpX, Execution/input_calib_C3_250.txt:
#     my_constants.Ly = 32*sigmay
#     geometry.prob_lo = -0.5*Lx -0.5*Ly -0.5*Lz
#     geometry.prob_hi =  0.5*Lx  0.5*Ly  0.5*Lz     -> domain spans -16..+16 sigma_y^*
#     my_constants.dy = Ly/ny                        -> dy = 32 sigma_y^*/ny = 2*16*sigma_y^*/ny
#   So Ly is the FULL domain height and the half-extent (the "cut") is 16 sigma_y^*.
C_X_WX, C_Y_WX = 16, 16
# GUINEA-PIG++, src/gridCPP.cc:129-137:
#     min_y_ = -((n_cell_y-2)/n_cell_y)*cut_y ; max_y_ = +(...)*cut_y
#     deltay = 2.0*cut_y/n_cell_y
#   So GP's cut_y is the HALF-extent, and the calibration deck runs cut_x/cut_y/cut_z = 20/20/3.5.
C_X_GP, C_Y_GP = 20, 20
# The two codes share the cell-width FORMULA (dy = 2 c_y sigma_y^*/n_y) but not the value of
# c_y: 16 for WarpX, 20 for GP++. kappa scales as 1/c_y and the occupancy prefactor as c_x c_y,
# so neither is comparable across codes until each is divided by its own deck's cuts.

# The ten D_y values of the emittance scan (Theory_Prediction.txt tab:disruption_scan)
D_Y_SCAN = (137.8, 97.4, 79.4, 68.8, 61.5, 48.5, 34.2, 27.9, 24.1, 21.5)
L_FRAC_NM, POOL_FALLBACK, TOL = 0.75, 0.02, 0.05
NM_FIT_MIN = 1e5        # Richardson overlay is fitted on rungs >= this (1e3/1e4 hook the p=1 series)
PLATEAU_TOL = 0.005     # L_inf rule: last two decade rungs (1e7, 1e8) agree within 0.5%

# beam parameters of the calib deck (input_calib_C3_250.txt)
N_PART, SIGMA_Z, SIGMA_X = 0.624e10, 100e-6, 210.12e-9
GAMMA, BETA_Y, R_E = 125e9 / 0.51099895e6, 0.12e-3, 2.8179403262e-15


def ny_theory(e_y):
    x = 256.0 * (8.0 / float(e_y)) ** 0.25
    return 2 ** int(math.floor(math.log2(x) + 0.5))


NY_OVERRIDE = {2.0: 256, 1.0: 256}   # 2026-08-16: 2 nm and 1 nm redone on the 256 grid (n_y = 512 study kept in results.csv)


def grid(e_y):
    return (NX, NY_OVERRIDE.get(float(e_y), ny_theory(e_y)), NZ)


def sigma_y(e_y_nm):
    return math.sqrt(BETA_Y * e_y_nm * 1e-9 / GAMMA)


ALPHA_FS = 1 / 137.035999


def upsilon(e_y_nm):
    """average beamstrahlung parameter Upsilon = (5/6) r_e^2 gamma N / (alpha sigma_z (sigma_x + sigma_y)); Upsilon_max ~ 2.4 x this"""
    sy = sigma_y(e_y_nm)
    return (5.0 / 6.0) * R_E ** 2 * GAMMA * N_PART / (ALPHA_FS * SIGMA_Z * (SIGMA_X + sy))


def D_y(e_y_nm):
    """vertical disruption, full formula D_y = 2 N r_e sigma_z / (gamma sigma_y (sigma_x + sigma_y))"""
    sy = sigma_y(e_y_nm)
    return 2 * N_PART * R_E * SIGMA_Z / (GAMMA * sy * (SIGMA_X + sy))


# conservative n_m^req(e_y) adopted 2026-08-16 for the Stage-B (n_y) runs: fit slope D_y^2.97 on the
# 20-4 nm points (512x256x128), intercept raised above every bracket. Frozen here so the n_y ladders
# stay reproducible even as the fit drifts with more seeds.
NM_CONS = {20: 1.0e4, 16: 1.4e4, 12: 2.2e4, 8: 4.0e4, 4: 1.1e5, 2: 3.1e5, 1: 8.8e5, 0.5: 2.5e6}

# conservative n_y^req(e_y) adopted 2026-08-19 for the calibrated-vs-uncalibrated comparison: the Stage-B fit
# slope (q = 0.447, 0.5 nm excluded), intercept raised above the upper bracket edge of every fitted point
# (C_y^cons = 38.6), the line rounded UP to the next power of 2 for the run grid. Frozen for reproducibility.
NY_CONS = {20: 256, 16: 256, 12: 256, 8: 256, 4: 256, 2: 256, 1: 512, 0.5: 512}

# extension of the tuned set to e_y = 40-100 nm (2026-08-21): the same conservative formulas evaluated at the new D_y
# (n_m^cons = 0.4349 D_y^3.272 rounded to 2 figures; n_y^cons = 38.3 D_y^0.449 rounded up to the next run rung, 512x n_y x128)
EYS_EXT = [40, 60, 80, 100]
NM_CONS.update({40: 3.2e3, 60: 1.6e3, 80: 9.9e2, 100: 6.8e2})
NY_CONS.update({40: 256, 60: 128, 80: 128, 100: 128})
# NB: NM_CONS / NY_CONS above are the older frozen tables that define the Stage-B ladder and comparison runs. They are
# NOT the production conservative locus below.

# ---------------------------------------------------------------------------
# Production conservative locus -- FROZEN. These constants defined the production runs (phase PN, submitted
# 2026-09-11) and the pinched-core dump re-runs (PQ2/PP2) at the time they were submitted:
#     n_m^cons = 0.440 D_y^3.269            (unrounded; the decks use ceil)
#     n_y^cons = smallest sampled power of two >= 38.1 D_y^0.450260
# with D_y = D_y(e_y) above at full precision. They must NOT be recomputed from the current ladder fit: that fit
# drifts as rungs are added (the raw locus slope moved from 3.26883 to 3.26750 once the production runs joined the
# ladders), and any check that re-derived the locus would eventually disagree with the decks that were run.
# ---------------------------------------------------------------------------
LOCUS_NM_PREFACTOR, LOCUS_NM_EXPONENT = 0.440, 3.269
LOCUS_NY_PREFACTOR, LOCUS_NY_EXPONENT = 38.1, 0.450260
LOCUS_NY_RUNGS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def n_m_cons(e_y_nm):
    """Production conservative macroparticle count at e_y (unrounded), from the frozen locus constants."""
    return LOCUS_NM_PREFACTOR * D_y(float(e_y_nm)) ** LOCUS_NM_EXPONENT


def n_y_cons(e_y_nm):
    """Production conservative vertical cell count at e_y: smallest sampled power of two >= the frozen law."""
    raw = LOCUS_NY_PREFACTOR * D_y(float(e_y_nm)) ** LOCUS_NY_EXPONENT
    return min(p for p in LOCUS_NY_RUNGS if p >= raw)


def ny_ladders(nm_tol=0.02):
    """{e_y: dict(ny, L, N, std)} for grid (512, n_y, 128) at n_m = NM_CONS[e_y] (deduped by seed)."""
    raw = defaultdict(list)
    for r in read_results("3d"):
        e = float(r["e_y_nm"])
        if e not in NM_CONS or int(r["nx"]) != NX or int(r["nz"]) != NZ:
            continue
        if abs(float(r["nm"]) / NM_CONS[e] - 1) > nm_tol:
            continue
        raw[(e, int(r["ny"]))].append((int(r["seed"]), float(r["L"])))
    runs = {k: dedupe_seeds(v) for k, v in raw.items()}
    out = {}
    for e in EYS:
        pts = sorted(v for (ee, v) in runs if ee == e)
        if not pts:
            continue
        L = np.array([np.mean(runs[(e, v)]) for v in pts]); N = np.array([len(runs[(e, v)]) for v in pts])
        std = np.array([np.std(runs[(e, v)], ddof=1) if len(runs[(e, v)]) > 1 else np.nan for v in pts])
        out[e] = dict(ny=np.array(pts, float), L=L, N=N, std=std)
    return out


def ny_requirements():
    """[(e_y, D_y, ny_req, lo, hi)] from the Stage-B ladders: L_inf = top-two-rung mean (0.5% rule,
    else provisional top rung), n_y^req = log-log interpolation between the two rungs straddling 5%."""
    out = []
    for e, d in ny_ladders().items():
        ny, L = d["ny"], d["L"]
        if len(ny) < 2:
            continue
        top_ok = abs(L[-1] / L[-2] - 1) <= PLATEAU_TOL
        Linf = float(0.5 * (L[-1] + L[-2])) if top_ok else float(L[-1])
        b = bracket(dict(nm=ny, L=L), Linf)
        if b and b["nreq"]:
            out.append((e, D_y(e), b["nreq"], b["lo"], b["hi"]))
    return out


def read_results(solver="3d"):
    """rows of results.csv for one field solver ('3d' = 3D IGF, the calibration; '2d' = 2D-slice IGF, phase PS).
    Rows written before the 'solver' column existed are 3D."""
    for r in csv.DictReader(open(RESULTS)):
        if r.get("solver", "3d") == solver:
            yield r


def dedupe_seeds(rows):
    """rows: [(seed, L)] -> one L per distinct seed (mean over duplicate same-seed rows, which are
    re-runs of the identical configuration and not independent samples)."""
    by = defaultdict(list)
    for sd, L in rows:
        by[sd].append(L)
    return [float(np.mean(v)) for v in by.values()]


def ladders():
    """{e_y: dict(nm, L, N, std, sem, pooled)} on the nominal grid of each e_y (deduped by seed)."""
    raw = defaultdict(list)
    for r in read_results("3d"):
        e = float(r["e_y_nm"])
        if (int(r["nx"]), int(r["ny"]), int(r["nz"])) != grid(e):
            continue
        raw[(e, int(float(r["nm"])))].append((int(r["seed"]), float(r["L"])))
    runs = {k: dedupe_seeds(v) for k, v in raw.items()}
    out = {}
    for e in EYS:
        pts = sorted(v for (ee, v) in runs if ee == e)
        if not pts:
            continue
        nm = np.array(pts, float)
        L = np.array([np.mean(runs[(e, v)]) for v in pts])
        N = np.array([len(runs[(e, v)]) for v in pts])
        std = np.array([np.std(runs[(e, v)], ddof=1) if len(runs[(e, v)]) > 1 else np.nan for v in pts])
        pooled = float(np.sqrt(np.nanmean(std ** 2))) if np.any(np.isfinite(std)) else POOL_FALLBACK * float(L.mean())
        sem = np.where(np.isfinite(std), std, pooled) / np.sqrt(N)
        out[e] = dict(nm=nm, L=L, N=N, std=std, sem=sem, pooled=pooled)
    return out


def nreq_from_fit(res):
    """largest n with |L(n)-L_inf| = TOL |L_inf| (§5) on a wide geometric grid, or None."""
    b, p, n0 = res["beta"], res["p"], res["n0"]
    x = np.geomspace(1e2, 1e13, 6000)
    dev = np.abs(R.model(b, x, p, n0) - b[0]) - TOL * abs(b[0])
    idx = np.where(np.sign(dev[:-1]) != np.sign(dev[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[-1]
    lx = np.log(x[i]) + (0 - dev[i]) * (np.log(x[i + 1]) - np.log(x[i])) / (dev[i + 1] - dev[i])
    return float(np.exp(lx))


def fit_ladder(d):
    """§2.4 L-fraction cut, p derived (=1 for an n_m sweep), windowed AIC fit. Returns (res, keep, nreq)."""
    keep = (d["L"] >= L_FRAC_NM * d["L"].max()) & (d["nm"] >= NM_FIT_MIN)
    p = R.derive_p(R.scan_alpha(d["nm"], d["nm"]))          # alpha = +1 -> p = 1
    res = R.fit(d["nm"][keep], d["L"][keep], d["sem"][keep], p, window_search=True, n0=1e6) \
        if keep.sum() >= 3 else None
    return res, keep, (nreq_from_fit(res) if res is not None else None)


def plateau(d):
    """L_inf = mean of the two highest decade rungs if they agree within PLATEAU_TOL, else None."""
    if len(d["nm"]) < 2:
        return None
    L1, L2 = d["L"][-2], d["L"][-1]
    if d["nm"][-1] / d["nm"][-2] < 9 or abs(L2 / L1 - 1) > PLATEAU_TOL:
        return None
    return float(0.5 * (L1 + L2))


def bracket(d, Linf):
    """Deficits dev_i = L_i/L_inf - 1 and the two adjacent rungs straddling the +/-TOL line:
    lo = largest rung with |dev| > TOL, hi = the next rung above it (must have |dev| <= TOL).
    n_req = log-log interpolation of |dev| between them. Returns dict or None."""
    nm, L = d["nm"], d["L"]
    dev = L / Linf - 1
    out_idx = [i for i in range(len(nm)) if _outside(np.array([dev[i]]))[0]]
    if not out_idx:
        return dict(dev=dev, lo=None, hi=nm[0], nreq=None, note="all rungs inside +/-5%: extend ladder DOWN")
    i = out_idx[-1]
    if i + 1 >= len(nm):
        return dict(dev=dev, lo=nm[i], hi=None, nreq=None, note="top rung outside +/-5%: extend ladder UP")
    j = i + 1
    a, b = abs(dev[i]), abs(dev[j])
    q = math.log(a / b) / math.log(nm[j] / nm[i])            # local slope of |dev| vs n_m
    nreq = float(nm[i] * (a / TOL) ** (1.0 / q)) if q > 0 else float(nm[j])
    return dict(dev=dev, lo=nm[i], hi=nm[j], nreq=nreq, q=q, width=nm[j] / nm[i],
                note=("bracket %.2gx" % (nm[j] / nm[i])))


MC_DRAWS, MC_SEED = 4000, 12345


def _sem_of(std, N, pooled):
    return np.where(np.isfinite(std), std, pooled) / np.sqrt(N)


def plateau_stat(d):
    """L_inf and its SEM from the two top rungs (pooled seed STD / sqrt(N1+N2)); None if the 0.5% rule fails."""
    Linf = plateau(d)
    if Linf is None:
        return None, None
    N1, N2 = d["N"][-2], d["N"][-1]
    s1, s2 = d["std"][-2], d["std"][-1]
    pooled = d.get("pooled", np.nan)
    ss = [x for x in (s1, s2) if np.isfinite(x)]
    sd = float(np.sqrt(np.mean(np.square(ss)))) if ss else (pooled if np.isfinite(pooled) else POOL_FALLBACK * Linf)
    return Linf, sd / math.sqrt(N1 + N2)


FROM_BELOW = True   # ladders converge from below: only dev < -TOL counts as 'outside' (a +TOL excursion is noise)


def _outside(dev):
    return (dev < -TOL) if FROM_BELOW else (np.abs(dev) > TOL)


def _interp_bracket(n, dev):
    """largest rung outside the band (from below) and the next rung inside -> log-log interpolated crossing (or nan)."""
    out_idx = np.where(_outside(dev))[0]
    if len(out_idx) == 0 or out_idx[-1] + 1 >= len(n):
        return np.nan, -1
    i = out_idx[-1]; j = i + 1
    a, c = abs(dev[i]), abs(dev[j])
    if c <= 0:
        return np.nan, i
    q = math.log(a / c) / math.log(n[j] / n[i])
    if q <= 0:
        return np.nan, i
    return n[i] * (a / TOL) ** (1.0 / q), i


def nreq_stat(d, Linf, sig_Linf, rng=None):
    """Statistical error on the interpolated n^req (methods doc §5): full-ladder Monte Carlo -- every rung
    mean and L_inf are redrawn from normals with their SEMs, the bracket is re-found on the resampled
    ladder and the crossing re-interpolated. sig_log = std of ln n^req over draws that have a crossing;
    frac_valid = fraction of draws with a crossing; also the fraction of draws in which the bracket
    moved off the central one (flip_frac). Central value = interpolation on the measured means."""
    b = bracket(d, Linf)
    if not b or not b["nreq"]:
        return None
    n = np.asarray(d["nm"], float); L = np.asarray(d["L"], float)
    pooled = d.get("pooled", POOL_FALLBACK * Linf)
    sem = _sem_of(d["std"], d["N"], pooled)
    rng = rng or np.random.default_rng(MC_SEED)
    i0 = int(np.where(n == b["lo"])[0][0])
    vals, flips = [], 0
    for _ in range(MC_DRAWS):
        Ld = rng.normal(L, sem); Lf = rng.normal(Linf, sig_Linf if sig_Linf else 0.0)
        v, i = _interp_bracket(n, Ld / Lf - 1)
        if np.isfinite(v):
            vals.append(math.log(v)); flips += (i != i0)
    vals = np.array(vals)
    if len(vals) < 50:
        return dict(nreq=b["nreq"], sig_log=np.nan, lo=b["lo"], hi=b["hi"], q=b["q"], n_valid=len(vals),
                    frac_valid=len(vals) / MC_DRAWS, flip_frac=np.nan, p16=np.nan, p84=np.nan)
    p16, p84 = np.percentile(vals, 16), np.percentile(vals, 84)
    return dict(nreq=b["nreq"], sig_log=float(0.5 * (p84 - p16)), sig_log_std=float(np.std(vals)),
                lo=b["lo"], hi=b["hi"], q=b["q"], n_valid=len(vals), frac_valid=len(vals) / MC_DRAWS,
                flip_frac=flips / max(1, len(vals)), p16=float(np.exp(p16)), p84=float(np.exp(p84)))


def requirements_stat():
    """Stage A: [(e_y, D_y, dict)] per e_y with a plateau and a bracketed crossing (statistical errors)."""
    out = []
    for e, d in ladders().items():
        Linf, sig = plateau_stat(d)
        if Linf is None:
            continue
        r = nreq_stat(d, Linf, sig)
        if r:
            r.update(Linf=Linf, sig_Linf=sig, cells=float(np.prod(grid(e))), N_lo=int(d["N"][list(d["nm"]).index(r["lo"])]),
                     N_hi=int(d["N"][list(d["nm"]).index(r["hi"])]),
                     dev_lo=float(d["L"][list(d["nm"]).index(r["lo"])] / Linf - 1),
                     dev_hi=float(d["L"][list(d["nm"]).index(r["hi"])] / Linf - 1))
            out.append((e, D_y(e), r))
    return out


def ny_requirements_stat():
    """Stage B: same as requirements_stat for the n_y ladders (L_inf = top-two-rung mean under the 0.5% rule)."""
    out = []
    for e, d in ny_ladders().items():
        ny, L, N, std = d["ny"], d["L"], d["N"], d["std"]
        if len(ny) < 2:
            continue
        pooled = float(np.sqrt(np.nanmean(std ** 2))) if np.any(np.isfinite(std)) else POOL_FALLBACK * float(L.mean())
        dd = dict(nm=ny, L=L, N=N, std=std, pooled=pooled)
        Linf, sig = plateau_stat(dd)
        if Linf is None:                       # provisional top rung
            Linf, sig = float(L[-1]), float((std[-1] if np.isfinite(std[-1]) else pooled) / math.sqrt(N[-1]))
        r = nreq_stat(dd, Linf, sig)
        if r:
            r.update(Linf=Linf, sig_Linf=sig, N_lo=int(N[list(ny).index(r["lo"])]), N_hi=int(N[list(ny).index(r["hi"])]),
                     dev_lo=float(L[list(ny).index(r["lo"])] / Linf - 1), dev_hi=float(L[list(ny).index(r["hi"])] / Linf - 1))
            out.append((e, D_y(e), r))
    return out


def powerlaw_wls(x, y, sig_log_y):
    """Weighted LS of ln y = a + b ln x with weights 1/sig^2 (methods §7). Returns dict(a,b,sa,sb,cov_ab,chi2,ndf)."""
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float)); w = 1.0 / np.asarray(sig_log_y, float) ** 2
    Sw, Sx, Sy, Sxx, Sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
    Dl = Sw * Sxx - Sx ** 2
    b = (Sw * Sxy - Sx * Sy) / Dl; a = (Sxx * Sy - Sx * Sxy) / Dl
    chi2 = float((w * (y - a - b * x) ** 2).sum())
    return dict(a=float(a), b=float(b), sa=float(np.sqrt(Sxx / Dl)), sb=float(np.sqrt(Sw / Dl)),
                cov_ab=float(-Sx / Dl), chi2=chi2, ndf=int(len(x) - 2))


# ---------------------------------------------------------------------------
# S2  Vertical compression factor R(D_y) = sigma_y^*/sigma_y^min
# ---------------------------------------------------------------------------
_R_CACHE = {}


def R_of_D(D):
    """R(D_y) = sigma_y^*/sigma_y^min from the Courant-Snyder envelope through the
    Gaussian pinch field (Theory_Prediction.txt eq:Ky_gauss, eq:beta_phi, eq:init_twiss).

    Units are normalised to sigma_z = 1, so beta_y^* = BETA_STAR_OVER_SIGZ. The two
    principal solutions of y'' = -K(s) y are integrated from s0 = -3 to +3 and the
    beta function beta = bi C^2 - 2 ai C S + gi S^2 is minimised over the crossing.

    Regression values (3 dp), D_Y_SCAN order:
        7.623 6.765 6.232 5.859 5.668 5.259 4.638 4.261 3.981 3.755
    whose power-law fit returns q = 0.374, LAMBDA = 1.217.
    """
    from scipy.integrate import solve_ivp
    key = round(float(D), 6)
    if key in _R_CACHE:
        return _R_CACHE[key]
    sz, bs = 1.0, BETA_STAR_OVER_SIGZ
    s0, s1, npts = -3.0, 3.0, 20000
    amp = (D / (math.sqrt(3) * sz ** 2)) * (2 * math.sqrt(3) / math.sqrt(2 * math.pi))

    def rhs(s, u):                     # u = [C, C', S, S']
        k = amp * math.exp(-2.0 * s * s / sz ** 2)
        return [u[1], -k * u[0], u[3], -k * u[2]]

    sol = solve_ivp(rhs, (s0, s1), [1.0, 0.0, 0.0, 1.0], rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.005)
    ss = np.linspace(s0, s1, npts); u = sol.sol(ss)
    bi = bs + s0 ** 2 / bs; ai = -s0 / bs; gi = (1 + ai ** 2) / bi
    beta = bi * u[0] ** 2 - 2 * ai * u[0] * u[2] + gi * u[2] ** 2
    val = float(math.sqrt(bs / beta.min()))
    _R_CACHE[key] = val
    return val


def R_table(Ds=D_Y_SCAN):
    """R(D_y) over the scan points; warms the cache."""
    return [R_of_D(D) for D in Ds]


def R_powerlaw(Ds=D_Y_SCAN):
    """Unweighted power-law fit of R(D_y): returns (q, LAMBDA). Must give (0.374, 1.217)."""
    r = R_table(Ds)
    b, a = np.polyfit(np.log(np.asarray(Ds, float)), np.log(r), 1)
    return float(b), float(math.exp(a))


# ---------------------------------------------------------------------------
# S3  kappa = cells per pinched vertical sigma  (Theory_Prediction.txt eq:ny_law)
# ---------------------------------------------------------------------------
def kappa_from_ny(ny_req, D, c_y):
    """kappa = n_y^req / (2 c_y R_1(D_y) D_y^(1/4)). Inverts the first-waist ny law point by point.

    c_y is REQUIRED and must come from the code's own deck (C_Y_WX / C_Y_GP): there is no
    shared default, because WarpX runs c_y = 16 and GP++ runs c_y = 20.

    The compression law is the first-waist model R_1(D_y) times the per-pass error
    accumulation factor D^(1/4) (n_y^req ∝ R_1 D^(1/4), exponent Q_N = Q_P + 1/4), so a
    flat kappa tests Q_N in the normalisation channel (drift slope = q_fit − Q_N)."""
    return float(ny_req) / (2.0 * c_y * R1_of_D(D) * D ** 0.25)


# ---------------------------------------------------------------------------
# S4  kappa drift test: is kappa constant across the scan?
# ---------------------------------------------------------------------------
def kappa_drift(D, ny_req, sig_log, c_y):
    """Per-point kappa_i, its weighted mean, and the weighted slope of ln kappa vs ln D_y.

    A slope consistent with zero validates R(D_y) independently of the exponent fit:
    d ln kappa / d ln D_y = q_fit - q_model, so it is the same test as the exponent
    comparison but expressed in the normalisation channel, and it is insensitive to
    c_y (which only shifts ln kappa by a constant).

    Returns dict(kappa=[...], mean, sig_mean, slope, sig_slope, chi2, ndf, drift).
    'drift' is the fractional change of the fitted line across the D_y band.
    """
    D = np.asarray(D, float); ny = np.asarray(ny_req, float); sig = np.asarray(sig_log, float)
    k = np.array([kappa_from_ny(n, d, c_y) for n, d in zip(ny, D)])
    w = 1.0 / sig ** 2
    mean = float(np.exp(np.sum(w * np.log(k)) / w.sum()))
    sig_mean = float(mean / math.sqrt(w.sum()))
    f = powerlaw_wls(D, k, sig)                      # ln kappa = a + b ln D_y
    lo, hi = float(D.min()), float(D.max())
    drift = float((hi / lo) ** f["b"] - 1.0)
    return dict(kappa=[float(x) for x in k], mean=mean, sig_mean=sig_mean,
                slope=f["b"], sig_slope=f["sb"], chi2=f["chi2"], ndf=f["ndf"],
                drift=drift, c_y=c_y)


# ---------------------------------------------------------------------------
# GUINEA-PIG++ values are read ONLY from the two CSVs exported by the GP++ analysis (gp-luminosity-calibration) into
# data/gp_exports/. GP runs its own deck cut (c_y = 20, recorded in the luminosity export header); nothing here
# assumes the WarpX value.
#   gp_luminosity_for_wx.csv    per-emittance luminosity aggregates in blocks nominal / conservative / frozen_extension
#   gp_requirements_for_wx.csv  tuning-ladder n_y^req and n_m^req rows; the published fit constants in its header
# ---------------------------------------------------------------------------
GP_DIR = ROOT / "data" / "gp_exports"
GP_LUMI_CSV = GP_DIR / "gp_luminosity_for_wx.csv"
GP_REQ_CSV = GP_DIR / "gp_requirements_for_wx.csv"


def gp_published_constants():
    """GP++ fit constants as stored in the gp_requirements_for_wx.csv header: C_y_fit, q_n_fit, C_m_fit, s_fit, and
    q_pred (from its 's_cons = b_cons + q_pred' line). Raises if any is missing; there is no hardcoded fallback."""
    hdr = "".join(l for l in open(GP_REQ_CSV) if l.startswith("#"))
    out = {}
    for name in ("C_y_fit", "q_n_fit", "C_m_fit", "s_fit"):
        m = re.search(rf"# constant {name}\s+published .*?\| stored in export ([-+0-9.eE]+)", hdr)
        if not m:
            raise ValueError(f"{name} not found in the {GP_REQ_CSV.name} header")
        out[name] = float(m.group(1))
    m = re.search(r"b_cons \+ q_pred = ([-+0-9.eE]+) \+ ([-+0-9.eE]+)", hdr)
    if not m:
        raise ValueError(f"q_pred not found in the {GP_REQ_CSV.name} header")
    out["q_pred"] = float(m.group(2))
    return out


def occupancy_prefactor(c_x, c_y):
    """2 c_x c_y/pi: converts n_m/(n_x n_y n_z) into the initial peak cell occupancy
    n_cell(0) (Theory_Prediction.txt eq:ncell). Code-specific through the deck cuts, so
    macroparticle requirements are comparable across codes only after this factor."""
    return 2.0 * c_x * c_y / math.pi


def occupancy_pinch(nm_per_cell, D, c_x, c_y):
    """n_pinch = (2 c_x c_y/pi) * [n_m/(n_x n_y n_z)] * R(D_y)   (eq:npinch)."""
    return occupancy_prefactor(c_x, c_y) * float(nm_per_cell) * R_of_D(D)


def R_schulte_eq24(e_y_nm, beta=None):
    """sigma_y^*/sigma_ybar_y from Schulte's empirical pinch parametrisation (his eqs. 2.3/2.4),
    kept only as the FAILING comparison of S5: it is a luminosity-equivalent size averaged over
    the crossing, not the transient minimum, and scales as D_y^0.13 instead of D_y^q."""
    beta = BETA_Y if beta is None else beta
    D = D_y(e_y_nm); sy = sigma_y(e_y_nm)
    H = 1 + D ** 0.25 * D ** 3 / (1 + D ** 3) * (math.log(1 + math.sqrt(D)) + math.log(SIGMA_Z / (0.8 * beta)))
    r = SIGMA_X / sy
    f = -(1 + 2 * r ** 3) / (6 * r ** 3)
    return float(H ** (-f))


def requirements():
    """[(e_y, D_y, n_req, cells, lo, hi)] for every e_y with a plateau and a bracketed crossing
    (n_req by log-log interpolation between the two straddling rungs)."""
    out = []
    for e, d in ladders().items():
        Linf = plateau(d)
        if Linf is None:
            continue
        b = bracket(d, Linf)
        if b and b["nreq"]:
            out.append((e, D_y(e), b["nreq"], float(np.prod(grid(e))), b["lo"], b["hi"]))
    return out
