#!/usr/bin/env python3
"""Regenerate every WarpX number reported in the paper from the committed data, and nothing else.

    python reproduce_paper_numbers.py

writes two files at the repository root:
  paper_numbers.json  every quantity at full precision; every uncertainty carries a 'definition'; every block states the
                      data file and the selection rule that produced it
  paper_numbers.md    the same quantities as tables, rounded to the precision the paper quotes

Inputs, all committed: data/results.csv (one row per run), data/joblist.csv (the run register, for each run's phase),
data/R1_table.npz (first-waist compression R_1(D_y)), the data/slice_widths_hw010_* / data/slice_fiterr_hw010_*
caches (pinched-core widths), and the GUINEA-PIG++ n_y^req export data/gp_exports/gp_requirements_for_wx.csv with its deck
cut from data/gp_exports/gp_luminosity_for_wx.csv (for the GP++ kappa that the WarpX kappa is compared against, and the GP++
luminosities in the WarpX/GP++ luminosity ratio). Nothing outside the repository is read and nothing is fetched. The only resampling, the
Monte Carlo behind the n^req uncertainties, uses a fixed generator seed recorded in the output, so repeated runs write
identical files. A missing input, or an expected emittance with no qualifying runs, raises instead of writing a partial
table.
"""
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
for _f in (DATA / "results.csv", DATA / "joblist.csv", DATA / "R1_table.npz",
           DATA / "gp_exports" / "gp_requirements_for_wx.csv", DATA / "gp_exports" / "gp_luminosity_for_wx.csv"):
    if not _f.is_file():                     # checked before importing nmreq, which would otherwise try to rebuild R_1
        raise FileNotFoundError(f"required input missing: {_f.relative_to(ROOT)}")

import nmreq as Q
import wxcal as W

OUT_JSON, OUT_MD = ROOT / "paper_numbers.json", ROOT / "paper_numbers.md"
ALL_EYS = sorted(Q.EYS + Q.EYS_EXT)
F_COLL = W.NUM_BUNCHES * W.TRAIN_REP                 # 133 bunches x 120 Hz
PS1_PUBLISHED = 1.35                                 # 1e34 cm^-2 s^-1
C_Y = Q.C_Y_WX                                       # WarpX vertical cut multiplier, enters kappa
C_Y_GP = Q.C_Y_GP                                    # GP++ vertical cut multiplier, checked against the GP export header
CUTS = {"x": 16, "y": 16, "z": 8}
CUT_NOTE = ("domain half-extent in units of the rms beam size at the interaction point: Lx = 32 sigma_x, "
            "Ly = 32 sigma_y, Lz = 16 sigma_z (inputs/input_calib_C3_250.txt); identical in every configuration")
NT_NOTE = "n_t = n_z: the deck sets dt = T / n_z and stop_time = T"

# Points left out of a fit; identical to paper_figures.NM_ANOMALY / NY_ANOMALY, which draw them faded and unfitted.
NM_EXCLUDED = {0.5: "highest-D_y regime, n_m^req anomalous; shown, not fitted",
               1.0: "highest-D_y regime, n_m^req anomalous; shown, not fitted"}
NY_EXCLUDED = {0.5: "n_y^req anomalous; shown, not fitted"}

STD = "sample standard deviation over independent seeds, ddof = 1 (same-seed repeat runs averaged first)"
SE = "standard error of the mean: sample standard deviation (ddof = 1) / sqrt(n_seeds)"
RATIO_SE = "standard error: the relative standard errors of numerator and denominator added in quadrature"
SCATTER = "relative seed scatter: sample standard deviation (ddof = 1) / mean"
MC = (f"half the 16th-84th percentile width of ln(n^req) over {Q.MC_DRAWS} Monte Carlo redraws of every rung mean "
      f"and of L_inf from normal distributions with their standard errors (numpy default_rng, seed {Q.MC_SEED}); "
      f"a 1-sigma-equivalent uncertainty on ln(n^req), not a standard deviation over seeds")
FIT = ("1-sigma parameter uncertainty from the covariance of the weighted least-squares fit of ln y = a + b ln D_y "
       "with weights 1/sigma_ln^2 (not rescaled by chi2/ndf)")
FIT_C = "C = exp(a); uncertainty C * sigma_a, the fit uncertainty on a propagated to first order"

LOCUS_RULE = ("solver 3d; (n_x, n_z) = (512, 128); n_y = n_y^cons(e_y); |n_m / n_m^cons(e_y) - 1| < 0.02, "
              "with n_y^cons and n_m^cons the frozen production locus (see 'fits.conservative_loci')")
FROZEN_RULE = "solver 3d; (n_x, n_y, n_z) = (512, 256, 128); |n_m / 1e4 - 1| < 0.02: the 20 nm conservative settings held fixed"
TUNING_TABLE_RULE = ("(n_x, n_z) = (512, 128); n_y = NY_CONS[e_y] and |n_m / NM_CONS[e_y] - 1| < 0.02, the tuning-ladder "
                  "table in nmreq.py (n_m = 2.5e6, 8.8e5, 3.1e5, 1.1e5, 4e4, 2.2e4, 1.4e4, 1e4 at 0.5-20 nm)")


class InputError(RuntimeError):
    pass


def fail(msg):
    raise InputError(msg)


def U(value, definition):
    return {"value": float(value), "definition": definition}


# ---------------------------------------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------------------------------------
def load_runs():
    """results.csv rows, each joined to its completed joblist.csv row for the phase. The join must be one-to-one."""
    key = lambda e, nx, ny, nz, nm, seed, solver, L: (float(e), int(nx), int(ny), int(nz), float(nm), int(seed), solver,
                                                     round(float(L), 6))
    phases = defaultdict(list)
    for j in csv.DictReader(open(DATA / "joblist.csv")):
        if j["status"] == "done":
            phases[key(j["e_y_nm"], j["nx"], j["ny"], j["nz"], j["nm"], j["seed"], W.SOLVER_TAG.get(j["phase"], "3d"),
                       j["L_rate"])].append(j["phase"])
    runs = []
    for r in csv.DictReader(open(DATA / "results.csv")):
        k = key(r["e_y_nm"], r["nx"], r["ny"], r["nz"], r["nm"], r["seed"], r["solver"], r["L"])
        if not phases[k]:
            fail(f"data/results.csv row {k} has no matching completed run in data/joblist.csv")
        runs.append(dict(e=k[0], grid=k[1:4], nm=k[4], seed=k[5], solver=k[6], L=float(r["L"]), phase=phases[k].pop()))
    if any(phases.values()):
        fail("data/joblist.csv has completed runs missing from data/results.csv")
    return runs


def aggregate(runs, name, solver, eys, match, rule):
    """One row per emittance: seed statistics of the runs passing the selection. Raises if an emittance has fewer than
    two seeds or the selected runs do not share one grid and n_m."""
    by = defaultdict(list)
    for r in runs:
        if r["solver"] == solver and r["e"] in eys and match(r["e"], r["grid"], r["nm"]):
            by[r["e"]].append(r)
    rows = []
    for e in eys:
        rs = by.get(e, [])
        per_seed = defaultdict(list)
        for r in rs:
            per_seed[r["seed"]].append(r["L"])
        if len(per_seed) < 2:
            fail(f"{name}: {len(per_seed)} seed(s) at e_y = {e:g} nm, need at least 2 (selection: {rule})")
        settings = {(r["grid"], r["nm"]) for r in rs}
        if len(settings) != 1:
            fail(f"{name}: runs at e_y = {e:g} nm mix settings {sorted(settings)}")
        (nx, ny, nz), nm = settings.pop()
        L = [float(np.mean(v)) for v in per_seed.values()]
        n = len(L); mean = float(np.mean(L)); std = float(np.std(L, ddof=1))
        rows.append(dict(configuration=name, e_y_nm=e, L_mean_1e34=mean, L_std_1e34=U(std, STD),
                         L_se_1e34=U(std / math.sqrt(n), SE), n_seeds=n, seeds=sorted(per_seed),
                         n_x=nx, n_y=ny, n_z=nz, n_t=nz, n_m=int(nm), cut_multipliers=CUTS,
                         provenance=dict(file="data/results.csv", selection=rule, solver=solver,
                                         runs_per_phase=dict(sorted(Counter(r["phase"] for r in rs).items())),
                                         same_seed_repeat_runs_averaged=len(rs) - n)))
    return rows


def ratio(a, b):
    """a / b for two dataset rows, with the standard error."""
    r = a["L_mean_1e34"] / b["L_mean_1e34"]
    se = r * math.hypot(a["L_se_1e34"]["value"] / a["L_mean_1e34"], b["L_se_1e34"]["value"] / b["L_mean_1e34"])
    return dict(value=r, se=U(se, RATIO_SE))


def gp_luminosity(block):
    """{e_y: row} for one block of data/gp_exports/gp_luminosity_for_wx.csv, with the mean and standard error of L."""
    path = DATA / "gp_exports" / "gp_luminosity_for_wx.csv"
    return {float(r["eps_y_nm"]): dict(L_mean_1e34=float(r["L_mean_1e34"]), L_se_1e34=U(float(r["L_sem_1e34"]), SE),
                                       n_seeds=int(r["n_seeds"]))
            for r in csv.DictReader(l for l in open(path) if not l.startswith("#")) if r["block"] == block}


def L_geom(e):
    """Geometric luminosity f_coll N^2 / (4 pi sigma_x sigma_y) in 1e34 cm^-2 s^-1."""
    return F_COLL * Q.N_PART ** 2 / (4 * math.pi * Q.SIGMA_X * Q.sigma_y(e) * 1e4) / 1e34


# ---------------------------------------------------------------------------------------------------------------------
# requirements, fits, kappa
# ---------------------------------------------------------------------------------------------------------------------
def requirement_rows(stat, quantity):
    rows = []
    for e, D, r in stat:
        excluded = (NM_EXCLUDED if quantity == "n_m" else NY_EXCLUDED).get(e)
        row = dict(e_y_nm=e, D_y=D, value=r["nreq"], uncertainty_ln=U(r["sig_log"], MC),
                   mc_percentile_16=r["p16"], mc_percentile_84=r["p84"], mc_draws_with_crossing_fraction=r["frac_valid"],
                   L_inf_1e34=r["Linf"], L_inf_se_1e34=U(r["sig_Linf"], "standard error of L_inf: the pooled seed standard "
                                                         "deviation (ddof = 1) of the rungs defining L_inf / sqrt(total seeds)"),
                   bracket_rungs=[r["lo"], r["hi"]], bracket_seeds=[r["N_lo"], r["N_hi"]],
                   used_in_fit=excluded is None, excluded_because=excluded)
        if quantity == "n_m":
            row.update(ladder_cells_nx_ny_nz=r["cells"], value_per_cell=r["nreq"] / r["cells"],
                       ladder_grid=list(Q.grid(e)))
        rows.append(row)
    return rows


def E(estimate, uncertainty, definition):
    return {"estimate": float(estimate), "uncertainty": U(uncertainty, definition)}


def wls(D, y, sig, use, exponent_fixed=None):
    """Weighted power-law fit y = C D^b on the points 'use'; with exponent_fixed only C is fitted."""
    D, y, sig = (np.asarray(v, float)[use] for v in (D, y, sig))
    if exponent_fixed is None:
        f = Q.powerlaw_wls(D, y, sig)
        C = math.exp(f["a"])
        return dict(exponent=E(f["b"], f["sb"], FIT), prefactor=E(C, C * f["sa"], FIT_C), chi2=f["chi2"], ndf=f["ndf"],
                    n_points=len(D))
    w = 1.0 / sig ** 2
    a = float(np.sum(w * (np.log(y) - exponent_fixed * np.log(D))) / w.sum())
    chi2 = float(np.sum(w * (np.log(y) - a - exponent_fixed * np.log(D)) ** 2))
    C = math.exp(a)
    return dict(exponent_fixed=exponent_fixed,
                prefactor=E(C, C / math.sqrt(w.sum()), "C = exp(a), a the weighted mean of ln y - q ln D_y; uncertainty "
                                                       "C / sqrt(sum of weights), weights 1/sigma_ln^2"),
                chi2=chi2, ndf=len(D) - 1, n_points=len(D))


def fits_and_kappa(A, B):
    eA = [a[0] for a in A]; DA = np.array([a[1] for a in A]); nA = np.array([a[2]["nreq"] for a in A])
    cA = np.array([a[2]["cells"] for a in A]); sA = np.array([a[2]["sig_log"] for a in A]); hA = np.array([a[2]["hi"] for a in A])
    eB = [b[0] for b in B]; DB = np.array([b[1] for b in B]); nB = np.array([b[2]["nreq"] for b in B])
    sB = np.array([b[2]["sig_log"] for b in B]); hB = np.array([b[2]["hi"] for b in B])
    useA = np.array([e not in NM_EXCLUDED for e in eA]); useB = np.array([e not in NY_EXCLUDED for e in eB])
    rho = nA / cA

    def nm_law(use):
        f = wls(DA, rho, sA, use)
        f["s"] = E(Q.s_from_bpercell(f["exponent"]["estimate"]), f["exponent"]["uncertainty"]["value"],
                   "uncertainty of the fitted exponent; q_p is a fixed constant")
        return f

    nm_fit, nm_all = nm_law(useA), nm_law(np.ones_like(useA))
    ny_free, ny_all = wls(DB, nB, sB, useB), wls(DB, nB, sB, np.ones_like(useB))
    ny_con = wls(DB, nB, sB, useB, exponent_fixed=Q.Q_N)
    exA = {f"{e:g}": NM_EXCLUDED[e] for e in eA if e in NM_EXCLUDED}
    exB = {f"{e:g}": NY_EXCLUDED[e] for e in eB if e in NY_EXCLUDED}

    # the fits the frozen loci were cut from, recomputed on the committed data: raw-n_m (resp. free n_y) slope, intercept
    # raised to the largest upper bracket rung among the fitted points
    raw = Q.powerlaw_wls(DA[useA], nA[useA], sA[useA])
    C_raw = math.exp(float(np.max(np.log(hA[useA]) - raw["b"] * np.log(DA[useA]))))
    qy = ny_free["exponent"]["estimate"]
    C_ycons = math.exp(float(np.max(np.log(hB[useB]) - qy * np.log(DB[useB]))))

    fits = dict(
        n_m=dict(law="n_m^req / (n_x n_y n_z) = C_m D_y^(s - q_p)", q_p=Q.Q_P, fit=nm_fit,
                 fitted_emittances_nm=[e for e, u in zip(eA, useA) if u], excluded=exA, same_fit_without_exclusions=nm_all,
                 provenance="section 'requirements.n_m_req': value_per_cell and uncertainty_ln of the rows with used_in_fit = true"),
        n_y_free=dict(law="n_y^req = C_y D_y^q", fit=ny_free, fitted_emittances_nm=[e for e, u in zip(eB, useB) if u],
                      excluded=exB, same_fit_without_exclusions=ny_all,
                      provenance="section 'requirements.n_y_req': value and uncertainty_ln of the rows with used_in_fit = true"),
        n_y_exponent_constrained=dict(law=f"n_y^req = C_y D_y^q_n, q_n = q_p + 1/4 = {Q.Q_N:.4f} fixed", fit=ny_con,
                                      fitted_emittances_nm=[e for e, u in zip(eB, useB) if u], excluded=exB,
                                      provenance="section 'requirements.n_y_req': value and uncertainty_ln of the rows with used_in_fit = true"),
        conservative_loci=dict(
            statement=("FROZEN values: the constants that defined the production runs (phase PN, submitted 2026-09-11) "
                       "and the pinched-core dump runs, fixed in nmreq.py. They are not recomputed from the current fit; "
                       "as definitions they carry no uncertainty, chi2 or ndf. The fits they were cut from are recomputed "
                       "on the committed data for comparison; they differ slightly because rungs were added after the "
                       "constants were frozen."),
            n_m=dict(law="n_m^cons = C D_y^b, absolute macroparticles (the decks use the ceiling)",
                     C_frozen=Q.LOCUS_NM_PREFACTOR, b_frozen=Q.LOCUS_NM_EXPONENT,
                     recomputed_from_current_data=dict(
                         rule="weighted fit of ln n_m^req on ln D_y over the n_m fit points; C raised to the largest upper "
                              "bracket rung among them",
                         b=E(raw["b"], raw["sb"], FIT), C=C_raw, chi2=raw["chi2"], ndf=raw["ndf"])),
            n_y=dict(law="n_y^cons = smallest of {32, 64, ..., 8192} >= C D_y^q",
                     C_frozen=Q.LOCUS_NY_PREFACTOR, q_frozen=Q.LOCUS_NY_EXPONENT,
                     recomputed_from_current_data=dict(
                         rule="free n_y fit exponent; C raised to the largest upper bracket rung among the fit points",
                         q=E(qy, ny_free["exponent"]["uncertainty"]["value"], FIT), C=C_ycons,
                         chi2=ny_free["chi2"], ndf=ny_free["ndf"]))))

    kappa = dict(
        definition=("kappa = n_y^req / (2 c_y R_1(D_y) D_y^(1/4)), with c_y each code's own vertical cut multiplier "
                    f"(WarpX {C_Y}, GUINEA-PIG++ {C_Y_GP}) and R_1(D_y) from data/R1_table.npz (linear interpolation in "
                    "D_y), the same table for both codes. <kappa> is a weighted mean of ln kappa, exponentiated: "
                    "exp(sum w ln kappa / sum w) with w = 1/sigma_ln^2, sigma_ln the uncertainty on ln n_y^req (equal to "
                    "that on ln kappa). The slope is the weighted least-squares fit of ln kappa against ln D_y with the "
                    "same weights."),
        warpx=kappa_block(eB, DB, nB, sB, useB, C_Y, {e: NY_EXCLUDED.get(e) for e in eB},
                          "section 'requirements.n_y_req' (value, uncertainty_ln, D_y); same points as the n_y fit"),
        gp=gp_kappa())
    w, g = kappa["warpx"]["weighted_mean"], kappa["gp"]["weighted_mean"]
    r = w["estimate"] / g["estimate"]
    kappa["ratio_warpx_over_gp"] = E(r, r * math.hypot(w["uncertainty"]["value"] / w["estimate"],
                                                       g["uncertainty"]["value"] / g["estimate"]),
                                     "the relative uncertainties of the two weighted means added in quadrature")
    kappa["agreement_sigma"] = dict(value=abs(w["estimate"] - g["estimate"]) / math.hypot(w["uncertainty"]["value"],
                                                                                          g["uncertainty"]["value"]),
                                    definition="|<kappa>_WarpX - <kappa>_GP| / sqrt(sigma_WarpX^2 + sigma_GP^2), the two "
                                               "weighted-mean uncertainties")
    return fits, kappa


KAPPA_MEAN_UNC = ("standard error of the weighted mean of ln kappa, 1/sqrt(sum w) with w = 1/sigma_ln^2, carried to "
                  "<kappa> by multiplying by <kappa>")


def kappa_block(es, D, n, sig, use, c_y, excluded, provenance):
    """Per-point kappa, the weighted mean of ln kappa and the ln kappa vs ln D_y slope for one code."""
    D, n, sig, use = (np.asarray(v) for v in (D, n, sig, use))
    k = Q.kappa_drift(D[use].astype(float), n[use].astype(float), sig[use].astype(float), c_y=c_y)
    per = []
    for e, d, nn, s_, u in zip(es, D, n, sig, use):
        kv = Q.kappa_from_ny(float(nn), float(d), c_y)
        per.append(dict(e_y_nm=float(e), D_y=float(d), n_y_req=float(nn), sigma_ln=float(s_), R_1=Q.R1_of_D(float(d)),
                        kappa=kv, uncertainty=U(kv * float(s_), "kappa * sigma_ln: the uncertainty on ln n_y^req carried to kappa"),
                        used_in_mean_and_slope=bool(u), excluded_because=None if u else excluded.get(float(e))))
    return dict(c_y=c_y, per_emittance=per, weighted_mean=E(k["mean"], k["sig_mean"], KAPPA_MEAN_UNC),
                slope_ln_kappa_vs_ln_D_y=dict(**E(k["slope"], k["sig_slope"], FIT), chi2=k["chi2"], ndf=k["ndf"]),
                excluded={f"{e:g}": v for e, v in excluded.items() if v}, provenance=provenance)


def gp_kappa():
    """GUINEA-PIG++ kappa from its n_y^req export. Raises if the export lacks the sigma_log column or its deck cut
    does not match C_Y_GP."""
    hdr = "".join(l for l in open(Q.GP_LUMI_CSV) if l.startswith("#"))
    if f"cut multipliers {C_Y_GP}/{C_Y_GP}/3.5" not in hdr:
        fail(f"data/gp_exports/gp_luminosity_for_wx.csv header does not state the GP++ cut multipliers {C_Y_GP}/{C_Y_GP}/3.5")
    rd = csv.DictReader(l for l in open(Q.GP_REQ_CSV) if not l.startswith("#"))
    if "sigma_log" not in (rd.fieldnames or []):
        fail(f"data/gp_exports/gp_requirements_for_wx.csv has no sigma_log column (columns: {rd.fieldnames})")
    rows = [r for r in rd if r["quantity"] == "n_y_req"]
    if not rows:
        fail("data/gp_exports/gp_requirements_for_wx.csv has no n_y_req rows")
    col = lambda k: [float(r[k]) for r in rows]
    return kappa_block(col("eps_y_nm"), col("D_y"), col("value"), col("sigma_log"), [True] * len(rows), C_Y_GP, {},
                       "data/gp_exports/gp_requirements_for_wx.csv, rows quantity = n_y_req: value, D_y and sigma_log as "
                       f"exported (every exported point; the export has no 0.5 nm n_y^req); c_y = {C_Y_GP} from the 'cut "
                       "multipliers' line of data/gp_exports/gp_luminosity_for_wx.csv")


# ---------------------------------------------------------------------------------------------------------------------
# pinched core
# ---------------------------------------------------------------------------------------------------------------------
def core_width(runs):
    rule = ("data/joblist.csv runs of phases PQ, PQ2 with n_y = 4 n_y^cons(e_y) and 0.90 <= n_m / n_m^cons(e_y) <= 1.10; "
            "per run the minimum over steps of the Gaussian core width of the central slice (|z - z_bar| < 0.10 sigma_z), "
            "averaged over the two beams, from data/slice_widths_hw010_<label>.csv; fit error from "
            "data/slice_fiterr_hw010_<label>.csv")
    per, nms, excluded = defaultdict(list), defaultdict(set), Counter()
    for r in csv.DictReader(open(DATA / "joblist.csv")):
        if r["phase"] not in ("PQ", "PQ2"):
            continue
        e = float(r["e_y_nm"]); ratio_nm = float(r["nm"]) / Q.n_m_cons(e)
        if not (int(r["ny"]) == 4 * Q.n_y_cons(e) and 0.90 <= ratio_nm <= 1.10):
            excluded[(r["phase"], e, int(r["ny"]), round(ratio_nm, 3))] += 1
            continue
        if r["status"] != "done":
            fail(f"core width: selected run {r['label']} is not complete (status '{r['status']}')")
        wf, ff = DATA / f"slice_widths_hw010_{r['label']}.csv", DATA / f"slice_fiterr_hw010_{r['label']}.csv"
        for f in (wf, ff):
            if not f.is_file():
                fail(f"core width: {f.relative_to(ROOT)} (selected run {r['label']}) is missing")
        w = np.genfromtxt(wf, delimiter=",", names=True); fe = np.genfromtxt(ff, delimiter=",", names=True)
        v = [np.nanmin(w["gauss" + b]) for b in ("1", "2")]
        per[e].append((0.5 * (v[0] + v[1]), 0.5 * float(np.hypot(fe["err"][0], fe["err"][1]))))
        nms[e].add((int(r["ny"]), int(float(r["nm"])), round(ratio_nm, 4)))
    if sorted(per) != Q.EYS:
        fail(f"core width: qualifying runs at {sorted(per)} nm, expected {Q.EYS}")
    Dt, R1t = (np.load(DATA / "R1_table.npz")[k] for k in ("D", "R1"))
    rows = []
    for e in Q.EYS:
        s0, D = Q.sigma_y(e), Q.D_y(e)
        g = np.mean([p[0] for p in per[e]]) / s0
        sd = np.std([p[0] for p in per[e]], ddof=1) / s0
        fer = np.mean([p[1] for p in per[e]]) / s0
        err = float(np.hypot(sd, fer)); R1 = float(np.interp(D, Dt, R1t))
        if len(nms[e]) != 1:
            fail(f"core width: selected runs at e_y = {e:g} nm mix settings {sorted(nms[e])}")
        (ny, nm, rnm), = nms[e]
        rows.append(dict(e_y_nm=e, D_y=D, sigma_min_over_sigma_y_star=float(g),
                         uncertainty=U(err, "sample standard deviation over seeds (ddof = 1) of the per-seed core minimum, "
                                            "and the mean Gaussian-fit parameter error, added in quadrature; normalised by sigma_y^*"),
                         first_waist_prediction_1_over_R1=1 / R1,
                         measured_over_prediction=float(g * R1), measured_over_prediction_uncertainty=U(err * R1, "the width uncertainty times R_1"),
                         n_seeds=len(per[e]), n_y=ny, n_m=nm, n_m_over_n_m_cons=rnm))
    note = {f"{p} {e:g} nm, n_y {ny}, n_m/n_m^cons {rr}": f"{c} runs: off the locus rule"
            for (p, e, ny, rr), c in sorted(excluded.items())}
    return dict(rows=rows, provenance=dict(selection=rule, R_1="data/R1_table.npz, linear interpolation in D_y"),
                note_12_16_nm="n_m / n_m^cons = 0.943 and 0.966 at 12 and 16 nm lie inside the [0.90, 1.10] window and are included",
                excluded_runs=note, luminosity_at_refined_grid=refined_grid_luminosity(runs, nms),
                width_at_halved_cell=halved_cell_width())


def halved_cell_width():
    """Core width at n_y = 2 n_y^cons against n_y = 4 n_y^cons, where both grids were run: e_y = 0.5 and 1 nm.
    Halving the cell size divides the per-pass deposition error by four. The two grids carry different n_m, so the
    per-cell occupancy of each is reported with it."""
    def widths(e, factor):
        out = []
        for r in csv.DictReader(open(DATA / "joblist.csv")):
            if r["phase"] not in ("PQ", "PQ2") or float(r["e_y_nm"]) != e or int(r["ny"]) != factor * Q.n_y_cons(e):
                continue
            wf = DATA / f"slice_widths_hw010_{r['label']}.csv"
            if r["status"] != "done" or not wf.is_file():
                fail(f"halved-cell width: run {r['label']} is not complete or its width cache is missing")
            w = np.genfromtxt(wf, delimiter=",", names=True)
            out.append((0.5 * sum(np.nanmin(w["gauss" + b]) for b in ("1", "2")), float(r["nm"])))
        if len(out) < 2:
            fail(f"halved-cell width: {len(out)} run(s) at e_y = {e:g} nm, n_y = {factor} n_y^cons")
        if len({nm for _, nm in out}) != 1:
            fail(f"halved-cell width: runs at e_y = {e:g} nm, n_y = {factor} n_y^cons mix n_m")
        v = np.array([g for g, _ in out]) / Q.sigma_y(e)
        nm = out[0][1]
        return dict(n_y=factor * Q.n_y_cons(e), n_m=int(nm), n_seeds=len(v),
                    occupancy_over_recommended=(nm / Q.n_m_cons(e)) / factor,
                    sigma_min_over_sigma_y_star=float(v.mean()),
                    se=U(float(v.std(ddof=1) / math.sqrt(len(v))), SE))
    rows = []
    for e in (0.5, 1):
        coarse, fine = widths(e, 2), widths(e, 4)
        a, b = coarse["sigma_min_over_sigma_y_star"], fine["sigma_min_over_sigma_y_star"]
        rows.append(dict(e_y_nm=e, coarser_grid=coarse, finer_grid=fine,
                         relative_change=b / a - 1,
                         relative_change_se=U(math.hypot(fine["se"]["value"], b * coarse["se"]["value"] / a) / a, RATIO_SE)))
    return dict(rows=rows, provenance=dict(
        selection="phases PQ, PQ2 at n_y = 2 n_y^cons and n_y = 4 n_y^cons; widths as in the core-width rows above",
        definition="relative_change = (width at 4 n_y^cons) / (width at 2 n_y^cons) - 1, at each emittance"))


def refined_grid_luminosity(runs, nms):
    """Luminosity of the core-width runs (n_y = 4 n_y^cons) against the recommended configuration (LOCUS_RULE) at the
    same emittance, with the n_m of each side stated. Seed means first; ratio of the two seed means."""
    rows = []
    for e in Q.EYS:
        (_, nm_ref, _), = nms[e]
        sides = {}
        for tag, grid, window in (("refined", (512, 4 * Q.n_y_cons(e), 128), lambda nm: nm == nm_ref),
                                  ("recommended", (512, Q.n_y_cons(e), 128), lambda nm: abs(nm / Q.n_m_cons(e) - 1) < 0.02)):
            per = defaultdict(list)
            for r in runs:
                if r["e"] == e and r["solver"] == "3d" and r["grid"] == grid and window(r["nm"]):
                    per[r["seed"]].append(r["L"])
            if len(per) < 2:
                fail(f"refined-grid luminosity: {len(per)} seed(s) at e_y = {e:g} nm on the {tag} grid {grid}")
            settings = {r["nm"] for r in runs if r["e"] == e and r["solver"] == "3d" and r["grid"] == grid and window(r["nm"])}
            if len(settings) != 1:
                fail(f"refined-grid luminosity: runs at e_y = {e:g} nm on the {tag} grid mix n_m {sorted(settings)}")
            L = [float(np.mean(v)) for v in per.values()]
            sides[tag] = dict(n_y=grid[1], n_m=int(settings.pop()), n_seeds=len(L), L_mean_1e34=float(np.mean(L)),
                              L_se_1e34=U(float(np.std(L, ddof=1) / math.sqrt(len(L))), SE))
        a, b = sides["refined"], sides["recommended"]
        rows.append(dict(e_y_nm=e, refined=a, recommended=b,
                         n_m_refined_over_recommended=a["n_m"] / b["n_m"],
                         L_ratio=a["L_mean_1e34"] / b["L_mean_1e34"],
                         L_ratio_se=U(a["L_mean_1e34"] / b["L_mean_1e34"] *
                                      math.hypot(a["L_se_1e34"]["value"] / a["L_mean_1e34"],
                                                 b["L_se_1e34"]["value"] / b["L_mean_1e34"]), RATIO_SE)))
    return dict(rows=rows, provenance=dict(
        selection="refined: the core-width runs above; recommended: " + LOCUS_RULE,
        definition="L_ratio = refined / recommended, each a mean over per-seed means of data/results.csv"))


# ---------------------------------------------------------------------------------------------------------------------
def build():
    runs = load_runs()
    on_locus = lambda e, g, nm: g == (512, Q.n_y_cons(e), 128) and abs(nm / Q.n_m_cons(e) - 1) < 0.02
    ds = {
        "nominal": aggregate(runs, "nominal", "3d", ALL_EYS, lambda e, g, nm: g == (512, 512, 256) and abs(nm - 1e5) < 1,
                             "solver 3d; (n_x, n_y, n_z) = (512, 512, 256); n_m = 1e5"),
        "conservative": aggregate(runs, "conservative", "3d", Q.EYS, on_locus, LOCUS_RULE),
        "extrapolated_extension": aggregate(runs, "extrapolated_extension", "3d", Q.EYS_EXT, on_locus,
                                            LOCUS_RULE + "; e_y beyond the range the locus was fitted on"),
        "frozen_extension": aggregate(runs, "frozen_extension", "3d", Q.EYS_EXT,
                                      lambda e, g, nm: g == (512, 256, 128) and abs(nm / 1e4 - 1) < 0.02, FROZEN_RULE),
    }
    at = lambda cfg, e: next(r for r in ds[cfg] if r["e_y_nm"] == e)

    lg = {f"{e:g}": dict(e_y_nm=e, sigma_y_star_m=Q.sigma_y(e), L_geom_1e34=L_geom(e)) for e in ALL_EYS}
    derived = dict(
        conservative_over_nominal=[dict(e_y_nm=e, **ratio(at("conservative", e), at("nominal", e))) for e in Q.EYS],
        H_D=dict(definition="H_D = L / L_geom; its standard deviation and standard error are those of L divided by L_geom",
                 L_geom=dict(formula="L_geom = f_coll N^2 / (4 pi sigma_x sigma_y^*), in 1e34 cm^-2 s^-1",
                             parameters=dict(f_coll_Hz=F_COLL, bunches=W.NUM_BUNCHES, train_rep_Hz=W.TRAIN_REP, N=Q.N_PART,
                                             sigma_x_m=Q.SIGMA_X, sigma_y_star="sqrt(beta_y eps_y / gamma)",
                                             beta_y_m=Q.BETA_Y, gamma=Q.GAMMA),
                             per_emittance=list(lg.values())),
                 values=[dict(configuration=c, e_y_nm=r["e_y_nm"], H_D=r["L_mean_1e34"] / lg[f"{r['e_y_nm']:g}"]["L_geom_1e34"],
                              std=U(r["L_std_1e34"]["value"] / lg[f"{r['e_y_nm']:g}"]["L_geom_1e34"], STD),
                              se=U(r["L_se_1e34"]["value"] / lg[f"{r['e_y_nm']:g}"]["L_geom_1e34"], SE))
                         for c in ds for r in ds[c]]),
        gain_20_to_2_nm_conservative=dict(definition="L_conservative(2 nm) / L_conservative(20 nm)",
                                          **ratio(at("conservative", 2.0), at("conservative", 20.0))),
        relative_seed_scatter=[dict(configuration=c, e_y_nm=r["e_y_nm"],
                                    value=U(r["L_std_1e34"]["value"] / r["L_mean_1e34"], SCATTER)) for c in ds for r in ds[c]],
        provenance="section 'luminosity_dataset'")

    A, B = Q.requirements_stat(), Q.ny_requirements_stat()
    for name, stat in (("n_m^req", A), ("n_y^req", B)):
        if sorted(s[0] for s in stat) != Q.EYS:
            fail(f"{name} extracted at {sorted(s[0] for s in stat)} nm, expected {Q.EYS}")
    requirements = dict(
        n_m_req=dict(rows=requirement_rows(A, "n_m"),
                     provenance="data/results.csv, solver 3d, the n_m tuning ladder of each e_y on its tuning grid "
                                "(512, n_y, 128) with n_y = 512 at 0.5 nm and 256 otherwise, every n_m rung, same-seed repeats "
                                "averaged. L_inf = mean of the two top rungs (a decade apart, agreeing within 0.5%). n_m^req "
                                "= log-log interpolation of the -5% crossing between the last rung below -5% and the next rung."),
        n_y_req=dict(rows=requirement_rows(B, "n_y"),
                     provenance="data/results.csv, solver 3d, the n_y tuning ladder of each e_y: (n_x, n_z) = (512, 128), "
                                "n_m = NM_CONS[e_y] (the tuning-ladder table in nmreq.py) within 2%, every n_y rung, same-seed "
                                "repeats averaged. L_inf = the top rung (the two-rung plateau rule needs rungs a decade apart; "
                                "n_y rungs are a factor 2 apart). n_y^req = log-log interpolation of the -5% crossing."))
    fits, kappa = fits_and_kappa(A, B)

    rec = []
    for e in ALL_EYS:
        D = Q.D_y(e); nmc = Q.n_m_cons(e); run = at("conservative" if e in Q.EYS else "extrapolated_extension", e)
        rec.append(dict(e_y_nm=e, D_y=D, n_x=512, n_y=Q.n_y_cons(e), n_z=128, n_m=math.ceil(nmc), n_m_unrounded=nmc,
                        n_y_law_unrounded=Q.LOCUS_NY_PREFACTOR * D ** Q.LOCUS_NY_EXPONENT,
                        n_m_per_cell=nmc / (512 * Q.n_y_cons(e) * 128), n_y_run=run["n_y"], n_m_run=run["n_m"],
                        range="fitted range" if e in Q.EYS else "extrapolated beyond the fitted range"))
    recommendation = dict(
        rule=f"n_y = smallest of {{32, ..., 8192}} >= {Q.LOCUS_NY_PREFACTOR:.1f} D_y^{Q.LOCUS_NY_EXPONENT:.6f}; "
             f"n_m = ceil({Q.LOCUS_NM_PREFACTOR:.3f} D_y^{Q.LOCUS_NM_EXPONENT:.3f}); (n_x, n_z) = (512, 128); frozen locus constants",
        rows=rec,
        note="n_y_run / n_m_run are the settings of the conservative and extrapolated_extension runs. At 20 nm the runs "
             "used n_m = 1e4, 0.2% below the locus; at 40-100 nm the runs used the locus rounded to two significant figures.")

    dy = dict(expression="D_y = 2 N r_e sigma_z / (gamma sigma_y^* (sigma_x + sigma_y^*)), sigma_y^* = sqrt(beta_y eps_y / gamma)",
              parameters=dict(N=Q.N_PART, r_e_m=Q.R_E, sigma_z_m=Q.SIGMA_Z, sigma_x_m=Q.SIGMA_X, beta_y_m=Q.BETA_Y,
                              gamma=Q.GAMMA, gamma_from="125 GeV / 0.51099895 MeV"),
              per_emittance=[dict(e_y_nm=e, sigma_y_star_m=Q.sigma_y(e), D_y=Q.D_y(e)) for e in ALL_EYS])

    tuning_table = lambda e, g, nm: g == (512, Q.NY_CONS[e], 128) and abs(nm / Q.NM_CONS[e] - 1) < 0.02
    variants = {}
    for key, solver, label in (("reference_3d_solver_3rd_order", "3d", "3D solver, 3rd-order deposition (the calibration configuration)"),
                               ("2d_slice_solver_3rd_order", "2d", "2D-slice solver, 3rd-order deposition (phase PS)"),
                               ("3d_solver_1st_order_cic", "3d_cic", "3D solver, 1st-order (CIC) deposition (phase PC3)"),
                               ("2d_slice_solver_1st_order_cic", "2d_cic", "2D-slice solver, 1st-order (CIC) deposition (phase PC2)")):
        rows = aggregate(runs, key, solver, Q.EYS, tuning_table, f"solver {solver}; " + TUNING_TABLE_RULE)
        for r in rows:
            r["conservative_locus_n_m"] = math.ceil(Q.n_m_cons(r["e_y_nm"]))
        variants[key] = dict(description=label, rows=rows)
    ref = {r["e_y_nm"]: r for r in variants["reference_3d_solver_3rd_order"]["rows"]}
    for key in list(variants)[1:]:
        variants[key]["over_reference"] = [dict(e_y_nm=r["e_y_nm"], **ratio(r, ref[r["e_y_nm"]])) for r in variants[key]["rows"]]
    variants_block = dict(note="All four were run at the tuning-ladder settings (NM_CONS / NY_CONS); 'conservative_locus_n_m' gives the "
                               "production-locus value for comparison. "
                               "Reported at every emittance run.", **variants)

    ps = at("nominal", 20.0)
    ps1 = dict(definition="PS1 reference: the nominal configuration at 20 nm", L_mean_1e34=ps["L_mean_1e34"],
               L_std_1e34=ps["L_std_1e34"], L_se_1e34=ps["L_se_1e34"], n_seeds=ps["n_seeds"],
               published_1e34=PS1_PUBLISHED, ratio_to_published=ps["L_mean_1e34"] / PS1_PUBLISHED,
               ratio_se=U(ps["L_se_1e34"]["value"] / PS1_PUBLISHED, "standard error of L / published (published value exact)"),
               provenance="section 'luminosity_dataset', configuration nominal, e_y = 20 nm")

    wx_tuned = {r["e_y_nm"]: (c, r) for c in ("conservative", "extrapolated_extension") for r in ds[c]}
    gp_tuned = {e: ("conservative", r) for e, r in gp_luminosity("conservative").items()}
    gp_tuned.update({e: ("frozen_extension", r) for e, r in gp_luminosity("frozen_extension").items() if e not in gp_tuned})
    per_e = [dict(e_y_nm=e, wx_configuration=wx_tuned[e][0], gp_block=gp_tuned[e][0], **ratio(wx_tuned[e][1], gp_tuned[e][1]))
             for e in ALL_EYS if e in wx_tuned and e in gp_tuned]
    band = [r for r in per_e if 8 <= r["e_y_nm"] <= 100]
    lo, hi = min(band, key=lambda r: r["value"]), max(band, key=lambda r: r["value"])
    wx_over_gp = dict(
        definition="L_WarpX / L_GP++ per emittance, each simulator at its recommended configuration: WarpX 'conservative' "
                   "(0.5-20 nm) and 'extrapolated_extension' (40-100 nm) from section 'luminosity_dataset'; GUINEA-PIG++ "
                   "blocks 'conservative' (1-20 nm) and 'frozen_extension' (40-100 nm) of "
                   "data/gp_exports/gp_luminosity_for_wx.csv. The series drawn in WX_code_ratio_vs_ey.",
        per_emittance=per_e,
        range_8_to_100_nm=dict(min=lo["value"], e_y_nm_at_min=lo["e_y_nm"], max=hi["value"], e_y_nm_at_max=hi["e_y_nm"]),
        at_1_nm=next(dict(value=r["value"], se=r["se"]) for r in per_e if r["e_y_nm"] == 1.0),
        provenance="section 'luminosity_dataset' and data/gp_exports/gp_luminosity_for_wx.csv")

    return dict(
        about=dict(content="Every WarpX number reported in the paper, regenerated from committed data by reproduce_paper_numbers.py.",
                   units="luminosity L in 1e34 cm^-2 s^-1 (per-crossing luminosity x 133 bunches x 120 Hz); D_y, H_D, "
                         "kappa, ratios dimensionless; lengths in m",
                   configurations=dict(nominal="untuned grid (512, 512, 256), n_m = 1e5",
                                       conservative="frozen production locus, 0.5-20 nm",
                                       extrapolated_extension="the same locus evaluated at 40-100 nm",
                                       frozen_extension="the 20 nm conservative settings held fixed at 40-100 nm; at 20 nm "
                                                        "this is the conservative point itself"),
                   same_seed_repeats="runs repeated at an identical configuration and seed are averaged into one sample "
                                     "before seed statistics; they are not independent",
                   monte_carlo=dict(generator="numpy.random.default_rng", seed=Q.MC_SEED, draws=Q.MC_DRAWS),
                   inputs=["data/results.csv", "data/joblist.csv", "data/R1_table.npz",
                           "data/gp_exports/gp_requirements_for_wx.csv", "data/gp_exports/gp_luminosity_for_wx.csv",
                           "data/slice_widths_hw010_<label>.csv", "data/slice_fiterr_hw010_<label>.csv"]),
        luminosity_dataset=dict(cut_multipliers_definition=CUT_NOTE, n_t_definition=NT_NOTE,
                                rows=[r for c in ds for r in ds[c]]),
        derived_luminosity=derived, requirements=requirements, fits=fits, kappa=kappa,
        recommendation_table=recommendation, disruption_parameter=dy, core_width=core_width(runs),
        solver_and_deposition_variants=variants_block, ps1_reference=ps1, wx_over_gp_luminosity=wx_over_gp)


# ---------------------------------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------------------------------
def pm(v, u, d):
    """value +- uncertainty at d decimals; an uncertainty that would round to zero keeps one significant figure."""
    us = f"{u:.{d}f}" if round(u, d) != 0 else f"{u:.1g}"
    return f"{v:.{d}f} ± {us}"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def markdown(P):
    ds = P["luminosity_dataset"]["rows"]; dv = P["derived_luminosity"]; fits = P["fits"]; out = []
    out.append("# WarpX paper numbers\n\nGenerated by `reproduce_paper_numbers.py` from the committed data; do not edit. "
               "Values are rounded to the precision the paper quotes; `paper_numbers.json` holds every value at full "
               "precision with the definition of each uncertainty and the selection rule behind each row.\n\n"
               "L in 10³⁴ cm⁻² s⁻¹. *std*: sample standard deviation over seeds (ddof = 1). *se*: standard error of the "
               "mean. Same-seed repeat runs are averaged before seed statistics.\n")
    out.append("## 1. Luminosity dataset\n\nCut multipliers (cuts x/y/z): " + P["luminosity_dataset"]["cut_multipliers_definition"] + ". "
               + P["luminosity_dataset"]["n_t_definition"] + ".\n")
    out.append(table(["configuration", "ε_y [nm]", "L ± std", "se", "seeds", "n_x × n_y × n_z", "n_t", "n_m", "cuts x/y/z", "runs by phase"],
                     [[r["configuration"], f"{r['e_y_nm']:g}", pm(r["L_mean_1e34"], r["L_std_1e34"]["value"], 2),
                       f"{r['L_se_1e34']['value']:.3f}", r["n_seeds"], f"{r['n_x']} × {r['n_y']} × {r['n_z']}", r["n_t"], r["n_m"],
                       "16/16/8", ", ".join(f"{k} {v}" for k, v in r["provenance"]["runs_per_phase"].items())] for r in ds]))
    out.append("\nSelection rules:\n\n" + "\n".join(
        f"- **{c}**: {next(r for r in ds if r['configuration'] == c)['provenance']['selection']}"
        for c in dict.fromkeys(r["configuration"] for r in ds)))

    out.append("\n## 2. Derived luminosity quantities\n\n**Conservative / nominal** (± se)\n")
    out.append(table(["ε_y [nm]", "cons/nom"], [[f"{r['e_y_nm']:g}", pm(r["value"], r["se"]["value"], 2)] for r in dv["conservative_over_nominal"]]))
    g = dv["gain_20_to_2_nm_conservative"]
    out.append(f"\n**Gain 20 → 2 nm, conservative**: L(2 nm)/L(20 nm) = {pm(g['value'], g['se']['value'], 2)} (± se)\n")
    L = dv["H_D"]["L_geom"]
    out.append(f"\n**H_D = L / L_geom** (± se). {L['formula']}; f_coll = 133 × 120 Hz, N = {L['parameters']['N']:g}, "
               f"σ_x = {L['parameters']['sigma_x_m'] * 1e9:g} nm, σ_y* = √(β_y ε_y/γ), β_y = 0.12 mm, γ = {Q.GAMMA:.1f}.\n")
    hd = {(r["configuration"], r["e_y_nm"]): r for r in dv["H_D"]["values"]}
    sc = {(r["configuration"], r["e_y_nm"]): r["value"]["value"] for r in dv["relative_seed_scatter"]}
    lgm = {r["e_y_nm"]: r for r in L["per_emittance"]}
    cfgs = list(dict.fromkeys(r["configuration"] for r in ds))
    out.append(table(["ε_y [nm]", "σ_y* [nm]", "L_geom"] + [f"H_D {c}" for c in cfgs],
                     [[f"{e:g}", f"{lgm[e]['sigma_y_star_m'] * 1e9:.3f}", f"{lgm[e]['L_geom_1e34']:.3f}"]
                      + [pm(hd[(c, e)]["H_D"], hd[(c, e)]["se"]["value"], 2) if (c, e) in hd else "—" for c in cfgs] for e in ALL_EYS]))
    out.append("\n**Relative seed scatter** std/L [%]\n")
    out.append(table(["ε_y [nm]"] + cfgs, [[f"{e:g}"] + [f"{100 * sc[(c, e)]:.1f}" if (c, e) in sc else "—" for c in cfgs] for e in ALL_EYS]))

    out.append("\n## 3. Requirement extractions\n\nUncertainty σ_ln: " + MC + ".\n")
    rq = P["requirements"]
    out.append(table(["ε_y [nm]", "D_y", "n_m^req", "n_m^req/(n_x n_y n_z)", "σ_ln", "16–84% range", "in fit"],
                     [[f"{r['e_y_nm']:g}", f"{r['D_y']:.1f}", f"{r['value']:.3g}", f"{r['value_per_cell']:.2e}", f"{r['uncertainty_ln']['value']:.2f}",
                       f"{r['mc_percentile_16']:.3g}–{r['mc_percentile_84']:.3g}", "yes" if r["used_in_fit"] else f"no: {r['excluded_because']}"]
                      for r in rq["n_m_req"]["rows"]]))
    out.append("\n" + table(["ε_y [nm]", "D_y", "n_y^req", "σ_ln", "16–84% range", "in fit"],
                            [[f"{r['e_y_nm']:g}", f"{r['D_y']:.1f}", f"{r['value']:.3g}", f"{r['uncertainty_ln']['value']:.2f}",
                              f"{r['mc_percentile_16']:.3g}–{r['mc_percentile_84']:.3g}", "yes" if r["used_in_fit"] else f"no: {r['excluded_because']}"]
                             for r in rq["n_y_req"]["rows"]]))
    out.append("\n\n" + rq["n_m_req"]["provenance"] + "\n\n" + rq["n_y_req"]["provenance"] + "\n")

    out.append("\n## 4. Fitted constants\n\nUncertainties: " + FIT + ".\n")
    fm, fy, fc = fits["n_m"]["fit"], fits["n_y_free"]["fit"], fits["n_y_exponent_constrained"]["fit"]
    fma, fya = fits["n_m"]["same_fit_without_exclusions"], fits["n_y_free"]["same_fit_without_exclusions"]
    ex = lambda f: "; ".join(f"{k} nm ({v})" for k, v in f["excluded"].items())
    out.append(table(["fit", "exponent", "prefactor", "χ²/ndf", "excluded"], [
        [f"n_m: {fits['n_m']['law']}", f"s = {pm(fm['s']['estimate'], fm['s']['uncertainty']['value'], 3)}",
         f"C_m = {fm['prefactor']['estimate']:.2e} ± {fm['prefactor']['uncertainty']['value']:.1e}", f"{fm['chi2']:.2f}/{fm['ndf']}", ex(fits["n_m"])],
        ["n_m, no exclusions", f"s = {pm(fma['s']['estimate'], fma['s']['uncertainty']['value'], 3)}", f"C_m = {fma['prefactor']['estimate']:.2e}",
         f"{fma['chi2']:.2f}/{fma['ndf']}", "none"],
        [f"n_y free: {fits['n_y_free']['law']}", f"q = {pm(fy['exponent']['estimate'], fy['exponent']['uncertainty']['value'], 3)}",
         f"C_y = {pm(fy['prefactor']['estimate'], fy['prefactor']['uncertainty']['value'], 1)}", f"{fy['chi2']:.2f}/{fy['ndf']}", ex(fits["n_y_free"])],
        ["n_y free, no exclusions", f"q = {pm(fya['exponent']['estimate'], fya['exponent']['uncertainty']['value'], 3)}",
         f"C_y = {fya['prefactor']['estimate']:.1f}", f"{fya['chi2']:.2f}/{fya['ndf']}", "none"],
        [f"n_y constrained: {fits['n_y_exponent_constrained']['law']}", f"q_n = {fc['exponent_fixed']:.4f} (fixed)",
         f"C_y = {pm(fc['prefactor']['estimate'], fc['prefactor']['uncertainty']['value'], 1)}", f"{fc['chi2']:.2f}/{fc['ndf']}",
         ex(fits["n_y_exponent_constrained"])]]))
    lo = fits["conservative_loci"]; rm, ry = lo["n_m"]["recomputed_from_current_data"], lo["n_y"]["recomputed_from_current_data"]
    out.append(f"\n**Conservative loci — frozen.** {lo['statement']}\n\n"
               f"- n_m^cons = {lo['n_m']['C_frozen']:.3f} D_y^{lo['n_m']['b_frozen']:.3f} (frozen). Recomputed today: exponent "
               f"{pm(rm['b']['estimate'], rm['b']['uncertainty']['value'], 3)}, χ²/ndf = {rm['chi2']:.2f}/{rm['ndf']}, C = {rm['C']:.3f}.\n"
               f"- n_y^cons = smallest power of two ≥ {lo['n_y']['C_frozen']:.1f} D_y^{lo['n_y']['q_frozen']:.6f} (frozen). Recomputed "
               f"today: exponent {pm(ry['q']['estimate'], ry['q']['uncertainty']['value'], 3)}, χ²/ndf = {ry['chi2']:.2f}/{ry['ndf']}, C = {ry['C']:.1f}.\n")

    k = P["kappa"]
    out.append("\n## 5. Criterion constant κ\n\n" + k["definition"] + "\n")
    for code, name in (("warpx", "WarpX"), ("gp", "GUINEA-PIG++")):
        b = k[code]; m = b["weighted_mean"]; s = b["slope_ln_kappa_vs_ln_D_y"]
        out.append(f"\n**{name}** (c_y = {b['c_y']:g})\n")
        out.append(table(["ε_y [nm]", "D_y", "n_y^req", "σ_ln", "R_1", "κ ± σ", "in mean and slope"],
                         [[f"{r['e_y_nm']:g}", f"{r['D_y']:.1f}", f"{r['n_y_req']:.3g}", f"{r['sigma_ln']:.3f}", f"{r['R_1']:.3f}",
                           pm(r["kappa"], r["uncertainty"]["value"], 3),
                           "yes" if r["used_in_mean_and_slope"] else f"no: {r['excluded_because']}"] for r in b["per_emittance"]]))
        out.append(f"\n⟨κ⟩ = {pm(m['estimate'], m['uncertainty']['value'], 3)}. Slope of ln κ vs ln D_y = "
                   f"{s['estimate']:+.3f} ± {s['uncertainty']['value']:.3f}, χ²/ndf = {s['chi2']:.2f}/{s['ndf']}.\n")
    r = k["ratio_warpx_over_gp"]
    out.append(f"\n**κ_WarpX / κ_GP** = {pm(r['estimate'], r['uncertainty']['value'], 3)}; agreement "
               f"{k['agreement_sigma']['value']:.1f}σ.\n\n")

    rt = P["recommendation_table"]
    out.append("\n## 6. Recommendation table (WarpX)\n\n" + rt["rule"] + ".\n")
    out.append(table(["ε_y [nm]", "D_y", "n_x", "n_y", "n_z", "n_m", "n_m/(n_x n_y n_z)", "run at (n_y, n_m)", "range"],
                     [[f"{r['e_y_nm']:g}", f"{r['D_y']:.1f}", r["n_x"], r["n_y"], r["n_z"], r["n_m"], f"{r['n_m_per_cell']:.3g}",
                       f"({r['n_y_run']}, {r['n_m_run']})", r["range"]] for r in rt["rows"]]))
    out.append("\n" + rt["note"] + "\n")

    dy = P["disruption_parameter"]
    out.append("\n## 7. Disruption parameter\n\n" + dy["expression"] + "; N = 6.24×10⁹, r_e = 2.8179403262×10⁻¹⁵ m, "
               "σ_z = 100 μm, σ_x = 210.12 nm, β_y = 0.12 mm, γ = 125 GeV / m_e c².\n")
    out.append(table(["ε_y [nm]", "σ_y* [nm]", "D_y"], [[f"{r['e_y_nm']:g}", f"{r['sigma_y_star_m'] * 1e9:.3f}", f"{r['D_y']:.1f}"]
                                                          for r in dy["per_emittance"]]))

    cw = P["core_width"]
    out.append("\n## 8. Pinched-core width\n\nSelection: " + cw["provenance"]["selection"] + ". " + cw["note_12_16_nm"] + ".\n")
    out.append(table(["ε_y [nm]", "D_y", "σ_min/σ_y* ± err", "1/R_1", "measured/(1/R_1)", "seeds", "n_y", "n_m", "n_m/n_m^cons"],
                     [[f"{r['e_y_nm']:g}", f"{r['D_y']:.1f}", pm(r["sigma_min_over_sigma_y_star"], r["uncertainty"]["value"], 3),
                       f"{r['first_waist_prediction_1_over_R1']:.3f}", pm(r["measured_over_prediction"], r["measured_over_prediction_uncertainty"]["value"], 2),
                       r["n_seeds"], r["n_y"], r["n_m"], f"{r['n_m_over_n_m_cons']:.3f}"] for r in cw["rows"]]))
    out.append("\nerr: " + cw["rows"][0]["uncertainty"]["definition"] + ". Excluded runs: "
               + "; ".join(f"{k} ({v})" for k, v in cw["excluded_runs"].items()) + ".\n")

    rg = cw["luminosity_at_refined_grid"]
    out.append("\nLuminosity on the refined grid against the recommended configuration (" + rg["provenance"]["definition"] + ").\n")
    out.append(table(["ε_y [nm]", "n_y refined / recommended", "n_m refined / recommended", "L refined", "L recommended", "ratio"],
                     [[f"{r['e_y_nm']:g}", f"{r['refined']['n_y']} / {r['recommended']['n_y']}",
                       f"{r['refined']['n_m']} / {r['recommended']['n_m']}",
                       f"{r['refined']['L_mean_1e34']:.4f} ({r['refined']['n_seeds']})",
                       f"{r['recommended']['L_mean_1e34']:.4f} ({r['recommended']['n_seeds']})",
                       pm(r["L_ratio"], r["L_ratio_se"]["value"], 4)] for r in rg["rows"]]))

    hc = cw["width_at_halved_cell"]
    out.append("\nCore width at n_y = 2 n_y^cons against 4 n_y^cons (" + hc["provenance"]["definition"] + ").\n")
    out.append(table(["ε_y [nm]", "n_y", "n_m", "occupancy / recommended", "σ_min/σ_y*", "change [%]"],
                     [row for r in hc["rows"] for row in (
                         [f"{r['e_y_nm']:g}", r["coarser_grid"]["n_y"], r["coarser_grid"]["n_m"],
                          f"{r['coarser_grid']['occupancy_over_recommended']:.3f}",
                          f"{r['coarser_grid']['sigma_min_over_sigma_y_star']:.4f}", ""],
                         ["", r["finer_grid"]["n_y"], r["finer_grid"]["n_m"],
                          f"{r['finer_grid']['occupancy_over_recommended']:.3f}",
                          f"{r['finer_grid']['sigma_min_over_sigma_y_star']:.4f}",
                          pm(100 * r["relative_change"], 100 * r["relative_change_se"]["value"], 2)])]))

    sv = P["solver_and_deposition_variants"]
    out.append("\n## 9. Solver and deposition variants\n\n" + sv["note"] + "\n")
    keys = [k for k in sv if k != "note"]
    ref = {r["e_y_nm"]: r for r in sv[keys[0]]["rows"]}
    rows = []
    for e in Q.EYS:
        row = [f"{e:g}", ref[e]["n_y"], ref[e]["n_m"], ref[e]["conservative_locus_n_m"]]
        for kk in keys:
            r = next(x for x in sv[kk]["rows"] if x["e_y_nm"] == e)
            cell = f"{pm(r['L_mean_1e34'], r['L_std_1e34']['value'], 2)} ({r['n_seeds']})"
            if "over_reference" in sv[kk]:
                o = next(x for x in sv[kk]["over_reference"] if x["e_y_nm"] == e)
                cell += f"; ×{pm(o['value'], o['se']['value'], 3)}"
            row.append(cell)
        rows.append(row)
    out.append(table(["ε_y [nm]", "n_y", "n_m run", "n_m^cons today"] + [sv[k]["description"] for k in keys], rows))
    out.append("\nCells: L ± std (seeds); for the variants, ×(ratio to the 3D 3rd-order reference ± se).\n")

    p = P["ps1_reference"]
    out.append("\n## 10. PS1 reference point\n\n"
               f"Nominal configuration at 20 nm: L = {pm(p['L_mean_1e34'], p['L_std_1e34']['value'], 2)} (std), "
               f"se {p['L_se_1e34']['value']:.3f}, {p['n_seeds']} seeds; ratio to the published {p['published_1e34']} = "
               f"{pm(p['ratio_to_published'], p['ratio_se']['value'], 3)} (± se).\n")

    g = P["wx_over_gp_luminosity"]; b = g["range_8_to_100_nm"]
    out.append("\n## 11. WarpX / GUINEA-PIG++ luminosity ratio\n\n" + g["definition"] + "\n")
    out.append(table(["ε_y [nm]", "WarpX configuration", "GP++ block", "L_WarpX / L_GP++ ± se"],
                     [[f"{r['e_y_nm']:g}", r["wx_configuration"], r["gp_block"], pm(r["value"], r["se"]["value"], 3)]
                      for r in g["per_emittance"]]))
    out.append(f"\nOver 8–100 nm: {b['min']:.3f} ({b['e_y_nm_at_min']:g} nm) to {b['max']:.3f} ({b['e_y_nm_at_max']:g} nm). "
               f"At 1 nm: {pm(g['at_1_nm']['value'], g['at_1_nm']['se']['value'], 3)}.\n")
    return "\n".join(out)


def main():
    P = build()
    OUT_JSON.write_text(json.dumps(P, indent=1, allow_nan=False) + "\n")
    OUT_MD.write_text(markdown(P))
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")


if __name__ == "__main__":
    main()
