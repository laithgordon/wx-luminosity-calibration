#!/usr/bin/env python3
"""Pinched-core width: analysis-choice systematic and exponent fit.

The minimum core width sigma_min/sigma_y* of plot_core_width (Gaussian fit to the central-slice y-histogram, per beam
the minimum over steps, beams averaged) depends on three estimator settings: the slice half-width HW, the fit threshold
(fraction of the histogram peak above which bins enter the fit) and the histogram bin count. Each is varied one at a
time about the default (HW = 0.10 sigma_z, threshold 0.20, 80 bins over +-5 IQR) on the same runs and seeds:
HW in {0.05, 0.20} sigma_z, threshold in {0.10, 0.30}, bin width x0.5 and x2 (160 and 40 bins).
The systematic per emittance is the maximum absolute deviation of the 10-seed mean from the default over the six
variations; the RMS deviation is recorded alongside.

The exponent fit is the weighted least-squares fit of ln R = ln(sigma_y*/sigma_min) = a + q ln D_y, weights 1/sigma^2
with sigma = (se (+) systematic) / value, where se = sample standard deviation over seeds (ddof = 1) / sqrt(n_seeds) of
the per-seed core minimum (the Gaussian-fit parameter error is not added: it is part of the seed-to-seed scatter) and
(+) is addition in quadrature. Lambda = exp(a). Uncertainties are not rescaled by chi2/ndf.

Reads only committed files: data/joblist.csv, data/R1_table.npz and the caches data/slice_widths_<tag>_<label>.csv,
data/slice_fiterr_<tag>_<label>.csv of the 80 runs on the core-width locus rule (tag 'hw010' for the default settings,
e.g. 'hw005_thr020_nb080' for a variation). Needs only numpy.

    python core_width_syst.py                 # print the systematic and fit tables
    WX_RUN_ROOT=... python core_width_syst.py --build [--workers N]   # write missing caches from the particle dumps
"""
import csv, math, os, sys
from pathlib import Path
import numpy as np
import nmreq as Q

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASELINE = (0.10, 0.20, 80)          # (HW [sigma_z], fit threshold [of the peak], bins): plot_core_width.HW, FIT_THRESHOLD, NBINS
VARIATIONS = {"HW 0.05": (0.05, 0.20, 80), "HW 0.20": (0.20, 0.20, 80),
              "threshold 0.10": (0.10, 0.10, 80), "threshold 0.30": (0.10, 0.30, 80),
              "bin width x0.5": (0.10, 0.20, 160), "bin width x2": (0.10, 0.20, 40)}
FIT_SETS = {"all_8": Q.EYS, "1_to_20_nm": [e for e in Q.EYS if e != 0.5]}
SELECTION = ("data/joblist.csv runs of phases PQ, PQ2 with n_y = 4 n_y^cons(e_y) and 0.90 <= n_m / n_m^cons(e_y) <= 1.10 "
             "(the pinched-core rows), all ten seeds at every emittance")
SE = "standard error of the mean: sample standard deviation over seeds (ddof = 1) of the per-seed core minimum / sqrt(n_seeds)"
SEED_SCATTER = ("sample standard deviation over seeds (ddof = 1) of the per-seed core minimum, and the mean Gaussian-fit "
                "parameter error, added in quadrature (the pinched-core rows' uncertainty)")
SYST_MAX = "maximum over the six variations of |mean(variation) - mean(default)|, same runs and seeds"
SYST_RMS = "root mean square over the six variations of mean(variation) - mean(default)"
PAIRED = "sample standard deviation over seeds (ddof = 1) of the per-seed difference variation - default, / sqrt(n_seeds)"
TOTAL = "se (+) systematic (maximum deviation), in quadrature"
SIG_LN = "sigma on ln R = (se (+) systematic) / value"
FIT = ("1-sigma parameter uncertainty from the covariance of the weighted least-squares fit of ln R = a + q ln D_y, "
       "weights 1/sigma_ln^2 with " + SIG_LN + " (not rescaled by chi2/ndf)")


def cache_tag(hw, thr, nb):
    """Cache-name tag of an estimator setting: 'hw010' for the default, 'hw<HW x 100>_thr<threshold x 100>_nb<bins>'
    (e.g. 'hw005_thr020_nb080') for any other."""
    tag = f"hw{round(hw * 100):03d}"
    return tag if (hw, thr, nb) == BASELINE else f"{tag}_thr{round(thr * 100):03d}_nb{nb:03d}"


def locus_runs():
    """{e_y: [labels in seed order]} of the completed runs on the pinched-core locus rule."""
    out = {}
    for r in csv.DictReader(open(DATA / "joblist.csv")):
        if r["phase"] not in ("PQ", "PQ2"):
            continue
        e = float(r["e_y_nm"])
        if int(r["ny"]) == 4 * Q.n_y_cons(e) and 0.90 <= float(r["nm"]) / Q.n_m_cons(e) <= 1.10:
            if r["status"] != "done":
                raise RuntimeError(f"core-width run {r['label']} is not complete (status '{r['status']}')")
            out.setdefault(e, []).append((int(r["seed"]), r["label"]))
    if sorted(out) != Q.EYS:
        raise RuntimeError(f"core-width runs at {sorted(out)} nm, expected {Q.EYS}")
    return {e: [l for _, l in sorted(v)] for e, v in out.items()}


def run_value(label, s, dumps=False):
    """(beam-averaged minimum core width [m], half the quadrature sum of the two beams' fit errors) of one run at
    setting s. Reads the caches; with dumps=True missing caches are first built from the particle dumps."""
    wf, ff = (DATA / f"slice_{k}_{cache_tag(*s)}_{label}.csv" for k in ("widths", "fiterr"))
    if not (wf.is_file() and ff.is_file()):
        if not dumps:
            raise FileNotFoundError(f"core-width cache missing: {(wf if not wf.is_file() else ff).relative_to(ROOT)}")
        import plot_core_width as PCW
        PCW.build_widths(label, [s])
    w = np.genfromtxt(wf, delimiter=",", names=True); fe = np.genfromtxt(ff, delimiter=",", names=True)
    v = [np.nanmin(w["gauss" + b]) for b in ("1", "2")]
    return 0.5 * (v[0] + v[1]), 0.5 * float(np.hypot(fe["err"][0], fe["err"][1]))


def point(e, labels, s, dumps=False):
    """10-seed statistics of sigma_min/sigma_y* at one emittance and setting (same arithmetic as the pinched-core rows)."""
    p = [run_value(l, s, dumps) for l in labels]
    s0 = Q.sigma_y(e)
    g = np.mean([v[0] for v in p]) / s0
    sd = np.std([v[0] for v in p], ddof=1) / s0
    fer = np.mean([v[1] for v in p]) / s0
    return dict(value=float(g), seed_scatter=float(np.hypot(sd, fer)), se=float(sd / math.sqrt(len(p))),
                per_seed=[v[0] / s0 for v in p])


def chi2_sf(x, k):
    """P(chi^2_k > x) for integer k >= 1, in closed form."""
    if k % 2 == 0:
        t = term = math.exp(-x / 2)
        for i in range(1, k // 2):
            term *= x / (2 * i); t += term
        return t
    t = math.erfc(math.sqrt(x / 2)); term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for i in range(1, (k + 1) // 2):
        t += term; term *= x / (2 * i + 1)
    return t


def power_fit(D, y, sig_ln):
    """Q.powerlaw_wls of ln y = a + b ln D_y with chi2 p-value."""
    f = Q.powerlaw_wls(D, y, sig_ln)
    f["p"] = chi2_sf(f["chi2"], f["ndf"])
    return f


def analysis(dumps=False):
    runs = locus_runs()
    Dt, R1t = (np.load(DATA / "R1_table.npz")[k] for k in ("D", "R1"))
    D = {e: Q.D_y(e) for e in Q.EYS}
    R1 = {e: float(np.interp(D[e], Dt, R1t)) for e in Q.EYS}
    base = {e: point(e, runs[e], BASELINE, dumps) for e in Q.EYS}
    var = {n: {e: point(e, runs[e], s, dumps) for e in Q.EYS} for n, s in VARIATIONS.items()}
    syst = {}
    for e in Q.EYS:
        dev = {n: var[n][e]["value"] - base[e]["value"] for n in VARIATIONS}
        syst[e] = dict(max=max(abs(v) for v in dev.values()), rms=math.sqrt(sum(v * v for v in dev.values()) / len(dev)),
                       dev=dev, paired={n: float(np.std(np.subtract(var[n][e]["per_seed"], base[e]["per_seed"]), ddof=1)
                                                 / math.sqrt(len(runs[e]))) for n in VARIATIONS})
    sig = lambda pts, e: math.hypot(pts[e]["se"], syst[e]["max"]) / pts[e]["value"]

    def fit(pts, es):
        x = [D[e] for e in es]
        f = power_fit(x, [1 / pts[e]["value"] for e in es], [sig(pts, e) for e in es])
        r = power_fit(x, [pts[e]["value"] * R1[e] for e in es], [sig(pts, e) for e in es])
        return f, r

    fits, slopes, per_var = {}, {}, {}
    for k, es in FIT_SETS.items():
        f, r = fit(base, es)
        fits[k] = dict(emittances_nm=es, q=f["b"], sigma_q=f["sb"], a=f["a"], sigma_a=f["sa"], cov_aq=f["cov_ab"],
                       Lambda=math.exp(f["a"]), sigma_Lambda=math.exp(f["a"]) * f["sa"], chi2=f["chi2"], ndf=f["ndf"], p=f["p"],
                       pull=(f["b"] - Q.Q_P) / f["sb"], pull_incl_sigma_q_p=(f["b"] - Q.Q_P) / math.hypot(f["sb"], Q.SQ_P))
        slopes[k] = dict(emittances_nm=es, slope=r["b"], sigma_slope=r["sb"], c=r["a"], sigma_c=r["sa"], cov_cs=r["cov_ab"],
                         chi2=r["chi2"], ndf=r["ndf"], p=r["p"])
        rows = {n: fit(var[n], es)[0] for n in VARIATIONS}
        dq = [rows[n]["b"] - f["b"] for n in VARIATIONS]
        per_var[k] = dict(rows={n: dict(q=v["b"], sigma_q=v["sb"], chi2=v["chi2"], ndf=v["ndf"], shift=v["b"] - f["b"])
                                for n, v in rows.items()},
                          q_min=min(v["b"] for v in rows.values()), q_max=max(v["b"] for v in rows.values()),
                          shift_min=min(dq), shift_max=max(dq), shift_rms=math.sqrt(sum(x * x for x in dq) / len(dq)))
    comp = {}
    for n, pts in {"default": base, **var}.items():
        R = {e: 1 / pts[e]["value"] for e in (0.5, 1)}
        sR = {e: pts[e]["se"] / pts[e]["value"] ** 2 for e in (0.5, 1)}
        comp[n] = dict(R_0p5=R[0.5], se_R_0p5=sR[0.5], R_1nm=R[1], se_R_1nm=sR[1], R1_0p5=R1[0.5], R1_1nm=R1[1],
                       difference=R[0.5] - R[1], se_difference=math.hypot(sR[0.5], sR[1]))
    return dict(runs=runs, D=D, R1=R1, base=base, var=var, syst=syst, fits=fits, slopes=slopes, per_var=per_var, comp=comp)


def block(A):
    """JSON block for paper_numbers.json: every uncertainty with its definition."""
    U = lambda v, d: {"value": float(v), "definition": d}
    rows = []
    for e in Q.EYS:
        b, s = A["base"][e], A["syst"][e]
        rows.append(dict(e_y_nm=e, D_y=A["D"][e], n_seeds=len(A["runs"][e]),
                         default=dict(sigma_min_over_sigma_y_star=b["value"], seed_scatter=U(b["seed_scatter"], SEED_SCATTER),
                                      se=U(b["se"], SE)),
                         variations={n: dict(sigma_min_over_sigma_y_star=A["var"][n][e]["value"], deviation=s["dev"][n],
                                             deviation_paired_se=U(s["paired"][n], PAIRED)) for n in VARIATIONS},
                         systematic_max_deviation=U(s["max"], SYST_MAX), systematic_rms_deviation=U(s["rms"], SYST_RMS),
                         total=U(math.hypot(b["se"], s["max"]), TOTAL),
                         R_1=A["R1"][e], extracted_over_prediction=b["value"] * A["R1"][e],
                         extracted_over_prediction_total=U(math.hypot(b["se"], s["max"]) * A["R1"][e], "the total uncertainty times R_1")))
    E = lambda v, u, d: dict(estimate=float(v), uncertainty=U(u, d))
    SL = FIT.replace("ln R = a + q ln D_y", "the law above")
    fits = {k: dict(emittances_nm=f["emittances_nm"], law="ln(sigma_y*/sigma_min) = a + q ln D_y",
                    q=E(f["q"], f["sigma_q"], FIT), a=E(f["a"], f["sigma_a"], FIT), cov_a_q=f["cov_aq"],
                    Lambda=E(f["Lambda"], f["sigma_Lambda"], "Lambda = exp(a); uncertainty Lambda * sigma_a"),
                    chi2=f["chi2"], ndf=f["ndf"], p_value=f["p"],
                    pull=U(f["pull"], "(q - q_p) / sigma_q"),
                    pull_incl_sigma_q_p=U(f["pull_incl_sigma_q_p"], "(q - q_p) / (sigma_q (+) sigma_q_p)"))
            for k, f in A["fits"].items()}
    slopes = {k: dict(emittances_nm=f["emittances_nm"], law="ln(sigma_min R_1(D_y) / sigma_y*) = c + slope ln D_y",
                      slope=E(f["slope"], f["sigma_slope"], SL), c=E(f["c"], f["sigma_c"], SL), cov_c_slope=f["cov_cs"],
                      chi2=f["chi2"], ndf=f["ndf"], p_value=f["p"]) for k, f in A["slopes"].items()}
    pv = {k: dict(definition="the exponent fit repeated on each variation's own points, sigma_ln = (se (+) systematic) / value "
                             "of that variation; shift = q(variation) - q(default). The fully correlated alternative to the "
                             "systematic in quadrature", **v) for k, v in A["per_var"].items()}
    comp = dict(definition="R = sigma_y*/sigma_min extracted (1 / the 10-seed mean) and the first-waist prediction R_1(D_y); "
                           "se_R = se / value^2; se_difference = se_R(0.5 nm) (+) se_R(1 nm)",
                rows={n: v for n, v in A["comp"].items()})
    return dict(provenance=dict(selection=SELECTION,
                                caches="data/slice_widths_<tag>_<label>.csv and data/slice_fiterr_<tag>_<label>.csv; tag "
                                       "'hw010' for the default settings, 'hw<HW x 100>_thr<threshold x 100>_nb<bins>' otherwise",
                                R_1="data/R1_table.npz, linear interpolation in D_y", q_p=Q.Q_P, sigma_q_p=Q.SQ_P),
                estimator=dict(default=dict(HW_sigma_z=BASELINE[0], fit_threshold=BASELINE[1], n_bins=BASELINE[2]),
                               definition="per beam the minimum over steps of the sigma of a Gaussian fitted to the weighted "
                                          "y-histogram (n_bins bins over +-5 IQR, IQR = interquartile range / 1.349) of the "
                                          "slice |z - z_bar| < HW sigma_z, about its centroid, using the bins above "
                                          "fit_threshold x the peak; the two beams averaged; normalised by sigma_y*",
                               variations={n: dict(HW_sigma_z=s[0], fit_threshold=s[1], n_bins=s[2]) for n, s in VARIATIONS.items()}),
                rows=rows, exponent_fit=fits, per_variation_exponent=pv, ratio_slope=slopes, compression_at_0p5_and_1_nm=comp)


def _build(label):
    import plot_core_width as PCW
    PCW.build_widths(label, [BASELINE, *VARIATIONS.values()])
    return label


if __name__ == "__main__":
    if "--build" in sys.argv:
        from multiprocessing import Pool
        n = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
        labels = [l for v in locus_runs().values() for l in v]
        with Pool(n) as pool:
            for k, l in enumerate(pool.imap_unordered(_build, labels), 1):
                print(f"{k}/{len(labels)} {l}", flush=True)
    A = analysis()
    print(f"{'e_y':>5} {'default':>8} {'se':>7} " + " ".join(f"{n:>14}" for n in VARIATIONS) + f" {'syst max':>8} {'rms':>7}")
    for e in Q.EYS:
        print(f"{e:>5g} {A['base'][e]['value']:8.4f} {A['base'][e]['se']:7.4f} "
              + " ".join(f"{A['var'][n][e]['value']:14.4f}" for n in VARIATIONS)
              + f" {A['syst'][e]['max']:8.4f} {A['syst'][e]['rms']:7.4f}")
    for k, f in A["fits"].items():
        pv = A["per_var"][k]; s = A["slopes"][k]
        print(f"{k}: q = {f['q']:.4f} ± {f['sigma_q']:.4f}, Lambda = {f['Lambda']:.3f} ± {f['sigma_Lambda']:.3f}, "
              f"chi2/ndf = {f['chi2']:.2f}/{f['ndf']}, p = {f['p']:.3f}, pull {f['pull']:+.2f} ({f['pull_incl_sigma_q_p']:+.2f} incl. sigma_q_p); "
              f"per-variation q {pv['q_min']:.4f}-{pv['q_max']:.4f} (rms shift {pv['shift_rms']:.4f}); "
              f"ratio slope {s['slope']:+.4f} ± {s['sigma_slope']:.4f}")
