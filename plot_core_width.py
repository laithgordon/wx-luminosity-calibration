#!/usr/bin/env python3
"""WX_core_width_vs_Dy: minimum CORE width of the central slice vs D_y, against the envelope model 1/R(D_y).

Same runs and slice selection as plot_slice_pinch.py (phase PP dumps, |z - z_bar| < 0.25 sigma_z), with slice half-width HW, but the
slice's vertical size is measured with tail-insensitive estimators instead of the rms:
  - gauss: sigma of a Gaussian fitted to the central peak of the weighted y-histogram (bins above 20% of peak);
  - iqr:   weighted interquartile range / 1.349 (the rms of a Gaussian with that IQR).
Per beam: min over steps; beams averaged; normalised by the nominal waist sigma_y0. The slice rms (from
plot_slice_pinch) and the grid cell size dy/sigma_y0 are drawn for context. Per-run per-step widths are
cached in Data/slice_widths_<label>.csv so re-plots do not re-read the dumps.
"""
import csv, glob, os
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in
from plot_slice_pinch import beam_yzw, SIGMA_Z

HW = 0.10          # slice half-width in sigma_z (converged; matches WX_pinch_evolution)
FIT_THRESHOLD = 0.20   # the Gaussian core fit uses the histogram bins above this fraction of the peak
NBINS = 80         # y-histogram bins over +-5 IQR (bin width 0.125 IQR)

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W
import core_width_syst as CWS
from core_width_syst import cache_tag, BASELINE
assert BASELINE == (HW, FIT_THRESHOLD, NBINS)


def wquantile(x, w, q):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def core_fit(y, z, w, hw=HW, thr=FIT_THRESHOLD, nb=NBINS):
    """Central slice |z - z_bar| < hw sigma_z, about its own centroid: (rms, iqr_width, gauss_width, gauss_err, rel_resid, n).
    gauss_err is the 1-sigma parameter error from the histogram fit covariance (residual-scaled); rel_resid is the rms
    residual of the fit relative to the peak; n the slice particle count. nan where there are fewer than 100 particles
    or fewer than 5 bins above thr, or the fit fails."""
    zc = np.average(z, weights=w)
    m = np.abs(z - zc) < hw * SIGMA_Z
    n = int(m.sum())
    if n < 100:
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    y, w = y[m], w[m]
    yc = np.average(y, weights=w)
    d = y - yc
    rms = float(np.sqrt(np.average(d ** 2, weights=w)))
    iqr = (wquantile(d, w, 0.75) - wquantile(d, w, 0.25)) / 1.349
    # Gaussian fit to the histogram peak
    h, edges = np.histogram(d, bins=nb, range=(-5 * iqr, 5 * iqr), weights=w)
    c = 0.5 * (edges[1:] + edges[:-1])
    sel = h > thr * h.max()
    gw = ge = rr = np.nan
    if sel.sum() >= 5:
        try:
            popt, pcov = curve_fit(lambda x, A, s: A * np.exp(-x * x / (2 * s * s)),
                                   c[sel], h[sel], p0=(h.max(), iqr), maxfev=2000)
            gw = abs(float(popt[1])); ge = float(np.sqrt(pcov[1, 1]))
            resid = h[sel] - popt[0] * np.exp(-c[sel] ** 2 / (2 * popt[1] ** 2))
            rr = float(np.sqrt(np.mean(resid ** 2)) / h.max())
        except Exception:
            pass
    return rms, float(iqr), gw, ge, rr, n


def core_widths(y, z, w, hw=HW, thr=FIT_THRESHOLD, nb=NBINS):
    """(rms, iqr_width, gauss_width) of the central slice, about its own centroid; nan if too few particles."""
    return core_fit(y, z, w, hw, thr, nb)[:3]


def _write_csv(path, header, rows):
    with open(path, "w") as fo:
        fo.write(header + "\n")
        for r in rows:
            fo.write(",".join(str(v) for v in r) + "\n")


def build_widths(label, settings):
    """Read every dump of one run once and write data/slice_widths_<tag>_<label>.csv and
    data/slice_fiterr_<tag>_<label>.csv for each (hw, thr, nb) in settings whose caches are missing.
    The fit-error cache holds, per beam, the fit at that beam's minimum step, as fit_at_min computes it."""
    todo = [s for s in settings if not (ROOT / "data" / f"slice_fiterr_{cache_tag(*s)}_{label}.csv").exists()]
    if not todo:
        return
    per = {s: [] for s in todo}
    fits = {s: {} for s in todo}                       # (step, beam) -> (gauss, err, rel_resid, n)
    allf = sorted(glob.glob(str(W.RUN_ROOT / label / "diags" / "pd" / "*.h5")))
    keep = W.drop_bad_ost(allf)
    if len(keep) < len(allf):
        raise RuntimeError(f"{label}: {len(allf) - len(keep)} dump(s) on a hung OST; not writing partial caches")
    for f in keep:
        with h5py.File(f, "r") as h5:
            for it in h5["data"]:
                rows = {s: [int(it)] for s in todo}
                for b in ("1", "2"):
                    try:
                        y, z, w = beam_yzw(h5, it, f"beam{b}")
                        res = {s: core_fit(y, z, w, *s) for s in todo}
                    except KeyError:
                        res = {s: (np.nan,) * 5 + (0,) for s in todo}
                    for s in todo:
                        rows[s] += list(res[s][:3]); fits[s][(int(it), b)] = res[s][2:]
                for s in todo:
                    per[s].append(rows[s])
    for s in todo:
        rows = sorted(per[s])
        wc = ROOT / "data" / f"slice_widths_{cache_tag(*s)}_{label}.csv"
        if not wc.exists():
            _write_csv(wc, "step,rms1,iqr1,gauss1,rms2,iqr2,gauss2", rows)
        d = np.genfromtxt(wc, delimiter=",", names=True)
        out = []
        for b in ("1", "2"):
            step = int(d["step"][np.nanargmin(d["gauss" + b])])
            gw, ge, rr, n = fits[s][(step, b)]
            out.append((gw, ge, rr, int(n)))
        _write_csv(ROOT / "data" / f"slice_fiterr_{cache_tag(*s)}_{label}.csv", "sigma,err,rel_resid,n", out)


def run_widths(label, hw=HW, thr=FIT_THRESHOLD, nb=NBINS):
    """per-step widths for one PP run, cached: returns dict of arrays rms/iqr/gauss per beam."""
    cache = ROOT / "data" / f"slice_widths_{cache_tag(hw, thr, nb)}_{label}.csv"
    if cache.exists():
        d = np.genfromtxt(cache, delimiter=",", names=True)
        return d
    rows = []
    allf = sorted(glob.glob(str(W.RUN_ROOT / label / "diags" / "pd" / "*.h5")))
    keep = W.drop_bad_ost(allf)
    if len(keep) < len(allf):           # dumps on a hung OST are skipped and listed for a rebuild
        with open(ROOT / "data" / "caches_built_during_ost_outage.txt", "a") as fo:
            skipped = ",".join(Path(f).name for f in allf if f not in keep)
            fo.write(f"{cache.name}: skipped {skipped}\n")
    for f in keep:
        with h5py.File(f, "r") as h5:
            for it in h5["data"]:
                r = [int(it)]
                for sp in ("beam1", "beam2"):
                    try:
                        r += list(core_widths(*beam_yzw(h5, it, sp), hw, thr, nb))
                    except KeyError:
                        r += [np.nan] * 3
                rows.append(r)
    rows.sort()
    _write_csv(cache, "step,rms1,iqr1,gauss1,rms2,iqr2,gauss2", rows)
    return np.genfromtxt(cache, delimiter=",", names=True)


def fit_at_min(label, hw=HW, thr=FIT_THRESHOLD, nb=NBINS):
    """Per beam at its own minimum step: (sigma_gauss, sigma_fit_err, rel_resid, n_slice_particles).
    sigma_fit_err is the 1-sigma parameter error from the histogram fit covariance (residual-scaled);
    rel_resid is the rms residual of the fit relative to the peak. Cached per run."""
    cache = ROOT / "data" / f"slice_fiterr_{cache_tag(hw, thr, nb)}_{label}.csv"
    if cache.exists():
        return np.genfromtxt(cache, delimiter=",", names=True)
    d = run_widths(label, hw, thr, nb)
    rows = []
    for bi, b in enumerate(("1", "2"), 1):
        step = int(d["step"][np.nanargmin(d["gauss" + b])])
        f = W.RUN_ROOT / label / "diags" / "pd" / f"openpmd_{step:06d}.h5"
        with h5py.File(f, "r") as h5:
            y, z, w = beam_yzw(h5, str(step), f"beam{bi}")
        _, _, gw, ge, rr, n = core_fit(y, z, w, hw, thr, nb)
        rows.append((gw, ge, rr, n))
    _write_csv(cache, "sigma,err,rel_resid,n", rows)
    return np.genfromtxt(cache, delimiter=",", names=True)


def R1_table():
    """First-oscillation waist compression R1(D) on a fixed D grid (cached): the earliest local
    minimum of the envelope beta(s), not the global one. Beyond D* ~ 18 the global minimum is
    carried by later oscillations that the real (phase-mixing) beam never reaches."""
    import math
    cache = ROOT / "data" / "R1_table.npz"
    if cache.exists():
        d = np.load(cache)
        return d["D"], d["R1"], d["Rg"], float(d["Dstar"])
    from scipy.integrate import solve_ivp
    bs = Q.BETA_STAR_OVER_SIGZ
    Ds = np.geomspace(15, 160, 160)
    R1s, Rgs, firsts = [], [], []
    for D in Ds:
        amp = (D / math.sqrt(3)) * (2 * math.sqrt(3) / math.sqrt(2 * math.pi))
        sol = solve_ivp(lambda s_, u: [u[1], -amp * math.exp(-2 * s_ * s_) * u[0], u[3], -amp * math.exp(-2 * s_ * s_) * u[2]],
                        (-3, 3), [1, 0, 0, 1], rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.01)
        ss = np.linspace(-3, 3, 8000); u = sol.sol(ss)
        bi = bs + 9 / bs; ai = 3 / bs; gi = (1 + ai * ai) / bi
        beta = bi * u[0] ** 2 - 2 * ai * u[0] * u[2] + gi * u[2] ** 2
        loc = [i for i in range(1, len(ss) - 1) if beta[i] < beta[i - 1] and beta[i] < beta[i + 1]]
        ig = min(loc, key=lambda k: beta[k])
        R1s.append(math.sqrt(bs / beta[loc[0]])); Rgs.append(math.sqrt(bs / beta[ig]))
        firsts.append(ig == loc[0])
    Dstar = float(Ds[np.argmax(~np.array(firsts))])          # first D where osc#1 loses the global minimum
    np.savez(cache, D=Ds, R1=np.array(R1s), Rg=np.array(Rgs), Dstar=Dstar)
    return Ds, np.array(R1s), np.array(Rgs), Dstar


def _dump_on_locus(r):
    """A pinched-core dump run qualifies iff n_y == 4 n_y^cons and n_m/n_m^cons in [0.90, 1.10], both from the frozen
    production locus in nmreq (n_y_cons, n_m_cons)."""
    e = float(r["e_y_nm"])
    # lower bound 0.90: the 12 and 16 nm PQ runs (n_m ratios 0.943, 0.966) are included;
    # the PQ runs at 0.5-8 nm (ratios 0.58-0.88) are excluded
    return int(r["ny"]) == 4 * Q.n_y_cons(e) and 0.90 <= float(r["nm"]) / Q.n_m_cons(e) <= 1.10


def points():
    """{e_y: [per-seed core minima]} from the pinched-core dump runs (phases PQ, PQ2) that sit on the locus rule of
    _dump_on_locus: n_y = 4 n_y^cons and n_m/n_m^cons in [0.90, 1.10]. In practice PQ2 at 0.5-8 nm (n_y 2048/2048/1024/1024/
    1024, n_m on locus) and PQ at 12, 16, 20 nm (n_y 1024)."""
    out = {}
    for r in csv.DictReader(open(W.JOBLIST)):
        if r["phase"] not in ("PQ", "PQ2") or not _dump_on_locus(r):
            continue                                      # n_y = 4 n_y^cons and n_m/n_m^cons in [0.90, 1.10]
        # trust the run directory when collect.py hasn't caught up yet
        if r["status"] != "done":
            st = W.RUN_ROOT / r["label"] / "job_status.txt"
            if W.on_bad_ost(st) or not (st.exists() and "exit=0" in st.read_text()):
                continue
        e = float(r["e_y_nm"])
        if os.environ.get("WX_CACHED_ONLY") and not (ROOT / "data" / f"slice_widths_{cache_tag(*BASELINE)}_{r['label']}.csv").exists():
            continue                                      # incremental refresh: use finished caches only
        d = run_widths(r["label"])
        v = [np.nanmin(d["gauss" + b]) for b in ("1", "2")]
        fe = fit_at_min(r["label"])
        out.setdefault(e, []).append((0.5 * (v[0] + v[1]),
                                      0.5 * float(np.hypot(fe["err"][0], fe["err"][1])),
                                      float(np.mean(fe["rel_resid"])), int(np.min(fe["n"]))))
    return out


def draw():
    """Two stacked panels: (a) core pinch minimum (10-seed mean, n_y = 4 n_y^cons) with two error bars, the seed scatter
    (+) fit error (capped) and the standard error (+) analysis-choice systematic of core_width_syst (thin), the
    first-waist prediction 1/R_1(D_y), and the fitted law exp(-a) D_y^(-q) over all eight emittances with its 1-sigma
    band; (b) extracted / (1/R_1) with the total error and the fitted slope of its logarithm against ln D_y."""
    P = points()
    es = np.array(sorted(P), float); D = np.array([Q.D_y(e) for e in es])
    s0 = np.array([Q.sigma_y(e) for e in es])
    gs = np.array([np.mean([v[0] for v in P[e]]) for e in es]) / s0
    sd = np.array([np.std([v[0] for v in P[e]], ddof=1) if len(P[e]) > 1 else 0.0 for e in es]) / s0
    fer = np.array([np.mean([v[1] for v in P[e]]) for e in es]) / s0     # mean per-seed Gaussian-fit error
    ge = np.hypot(sd, fer)                                              # seed scatter (+) fit error
    rr = np.array([np.mean([v[2] for v in P[e]]) for e in es])
    nsl = np.array([np.min([v[3] for v in P[e]]) for e in es])
    N = np.array([len(P[e]) for e in es])
    th = np.array([1.0 / Q.R1_of_D(d) for d in D])
    fit = None
    if len(es) >= 3:
        sig = ge / gs                                     # absolute error on the ratio -> log error
        f = Q.powerlaw_wls(D, gs, sig)
        wgt = 1.0 / sig ** 2                              # constrained p = -q_p: normalisation only
        a_c = float(np.sum(wgt * (np.log(gs) + Q.Q_P * np.log(D))) / wgt.sum())
        chi2_c = float(np.sum(wgt * (np.log(gs) - a_c + Q.Q_P * np.log(D)) ** 2))
        fit = dict(C=float(np.exp(f["a"])), sC=float(np.exp(f["a"]) * f["sa"]),
                   p=f["b"], sp=f["sb"], chi2=f["chi2"], ndf=f["ndf"],
                   amp=float(np.exp(a_c) * Q.LAMBDA1), chi2_c=chi2_c, ndf_c=len(D) - 1,
                   nsig=float(abs(f["b"] + Q.Q_P) / np.hypot(f["sb"], Q.SQ_P)))
    A = CWS.analysis(dumps=not os.environ.get("WX_CACHED_ONLY"))   # all eight emittances, every estimator setting
    Da = np.array([A["D"][e] for e in Q.EYS]); g = np.array([A["base"][e]["value"] for e in Q.EYS])
    sc = np.array([A["base"][e]["seed_scatter"] for e in Q.EYS])
    tot = np.array([np.hypot(A["base"][e]["se"], A["syst"][e]["max"]) for e in Q.EYS])
    r1 = np.array([A["R1"][e] for e in Q.EYS])
    F, S = A["fits"]["all_8"], A["slopes"]["all_8"]
    with plt.rc_context(PRL):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)   # vertical stack, one journal column, shared x
        dx = np.geomspace(Da.min() / 1.15, Da.max() * 1.15, 200); lx = np.log(dx)
        Dt, R1t, Rgt, Dstar = R1_table()
        ax.plot(dx, np.interp(dx, Dt, 1.0 / R1t), "-", color="0.25", lw=1.0, zorder=4,
                label=fr"first waist: $1/R_1(D_y)$, $R_1\simeq{Q.LAMBDA1}\,D_y^{{{Q.Q_P}}}$")
        yf = np.exp(-(F["a"] + F["q"] * lx))
        sf = np.sqrt(F["sigma_a"] ** 2 + lx ** 2 * F["sigma_q"] ** 2 + 2 * lx * F["cov_aq"])
        ax.fill_between(dx, yf * np.exp(-sf), yf * np.exp(sf), color="C0", alpha=0.18, lw=0, zorder=2)
        ax.plot(dx, yf, "-", color="C0", lw=1.0, zorder=3,
                label=fr"fit $e^{{-a}}D_y^{{-q}}$: $q={F['q']:.3f}\pm{F['sigma_q']:.3f}$, $\chi^2/\mathrm{{ndf}}={F['chi2']:.1f}/{F['ndf']}$")
        ax.errorbar(Da, g, yerr=tot, fmt="none", ecolor="C2", elinewidth=0.5, capsize=0, zorder=5)
        ax.errorbar(Da, g, yerr=sc, fmt="D", color="C2", ecolor="C2", ms=3.6, capsize=1.5,
                    elinewidth=0.6, zorder=6, label="WarpX pinched core (Gaussian fit)")
        for e, d, y in zip(Q.EYS, Da, g):
            ax.annotate(fr"${e:g}\,$nm", (d, y), textcoords="offset points", xytext=(3, 3), fontsize=5.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.25, 0.3, 0.35]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g")); ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylabel(r"$\sigma_y^{\min}/\sigma_y^*$")                       # x label only on the lower (shared) axis
        ax.legend(loc="lower left", frameon=False, fontsize=5.6)
        ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        _ticks_in(ax)
        yr = np.exp(S["c"] + S["slope"] * lx)
        sr = np.sqrt(S["sigma_c"] ** 2 + lx ** 2 * S["sigma_slope"] ** 2 + 2 * lx * S["cov_cs"])
        ax2.fill_between(dx, yr * np.exp(-sr), yr * np.exp(sr), color="C0", alpha=0.18, lw=0, zorder=2)
        ax2.plot(dx, yr, "-", color="C0", lw=1.0, zorder=3,
                 label=fr"fit, slope ${S['slope']:+.3f}\pm{S['sigma_slope']:.3f}$ in $\ln D_y$, $\chi^2/\mathrm{{ndf}}={S['chi2']:.1f}/{S['ndf']}$")
        ax2.errorbar(Da, g * r1, yerr=tot * r1, fmt="D", color="C2", ecolor="C2", ms=3.6, capsize=0,
                     elinewidth=0.6, zorder=5)
        ax2.axhline(1.0, color="0.35", lw=0.8, ls=":")
        ax2.set_xscale("log")
        ax2.set_xlim(dx[0] / 1.08, dx[-1] * 1.08)        # explicit, so the frame does not depend on matplotlib autoscaling
        ax2.xaxis.set_major_locator(mticker.FixedLocator([20, 30, 50, 70, 100, 140]))
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax2.set_xlabel(r"$D_y(\varepsilon_y)$")
        ax2.set_ylabel(r"extracted / $(1/R_1)$")
        ax2.set_ylim(bottom=0.9)
        ax2.legend(loc="lower right", frameon=False, fontsize=5.6)
        ax2.text(0.03, 0.95, "(b)", transform=ax2.transAxes, va="top", fontsize=7)
        _ticks_in(ax2)
        fig.tight_layout()
        save_fig(fig, "WX_core_width_vs_Dy"); plt.close(fig)
    return list(zip(es, D, gs, sd, fer, ge, rr, nsl, th, N)), fit


if __name__ == "__main__":
    rows, fit = draw()
    for e, d, g_, sd_, fe_, ge_, rr_, ns_, th_, n_ in rows:
        print(f"e_y={e:>4g} nm  D_y={d:6.2f}  core={g_:.3f} ± {ge_:.3f} (seed {sd_:.3f}, fit {fe_:.3f})  "
              f"resid={rr_*100:.1f}%  n_slice>={ns_}  theory 1/R={th_:.3f}  ratio={g_/th_:.2f}  ({n_} seeds)")
    if fit:
        print(f"free fit:  sigma/sigma0 = ({fit['C']:.3f} ± {fit['sC']:.3f}) D_y^({fit['p']:.4f} ± {fit['sp']:.4f}),"
              f"  chi2/ndf = {fit['chi2']:.2f}/{fit['ndf']}")
        print(f"theory  :  p = -q_p = -{Q.Q_P} ± {Q.SQ_P}  ->  exponents agree to {fit['nsig']:.1f} sigma")
        print(f"constrained p = -q_p: amplitude x{fit['amp']:.3f} above 1/R_1, chi2/ndf = {fit['chi2_c']:.2f}/{fit['ndf_c']}")
