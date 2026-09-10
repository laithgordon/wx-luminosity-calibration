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
import csv, glob, os, sys
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

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W


def wquantile(x, w, q):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def core_widths(y, z, w):
    """(rms, iqr_width, gauss_width) of the central slice, about its own centroid; nan if too few particles."""
    zc = np.average(z, weights=w)
    m = np.abs(z - zc) < HW * SIGMA_Z
    if m.sum() < 100:
        return np.nan, np.nan, np.nan
    y, w = y[m], w[m]
    yc = np.average(y, weights=w)
    d = y - yc
    rms = float(np.sqrt(np.average(d ** 2, weights=w)))
    iqr = (wquantile(d, w, 0.75) - wquantile(d, w, 0.25)) / 1.349
    # Gaussian fit to the histogram peak
    h, edges = np.histogram(d, bins=80, range=(-5 * iqr, 5 * iqr), weights=w)
    c = 0.5 * (edges[1:] + edges[:-1])
    sel = h > 0.2 * h.max()
    gw = np.nan
    if sel.sum() >= 5:
        try:
            popt, _ = curve_fit(lambda x, A, s: A * np.exp(-x * x / (2 * s * s)),
                                c[sel], h[sel], p0=(h.max(), iqr), maxfev=2000)
            gw = abs(float(popt[1]))
        except Exception:
            pass
    return rms, float(iqr), gw


def run_widths(label):
    """per-step widths for one PP run, cached: returns dict of arrays rms/iqr/gauss per beam."""
    cache = ROOT / "data" / f"slice_widths_hw{int(HW*100):03d}_{label}.csv"
    if cache.exists():
        d = np.genfromtxt(cache, delimiter=",", names=True)
        return d
    rows = []
    allf = sorted(glob.glob(str(W.RUN_ROOT / label / "diags" / "pd" / "*.h5")))
    keep = W.drop_bad_ost(allf)
    if len(keep) < len(allf):           # dumps stranded on a hung OST: rebuild this cache after the outage
        with open(ROOT / "data" / "caches_built_during_ost_outage.txt", "a") as fo:
            skipped = ",".join(Path(f).name for f in allf if f not in keep)
            fo.write(f"{cache.name}: skipped {skipped}\n")
    for f in keep:
        with h5py.File(f, "r") as h5:
            for it in h5["data"]:
                r = [int(it)]
                for sp in ("beam1", "beam2"):
                    try:
                        r += list(core_widths(*beam_yzw(h5, it, sp)))
                    except KeyError:
                        r += [np.nan] * 3
                rows.append(r)
    rows.sort()
    with open(cache, "w") as fo:
        fo.write("step,rms1,iqr1,gauss1,rms2,iqr2,gauss2\n")
        for r in rows:
            fo.write(",".join(str(v) for v in r) + "\n")
    return np.genfromtxt(cache, delimiter=",", names=True)


def fit_at_min(label):
    """Per beam at its own minimum step: (sigma_gauss, sigma_fit_err, rel_resid, n_slice_particles).
    sigma_fit_err is the 1-sigma parameter error from the histogram fit covariance (residual-scaled);
    rel_resid is the rms residual of the fit relative to the peak. Cached per run."""
    cache = ROOT / "data" / f"slice_fiterr_hw{int(HW*100):03d}_{label}.csv"
    if cache.exists():
        return np.genfromtxt(cache, delimiter=",", names=True)
    d = run_widths(label)
    rows = []
    for bi, b in enumerate(("1", "2"), 1):
        step = int(d["step"][np.nanargmin(d["gauss" + b])])
        f = W.RUN_ROOT / label / "diags" / "pd" / f"openpmd_{step:06d}.h5"
        with h5py.File(f, "r") as h5:
            y, z, w = beam_yzw(h5, str(step), f"beam{bi}")
        zc = np.average(z, weights=w)
        m = np.abs(z - zc) < HW * SIGMA_Z
        dd = y[m] - np.average(y[m], weights=w[m]); ww = w[m]
        iqr = (wquantile(dd, ww, .75) - wquantile(dd, ww, .25)) / 1.349
        h_, e_ = np.histogram(dd, bins=80, range=(-5 * iqr, 5 * iqr), weights=ww)
        c_ = 0.5 * (e_[1:] + e_[:-1]); sel = h_ > 0.2 * h_.max()
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(lambda x, A, sg: A * np.exp(-x * x / (2 * sg * sg)),
                               c_[sel], h_[sel], p0=(h_.max(), iqr), maxfev=2000)
        resid = h_[sel] - popt[0] * np.exp(-c_[sel] ** 2 / (2 * popt[1] ** 2))
        rows.append((abs(float(popt[1])), float(np.sqrt(pcov[1, 1])),
                     float(np.sqrt(np.mean(resid ** 2)) / h_.max()), int(m.sum())))
    with open(cache, "w") as fo:
        fo.write("sigma,err,rel_resid,n\n")
        for r in rows:
            fo.write(",".join(str(v) for v in r) + "\n")
    return np.genfromtxt(cache, delimiter=",", names=True)


def R1_table():
    """First-oscillation waist compression R1(D) on a fixed D grid (cached): the earliest local
    minimum of the envelope beta(s), not the global one. Beyond D* ~ 18 the global minimum is
    carried by later oscillations that the real (phase-mixing) beam never reaches."""
    import math
    from pathlib import Path as _P
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


def points():
    """{e_y: [per-seed core minima]} from the phase-PQ runs (512 x 1024 x 128, n_m = NM_CONS)."""
    out = {}
    for r in csv.DictReader(open(W.JOBLIST)):
        if r["phase"] != "PQ" or int(r["ny"]) != 1024:
            continue
        # trust the run directory when collect.py hasn't caught up yet
        if r["status"] != "done":
            st = W.RUN_ROOT / r["label"] / "job_status.txt"
            if W.on_bad_ost(st) or not (st.exists() and "exit=0" in st.read_text()):
                continue
        e = float(r["e_y_nm"])
        if os.environ.get("WX_CACHED_ONLY") and not (ROOT / "data" / f"slice_widths_hw{int(HW*100):03d}_{r['label']}.csv").exists():
            continue                                      # incremental refresh: use finished caches only
        d = run_widths(r["label"])
        v = [np.nanmin(d["gauss" + b]) for b in ("1", "2")]
        fe = fit_at_min(r["label"])
        out.setdefault(e, []).append((0.5 * (v[0] + v[1]),
                                      0.5 * float(np.hypot(fe["err"][0], fe["err"][1])),
                                      float(np.mean(fe["rel_resid"])), int(np.min(fe["n"]))))
    return out


def draw():
    """Two panels: (a) core pinch minimum (10-seed mean +/- STD, n_y = 1024) and theory 1/R(D_y) vs D_y;
    (b) their ratio. The kink in the theory curve near D_y ~ 72 is real: the envelope's deepest waist
    switches from one betatron oscillation to the next there."""
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
    th = np.array([1.0 / Q.R_of_D(d) for d in D])
    with plt.rc_context(PRL):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
        dx = np.geomspace((D.min() if len(D) else 20) / 1.15, (D.max() if len(D) else 140) * 1.15, 200)
        Dt, R1t, Rgt, Dstar = R1_table()
        ax.plot(dx, np.interp(dx, Dt, 1.0 / R1t), "-", color="0.25", lw=1.0, zorder=4,
                label=fr"theory: $1/R_1(D_y)$, $R_1\simeq{Q.LAMBDA1}\,D_y^{{{Q.Q_P}}}$")
        ax.errorbar(D, gs, yerr=ge, fmt="D", ls="--", color="C2", ecolor="C2", lw=0.9, ms=3.6, capsize=1.5,
                    elinewidth=0.6, zorder=6, label="WarpX pinched core (Gaussian fit)")
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
            # the free power-law fit is computed (quoted in the results file) but no longer drawn (2026-09-10)
        for e, d, y in zip(es, D, gs):
            ax.annotate(fr"${e:g}\,$nm", (d, y), textcoords="offset points", xytext=(3, 3), fontsize=5.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$D_y(\varepsilon_y)$"); ax.set_ylabel(r"$\sigma_{y,\min}/\sigma_{y,0}$")
        ax.legend(loc="lower left", frameon=False, fontsize=5.6)
        ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, va="top", fontsize=7)
        _ticks_in(ax)
        if len(es):
            th1 = np.interp(D, Dt, 1.0 / R1t)
            ax2.errorbar(D, gs / th1, yerr=ge / th1, fmt="D", ls="--", color="C2", ecolor="C2", lw=0.9, ms=3.6,
                         capsize=1.5, elinewidth=0.6, zorder=5)
        ax2.axhline(1.0, color="0.35", lw=0.8, ls=":")
        ax2.set_xscale("log")
        ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax2.set_xlabel(r"$D_y(\varepsilon_y)$")
        ax2.set_ylabel(r"measured / $(1/R_1)$")
        ax2.set_ylim(bottom=0.9)
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
