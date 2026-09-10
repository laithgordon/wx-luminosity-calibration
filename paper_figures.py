#!/usr/bin/env python3
"""Paper figures in the GP_CALIBRATION.ipynb style (PRL single-column, serif, viridis / C3 / black / gold),
saved to plots/<name>.{png,pdf}:

  WX_L_vs_nm             (1) all Stage-A ladders on one axis: L/L_inf vs n_m per e_y (viridis), mean +/- STD,
                          star at (n_m^req, 0.95), grey +/-5% band, legend above the axes
  WX_nm_req_calibration  (2) n_m^req/(n_x n_y n_z) vs D_y: points (C3, statistical bars), ONE weighted fit
                          C_m D_y^(s-q) on the 20-2 nm points drawn across the whole range (1-sigma band); the
                          1/0.5 nm points faded & labelled, not fitted; conservative locus (gold dashed); GP++ law
  WX_L_vs_ny             (3) all Stage-B ladders: L/L_inf vs n_y per e_y, star at n_y^req
  WX_ny_req_calibration  (4) n_y^req vs D_y: points, weighted fit C_y D_y^q with band, theory q = Q_PRED with the
                          normalisation fitted (dashed C0), chi2/ndf box
  kappa_vs_Dy            (5) kappa_i = n_y^req/(2 c_y R(D_y)) vs D_y with the weighted mean; the same kappa built
                          on Schulte Eq. 2.4 as the failing comparison (climbs monotonically). GP++ series is
                          drawn when Data/gp_ny_req.csv exists.

Fit parameters are also written by write_results.py (section 'Paper figures'). Run after collect.py.
"""
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nmreq as Q

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
NOM_CELLS = 512 * 256 * 128
MID, HIGH = (4, 20), (0.5, 2)          # e_y ranges [nm] of the two D_y regimes
# GUINEA-PIG++ laws: read from data/gp/gp_constants_export.csv when present (the GP++ analysis is the
# authority for them), else the last hardcoded values. GP quotes s = b + q with the same q used here.
_GPC = Q.gp_constants()
GP_CM = _GPC["C_m_fit"][0] if "C_m_fit" in _GPC else 1.4e-07
# GP's exported s_fit was formed with the OLD q = 0.379; rebuild s from the raw slope b with Q_P.
GP_B = _GPC["b_fit"][0] if "b_fit" in _GPC else (_GPC["s_fit"][0] - 0.379 if "s_fit" in _GPC else 3.115)
GP_S = GP_B + Q.Q_P
GP_CY = _GPC["C_y_fit"][0] if "C_y_fit" in _GPC else 26.2
GP_Q = _GPC["q_fit"][0] if "q_fit" in _GPC else 0.402
NY_ANOMALY = (0.5,)                     # e_y whose n_y^req is anomalous: faded, labelled, excluded from the fits
NM_ANOMALY = (0.5, 1)                   # e_y whose n_m^req is anomalous (highest-D_y regime): faded, labelled, excluded from the fit
                                        # (2 nm re-included 2026-08-20: after seeding, its window moved to [1.7e5, 3.1e5] and the point sits 0.3 sigma from the 20-4 nm law)

PRL = {'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
       'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.7, 'axes.grid': True, 'grid.alpha': 0.25,
       'grid.linewidth': 0.4, 'lines.linewidth': 1.0, 'lines.markersize': 3.2, 'text.usetex': False,
       'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'ps.fonttype': 42}


def save_fig(fig, name, dpi=300):
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    print(f"Saved: plots/{name}.pdf, .png")


def _stdform(x):
    if not (x and math.isfinite(x)):
        return ("0", 0)
    e = int(math.floor(math.log10(abs(x))))
    return (f"{x / 10 ** e:.1f}", e)


def _ticks_in(ax):
    ax.tick_params(direction="in", length=2.5, width=0.5, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=1.5, width=0.4, top=True, right=True)


def compact_ladder(name, lad, key, reqs, xlabel, log_base, ticks=None):
    """(1)/(3): all ladders on one axis (absolute L). Three classes of rung:
    plateau rungs (the two top rungs that define L_inf) and bracket rungs (the two straddling 5%) are drawn
    filled with their seed-STD bars; every other rung is drawn transparent without a bar (single seed)."""
    reqd = {e: r for e, D, r in reqs}
    eys = [e for e in Q.EYS if e in lad and e in reqd]
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(eys)))
        for e, c in zip(eys, colors):
            d = lad[e]; r = reqd[e]; Linf = r["Linf"]
            x = np.asarray(d[key], float); y = np.asarray(d["L"], float)
            sd = np.where(np.isfinite(d["std"]), d["std"], 0.0)
            plateau = np.zeros(len(x), bool); plateau[-2:] = True
            brk = (x == r["lo"]) | (x == r["hi"])
            other = ~(plateau | brk)
            ax.plot(x, y, "-", color=c, lw=0.7, alpha=0.6, zorder=3)
            ax.errorbar(x[plateau], y[plateau], yerr=sd[plateau], fmt="o", color=c, ecolor=c, capsize=1.2, elinewidth=0.6,
                        lw=0, ms=3.0, zorder=6)
            ax.errorbar(x[brk], y[brk], yerr=sd[brk], fmt="s", color=c, ecolor=c, capsize=1.5, elinewidth=0.7,
                        lw=0, ms=3.2, markeredgecolor="black", markeredgewidth=0.3, zorder=7)
            ax.plot(x[other], y[other], "o", color=c, ms=2.6, alpha=0.35, lw=0, zorder=4)
            ax.plot([r["nreq"]], [0.95 * Linf], "*", color=c, ms=7, markeredgecolor="black", markeredgewidth=0.35, zorder=10)
        ax.set_xscale("log", base=log_base)
        if ticks:
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        _ticks_in(ax)
        # seed counts for the legend text
        n_pl = sorted({int(n) for e in eys for n in lad[e]["N"][-2:]})
        n_br = sorted({int(reqd[e]["N_lo"]) for e in eys} | {int(reqd[e]["N_hi"]) for e in eys})
        fmt_n = lambda ns: (f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}–{ns[-1]}")
        handles = [plt.Line2D([0], [0], marker="o", color=c, lw=0, markersize=3) for c in colors]
        labels = [fr"$\varepsilon_y={e:g}\,$nm" for e in eys]
        handles += [plt.Line2D([0], [0], marker="o", color="0.35", lw=0, markersize=3),
                    plt.Line2D([0], [0], marker="s", color="0.35", lw=0, markersize=3.2, markeredgecolor="black", markeredgewidth=0.3),
                    plt.Line2D([0], [0], marker="o", color="0.35", lw=0, markersize=2.6, alpha=0.35),
                    plt.Line2D([0], [0], marker="*", color="0.35", lw=0, markersize=6, markeredgecolor="black", markeredgewidth=0.35)]
        labels += [fr"plateau rungs ({fmt_n(n_pl)} seeds, $\pm$STD)", fr"bracket rungs ({fmt_n(n_br)} seeds, $\pm$STD)",
                   "other rungs (1 seed)", r"$n^{\mathrm{req}}$ at $0.95\,\mathscr{L}_\infty$"]
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=5.6,
                  ncol=4, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
        save_fig(fig, name)
        plt.close(fig)


def fit_regime(D, y, sig, mask):
    f = Q.powerlaw_wls(D[mask], y[mask], sig[mask])
    f["s"] = Q.s_from_bpercell(f["b"]); f["C"] = math.exp(f["a"]); f["sC"] = f["C"] * f["sa"]
    return f


def band(ax, f, dx, color="k", alpha=0.13, label=None):
    logdx = np.log(dx)
    var = f["sa"] ** 2 + 2 * logdx * f["cov_ab"] + logdx ** 2 * f["sb"] ** 2
    yc = math.exp(f["a"]) * dx ** f["b"]
    ax.fill_between(dx, yc * np.exp(-np.sqrt(var)), yc * np.exp(np.sqrt(var)), color=color, alpha=alpha, lw=0, zorder=3, label=label)


def WX_nm_req_calibration(A):
    """(2) per-cell requirement vs D_y: one weighted fit on the non-anomalous points (20-4 nm), drawn across the
    whole range; the high-D_y points (NM_ANOMALY) faded, labelled, not fitted; conservative locus; GP++ law."""
    e = np.array([a[0] for a in A]); D = np.array([a[1] for a in A])
    nr = np.array([a[2]["nreq"] for a in A]); cells = np.array([a[2]["cells"] for a in A])
    p16 = np.array([a[2]["p16"] for a in A]); p84 = np.array([a[2]["p84"] for a in A]); sig = np.array([a[2]["sig_log"] for a in A])
    hi = np.array([a[2]["hi"] for a in A]); rho = nr / cells
    use = np.array([ee not in NM_ANOMALY for ee in e])
    out = {}
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.errorbar(D[use], rho[use], yerr=[(nr[use] - p16[use]) / cells[use], (p84[use] - nr[use]) / cells[use]], fmt="o", ms=4.5,
                    color="C3", capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, zorder=5, label="calibration")
        if (~use).any():
            ax.errorbar(D[~use], rho[~use], yerr=[(nr[~use] - p16[~use]) / cells[~use], (p84[~use] - nr[~use]) / cells[~use]], fmt="o",
                        ms=4.5, color="C3", capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, zorder=5, alpha=0.3)
        dx = np.geomspace(D.min() * 0.55, D.max() * 1.9, 80)
        if use.sum() >= 3:
            f = fit_regime(D, rho, sig, use); out["fit"] = f
            m, ex = _stdform(f["C"])
            ax.plot(dx, f["C"] * dx ** f["b"], "-", color="k", lw=1.0, zorder=4,
                    label=fr"fit: $C_m^{{\mathrm{{fit}}}}\,D_y^{{s^{{\mathrm{{fit}}}}-q_p}}$, $C_m^{{\mathrm{{fit}}}}={m}\!\times\!10^{{{ex}}}$, $s^{{\mathrm{{fit}}}}={f['s']:.2f}$")
            band(ax, f, dx, label=r"fit $1\sigma$ band")
            f_allm = fit_regime(D, rho, sig, np.ones_like(use, bool)); out["fit_all"] = f_allm
            ax.plot(dx, f_allm["C"] * dx ** f_allm["b"], "-.", color="0.45", lw=0.9, zorder=4,
                    label=fr"fit, no exclusions: $s^{{\mathrm{{fit}}}}={f_allm['s']:.2f}$")
            # conservative locus: raw fit slope on the fitted points, intercept above every upper bracket edge; per nominal cell
            fc = Q.powerlaw_wls(D[use], nr[use], sig[use]); ac = float(np.max(np.log(hi[use]) - fc["b"] * np.log(D[use])))
            Cc = math.exp(ac) / NOM_CELLS; mc, exc = _stdform(Cc)
            # Conservative locus: a RAW n_m recommendation, so it keeps the RAW slope b_raw.
            # No s is derived from it: s is defined through the per-cell law, s = b_percell + q,
            # and b_raw != b_percell whenever the fit set spans more than one grid.
            ax.plot(dx, Cc * dx ** fc["b"], "--", color="gold", lw=1.4, zorder=4,
                    label=fr"cons: $C_m^{{\mathrm{{cons}}}}\,D_y^{{b^{{\mathrm{{raw}}}}}}$, $C_m^{{\mathrm{{cons}}}}={mc}\!\times\!10^{{{exc}}}$, $b^{{\mathrm{{raw}}}}={fc['b']:.3f}$")
            out["cons"] = dict(C=Cc, C_raw=math.exp(ac), b_raw=fc["b"], sb_raw=fc["sb"],
                               b_percell=f["b"], sb_percell=f["sb"],
                               fit_set=[float(x) for x in e[use]],
                               grids=sorted({float(c) for c in cells[use]}))
            ax.text(0.97, 0.04, fr"$\chi^2/\mathrm{{ndf}} = {f['chi2']:.2f}/{f['ndf']}$", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, linespacing=1.2, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5", boxstyle="round,pad=0.25"))
        mg, eg = _stdform(GP_CM)
        ax.plot(dx, GP_CM * dx ** (GP_S - Q.Q_P), ":", color="0.4", lw=1.1, zorder=4,
                label=fr"GP++ fit: $C_m={mg}\!\times\!10^{{{eg}}}$, $s={GP_S:.2f}$")
        for ee, dd, rr in zip(e, D, rho):
            ax.annotate(fr"${ee:g}\,$nm", (dd, rr), textcoords="offset points", xytext=(5, -3), fontsize=5.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())     # majors only (10^2)
        ax.set_xlabel(r"$D_y(\varepsilon_y)$"); ax.set_ylabel(r"$n_m^{\mathrm{req}}/(n_x n_y n_z)$")
        _ticks_in(ax); ax.legend(loc="upper left", frameon=False, fontsize=6.2)
        save_fig(fig, "WX_nm_req_calibration"); plt.close(fig)
    out["excluded"] = [float(x) for x in e[~use]]
    return out


def WX_ny_req_calibration(B):
    e = np.array([b[0] for b in B]); D = np.array([b[1] for b in B])
    nr = np.array([b[2]["nreq"] for b in B]); p16 = np.array([b[2]["p16"] for b in B]); p84 = np.array([b[2]["p84"] for b in B])
    sig = np.array([b[2]["sig_log"] for b in B])
    use = np.array([ee not in NY_ANOMALY for ee in e])
    f = Q.powerlaw_wls(D[use], nr[use], sig[use]); Cy = math.exp(f["a"])
    f_all = Q.powerlaw_wls(D, nr, sig)                  # same fit with the anomalous point put back
    w = 1 / sig[use] ** 2; a_th = float(np.sum(w * (np.log(nr[use]) - Q.Q_N * np.log(D[use]))) / w.sum())
    chi2_th = float(np.sum(w * (np.log(nr[use]) - a_th - Q.Q_N * np.log(D[use])) ** 2)); Cy_th = math.exp(a_th)
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.errorbar(D[use], nr[use], yerr=[nr[use] - p16[use], p84[use] - nr[use]], fmt="o", ms=4.5, color="C3", capsize=1.5,
                    ecolor="C3", elinewidth=0.7, lw=0, zorder=5, label="calibration")
        if (~use).any():   # anomalous point(s): same marker at low opacity, labelled, not fitted, not in the legend
            ax.errorbar(D[~use], nr[~use], yerr=[nr[~use] - p16[~use], p84[~use] - nr[~use]], fmt="o", ms=4.5, color="C3",
                        capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, zorder=5, alpha=0.3)
        dx = np.geomspace(D.min() * 0.55, D.max() * 1.9, 80)
        m, ex = _stdform(Cy)
        ax.plot(dx, Cy * dx ** f["b"], "-", color="k", lw=1.0, zorder=4,
                label=fr"fit: $C_y^{{\mathrm{{fit}}}}\,D_y^{{q^{{\mathrm{{fit}}}}}}$, $C_y^{{\mathrm{{fit}}}}={m}\!\times\!10^{{{ex}}}$, $q^{{\mathrm{{fit}}}}={f['b']:.2f}$")
        mt, ext = _stdform(Cy_th)
        ax.plot(dx, Cy_th * dx ** Q.Q_N, "--", color="C0", lw=1.0, zorder=4,
                label=fr"theory: $C_y^{{\mathrm{{th}}}}\,D_y^{{q_n}}$, $C_y^{{\mathrm{{th}}}}={mt}\!\times\!10^{{{ext}}}$, $q_n={Q.Q_N:.3f}$")
        band(ax, f, dx, label=r"fit $1\sigma$ band")
        ax.plot(dx, math.exp(f_all["a"]) * dx ** f_all["b"], "-.", color="0.45", lw=0.9, zorder=4,
                label=fr"fit, no exclusions: $q^{{\mathrm{{fit}}}}={f_all['b']:.2f}$")
        mg, eg = _stdform(GP_CY)
        ax.plot(dx, GP_CY * dx ** GP_Q, ":", color="0.4", lw=1.1, zorder=4, label=fr"GP++ fit: $C_y={mg}\!\times\!10^{{{eg}}}$, $q={GP_Q:.2f}$")
        ax.text(0.97, 0.04, fr"$\chi^2/\mathrm{{ndf}} = {f['chi2']:.2f}/{f['ndf']}$", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, linespacing=1.2, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5", boxstyle="round,pad=0.25"))
        for ee, dd, rr in zip(e, D, nr):
            ax.annotate(fr"${ee:g}\,$nm", (dd, rr), textcoords="offset points", xytext=(5, -3), fontsize=5.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())     # majors only (10^2)
        ax.set_ylim(min(p16.min(), 40) * 0.8, max(nr.max(), 200) * 3.2)      # headroom for the legend; long MC tails clipped (values in RESULTS)
        ax.set_xlabel(r"$D_y(\varepsilon_y)$"); ax.set_ylabel(r"$n_y^{\mathrm{req}}$")
        _ticks_in(ax); ax.legend(loc="upper left", frameon=False, fontsize=6.2)
        save_fig(fig, "WX_ny_req_calibration"); plt.close(fig)
    # kappa: the per-point route (S3/S4) only. c_y is the WarpX deck cut, not a shared default.
    kap = Q.kappa_drift(D[use], nr[use], sig[use], c_y=Q.C_Y_WX)
    kap_at20 = Q.kappa_drift(D[use], nr[use], sig[use], c_y=20)   # same data at the GP++ cut, for scale
    return dict(fit=f, C_y=Cy, sC_y=Cy * f["sa"], C_y_th=Cy_th, chi2_th=chi2_th, ndf_th=int(use.sum()) - 1,
                excluded=[float(x) for x in e[~use]], kappa=kap, kappa_at20=kap_at20, fit_all=f_all,
                sig_q=float((f["b"] - Q.Q_N) / f["sb"]), sig_q_all=float((f_all["b"] - Q.Q_N) / f_all["sb"]),
                e=[float(x) for x in e], D=[float(x) for x in D], nr=[float(x) for x in nr],
                sig=[float(x) for x in sig], use=[bool(x) for x in use])


# ---------------------------------------------------------------------------
# S5  kappa vs D_y: the normalisation channel of the eq:ny_law test.
# A flat kappa validates R(D_y); a monotonic climb falsifies the size law used.
# ---------------------------------------------------------------------------
def _gp_kappa():
    """GP++ per-point kappa table (data/gp/gp_kappa_export.csv), already at GP++'s own c_y."""
    return Q.gp_kappa()


def kappa_vs_Dy(B, c_y=None, show_eq24=False):
    """kappa_i = n_y^req/(2 c_y R(D_y)) vs D_y for both codes, with weighted means as
    horizontal lines, plus the Schulte eq. 2.4 series as the failing comparison.

    c_y defaults to the WarpX deck cut (C_Y_WX = 16); the GP++ series uses its own exported c_y.
    There is no shared default -- each code is divided by its own deck's cut.

    show_eq24 (default False) overlays the same kappa rebuilt on Schulte's eq. 2.4 effective size.
    It is OFF for the paper figure: eq. 2.4 is a luminosity-equivalent size averaged over the
    crossing and was never intended to give the transient minimum, so plotting it reads as a
    strawman and it costs the figure most of its vertical dynamic range. The number is still
    computed and quoted in the text (write_results.py) and R_schulte_eq24 remains in nmreq.py."""
    c_y = Q.C_Y_WX if c_y is None else c_y
    e = np.array([b[0] for b in B]); D = np.array([b[1] for b in B])
    nr = np.array([b[2]["nreq"] for b in B]); sig = np.array([b[2]["sig_log"] for b in B])
    use = np.array([ee not in NY_ANOMALY for ee in e])
    kap = Q.kappa_drift(D[use], nr[use], sig[use], c_y=c_y)
    k = np.array(kap["kappa"]); ks = k * sig[use]
    out = {"WX": kap}
    gp = _gp_kappa()
    with plt.rc_context(PRL):
        fig_, ax = plt.subplots(figsize=(3.5, 2.7))
        dx = np.geomspace(D.min() * 0.75, D.max() * 1.35, 60)
        dodge = 1.018                                    # +/-1.8% in D_y, purely cosmetic
        ax.errorbar(D[use] * dodge, k, yerr=ks, fmt="o", ms=4.5, color="C3", capsize=1.5, ecolor="C3",
                    elinewidth=0.7, lw=0, zorder=6, label=fr"WarpX ($c_y={c_y:g}$)")
        ax.axhline(kap["mean"], color="C3", ls="-", lw=1.0, zorder=4,
                   label=fr"WX $\langle\kappa\rangle={kap['mean']:.3f}$, slope $={kap['slope']:+.3f}\pm{kap['sig_slope']:.3f}$")
        if (~use).any():        # anomalous point: shown, not fitted
            ka = np.array([Q.kappa_from_ny(n, d, c_y) for n, d in zip(nr[~use], D[~use])])
            ax.errorbar(D[~use] * dodge, ka, yerr=ka * sig[~use], fmt="o", ms=4.5, color="C3", capsize=1.5,
                        ecolor="C3", elinewidth=0.7, lw=0, alpha=0.3, zorder=5)
        if gp is not None:
            # GP++ carries its own c_y in the export; its kappa is already divided by 2*c_y_GP.
            Dg, kg, kgs, cyg = gp["D_y"], gp["kappa"], gp["sigma_kappa"], float(gp["c_y"][0])
            kg_d = Q.kappa_drift(Dg, gp["n_y_req"], gp["sigma_log"], c_y=cyg)
            out["GP"] = kg_d; out["GP_c_y"] = cyg
            kg = np.array(kg_d["kappa"]); kgs = kg * np.asarray(gp["sigma_log"])   # recomputed on R_1 D^(1/4) (export kappa used the old R)
            ax.errorbar(Dg / dodge, kg, yerr=kgs, fmt="s", ms=4.0, color="C0", capsize=1.5, ecolor="C0",
                        elinewidth=0.7, lw=0, zorder=6, label=fr"GUINEA-PIG++ ($c_y={cyg:g}$)")
            ax.axhline(kg_d["mean"], color="C0", ls="-", lw=1.0, zorder=4,
                       label=fr"GP++ $\langle\kappa\rangle={kg_d['mean']:.3f}$, slope $={kg_d['slope']:+.3f}\pm{kg_d['sig_slope']:.3f}$")
        # Same kappa rebuilt on Schulte eq. 2.4 -- always COMPUTED (quoted in the text), only
        # PLOTTED when show_eq24 is set. See the docstring for why it is off by default.
        k24 = np.array([nr[i] / (2 * c_y * Q.R_schulte_eq24(e[i])) for i in range(len(e)) if use[i]])
        f24 = Q.powerlaw_wls(D[use], k24, sig[use])
        out["eq24"] = dict(kappa=[float(x) for x in k24], slope=f24["b"], sig_slope=f24["sb"],
                           drift=float((D[use].max() / D[use].min()) ** f24["b"] - 1.0))
        if show_eq24:
            ax.plot(D[use], k24, "^", ms=3.6, color="0.45", lw=0, zorder=5, label=r"WX under Schulte Eq. 2.4")
            ax.plot(dx, math.exp(f24["a"]) * dx ** f24["b"], ":", color="0.45", lw=1.1, zorder=3,
                    label=fr"Eq. 2.4 slope $={f24['b']:+.3f}\pm{f24['sb']:.3f}$")
        ax.set_xscale("log")          # y stays LINEAR: kappa spans a narrow range and the test is
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())   # "is the line flat", read better linearly
        ax.set_xlabel(r"$D_y(\varepsilon_y)$")
        ax.set_ylabel(fr"$\kappa = n_y^{{\mathrm{{req}}}}/(2c_y R(D_y))$")
        _ticks_in(ax)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=5.8,
                  ncol=2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
        save_fig(fig_, "kappa_vs_Dy"); plt.close(fig_)
    return out


def WX_GP_comparison(A, B):
    """Inter-code comparison, two panels:
    (a) n_y^req vs D_y for both codes with their free fits;
    (b) pinched-core occupancy n_pinch = (2 c_x c_y/pi)(n_m/(n_x n_y n_z)) R(D_y)  (eq:npinch).

    Panel (b) is deliberately NOT the raw per-cell n_m: that quantity carries the code's own
    cut multipliers through the occupancy prefactor 2 c_x c_y/pi, which differs by
    (20*20)/(16*16) = 1.5625 between the two decks, so raw C_m values are not comparable.
    """
    gk = Q.gp_kappa(); gm = Q.gp_calibration("n_m")
    if gk is None:
        return {}
    e = np.array([b[0] for b in B]); D = np.array([b[1] for b in B])
    nr = np.array([b[2]["nreq"] for b in B]); sig = np.array([b[2]["sig_log"] for b in B])
    use = np.array([ee not in NY_ANOMALY for ee in e])
    em = np.array([a[0] for a in A]); Dm = np.array([a[1] for a in A])
    rho = np.array([a[2]["nreq"] / a[2]["cells"] for a in A]); sigm = np.array([a[2]["sig_log"] for a in A])
    usem = np.array([ee not in NM_ANOMALY for ee in em])
    out = {}
    with plt.rc_context(PRL):
        fig_, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
        # ---- (a) n_y^req -----------------------------------------------------
        ax = axes[0]
        dx = np.geomspace(min(D.min(), gk["D_y"].min()) * 0.75, max(D.max(), gk["D_y"].max()) * 1.35, 60)
        ax.errorbar(D[use], nr[use], yerr=nr[use] * sig[use], fmt="o", ms=4.2, color="C3", capsize=1.5,
                    ecolor="C3", elinewidth=0.7, lw=0, zorder=6, label="WarpX")
        if (~use).any():
            ax.errorbar(D[~use], nr[~use], yerr=nr[~use] * sig[~use], fmt="o", ms=4.2, color="C3",
                        capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, alpha=0.3, zorder=5)
        fw = Q.powerlaw_wls(D[use], nr[use], sig[use])
        ax.plot(dx, math.exp(fw["a"]) * dx ** fw["b"], "-", color="C3", lw=1.0, zorder=4,
                label=fr"WX fit $q={fw['b']:.3f}\pm{fw['sb']:.3f}$")
        ax.errorbar(gk["D_y"], gk["n_y_req"], yerr=gk["n_y_req"] * gk["sigma_log"], fmt="s", ms=3.8,
                    color="C0", capsize=1.5, ecolor="C0", elinewidth=0.7, lw=0, zorder=6, label="GUINEA-PIG++")
        ax.plot(dx, GP_CY * dx ** GP_Q, "-", color="C0", lw=1.0, zorder=4,
                label=fr"GP++ fit $q={GP_Q:.3f}$")
        ax.plot(dx, 2 * Q.C_Y_WX * 0.303 * Q.LAMBDA1 * dx ** Q.Q_N, "--", color="k", lw=1.0, zorder=3,
                label=fr"model $q_n={Q.Q_N:.3f}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$D_y(\varepsilon_y)$"); ax.set_ylabel(r"$n_y^{\mathrm{req}}$")
        ax.set_title("(a) vertical resolution", fontsize=7, pad=3)
        _ticks_in(ax); ax.legend(loc="upper left", frameon=False, fontsize=5.8)
        # ---- (b) pinched-core occupancy -------------------------------------
        ax = axes[1]
        occ_wx = np.array([Q.occupancy_pinch(r, d, Q.C_X_WX, Q.C_Y_WX) for r, d in zip(rho, Dm)])
        ax.errorbar(Dm[usem], occ_wx[usem], yerr=occ_wx[usem] * sigm[usem], fmt="o", ms=4.2, color="C3",
                    capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, zorder=6, label="WarpX")
        if (~usem).any():
            ax.errorbar(Dm[~usem], occ_wx[~usem], yerr=occ_wx[~usem] * sigm[~usem], fmt="o", ms=4.2,
                        color="C3", capsize=1.5, ecolor="C3", elinewidth=0.7, lw=0, alpha=0.3, zorder=5)
        out["occ_wx"] = dict(D=[float(x) for x in Dm], occ=[float(x) for x in occ_wx])
        if gm is not None:
            cxg = cyg = float(gk["c_y"][0])
            occ_gp = np.array([Q.occupancy_pinch(v, d, cxg, cyg) for v, d in zip(gm["value"], gm["D_y"])])
            ax.errorbar(gm["D_y"], occ_gp, yerr=occ_gp * gm["sigma_log"], fmt="s", ms=3.8, color="C0",
                        capsize=1.5, ecolor="C0", elinewidth=0.7, lw=0, zorder=6, label="GUINEA-PIG++")
            out["occ_gp"] = dict(D=[float(x) for x in gm["D_y"]], occ=[float(x) for x in occ_gp])
            fg = Q.powerlaw_wls(gm["D_y"], occ_gp, gm["sigma_log"])
            out["occ_gp_slope"] = (fg["b"], fg["sb"])
        fo = Q.powerlaw_wls(Dm[usem], occ_wx[usem], sigm[usem])
        out["occ_wx_slope"] = (fo["b"], fo["sb"])
        dxm = np.geomspace(Dm.min() * 0.7, Dm.max() * 1.5, 60)
        ax.plot(dxm, math.exp(fo["a"]) * dxm ** fo["b"], "-", color="C3", lw=1.0, zorder=4,
                label=fr"WX slope $={fo['b']:.2f}\pm{fo['sb']:.2f}$")
        if gm is not None:
            ax.plot(dxm, math.exp(fg["a"]) * dxm ** fg["b"], "-", color="C0", lw=1.0, zorder=4,
                    label=fr"GP++ slope $={fg['b']:.2f}\pm{fg['sb']:.2f}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$D_y(\varepsilon_y)$")
        ax.set_ylabel(r"$\bar{n}_{\mathrm{pinch}}$  at $n_m^{\mathrm{req}}$")
        ax.set_title("(b) pinched-core occupancy", fontsize=7, pad=3)
        _ticks_in(ax); ax.legend(loc="upper left", frameon=False, fontsize=5.8)
        fig_.tight_layout()
        save_fig(fig_, "WX_GP_comparison"); plt.close(fig_)
    return out


def make_all(verbose=True, gp_comparison=True):
    A = Q.requirements_stat(); B = Q.ny_requirements_stat()
    lad_m = Q.ladders(); lad_y = Q.ny_ladders()
    for e in lad_y:            # ny_ladders lacks 'nm'/'pooled'; give it the keys compact_ladder needs
        lad_y[e]["ny"] = lad_y[e]["ny"]
    compact_ladder("WX_L_vs_nm", lad_m, "nm", A, r"$n_m$", 10)
    compact_ladder("WX_L_vs_ny", lad_y, "ny", B, r"$n_y$", 2, ticks=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    resm = WX_nm_req_calibration(A) if len(A) >= 3 else {}
    resy = WX_ny_req_calibration(B) if len(B) >= 3 else {}
    resk = kappa_vs_Dy(B) if len(B) >= 3 else {}
    resc = WX_GP_comparison(A, B) if (gp_comparison and len(A) >= 3 and len(B) >= 3) else {}
    if resk:
        resy["kappa_fig"] = resk
    if resc:
        resy["comparison"] = resc
    if verbose:
        if "fit" in resm:
            f = resm["fit"]; print(f"  nm_req fit (excl. {resm['excluded']}): s = {f['s']:.3f} ± {f['sb']:.3f}, C_m = {f['C']:.3e} ± {f['sC']:.2e}, chi2/ndf = {f['chi2']:.2f}/{f['ndf']}")
        if "cons" in resm:
            c = resm["cons"]
            print(f"  cons locus (raw n_m): C_m^cons = {c['C']:.3e} per nominal cell, exponent b_raw = {c['b_raw']:.3f} ± {c['sb_raw']:.3f}")
            print(f"    b_percell = {c['b_percell']:.3f} ± {c['sb_percell']:.3f} (feeds s = b_percell + q_p = {Q.s_from_bpercell(c['b_percell']):.3f}); "
                  f"fit set e_y = {c['fit_set']} nm on {len(c['grids'])} grid(s) {[int(g) for g in c['grids']]}")
        if resy:
            f = resy["fit"]; print(f"  ny_req: q = {f['b']:.3f} ± {f['sb']:.3f}, C_y = {resy['C_y']:.2f}, chi2/ndf = {f['chi2']:.2f}/{f['ndf']}; "
                                   f"theory q_n={Q.Q_N}: C_y = {resy['C_y_th']:.2f}, chi2/ndf = {resy['chi2_th']:.2f}/{resy['ndf_th']}")
            print(f"  significance: (q_fit - q_pred)/sigma_q = {resy['sig_q']:+.2f} sigma "
                  f"[q_fit/0 = {f['b']/f['sb']:.1f} sigma]")
            k = resy["kappa"]
            print(f"  kappa (WarpX deck, c_y = {k['c_y']}): mean = {k['mean']:.3f} ± {k['sig_mean']:.3f}, "
                  f"slope = {k['slope']:+.3f} ± {k['sig_slope']:.3f} ({abs(k['slope'])/k['sig_slope']:.1f} sigma from flat), "
                  f"drift over band = {100*k['drift']:+.1f}%")
            k20 = resy["kappa_at20"]
            print(f"  kappa (same data at c_y = 20, for scale only): mean = {k20['mean']:.3f} ± {k20['sig_mean']:.3f}")
            print(f"  W6 exclusion check: including e_y = {resy['excluded']} nm gives "
                  f"q = {resy['fit_all']['b']:.3f} ± {resy['fit_all']['sb']:.3f} "
                  f"({resy['sig_q_all']:+.2f} sigma from q_n = {Q.Q_N})")
    return resm, resy


if __name__ == "__main__":
    make_all()
