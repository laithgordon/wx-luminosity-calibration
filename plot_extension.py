#!/usr/bin/env python3
"""WX_extension_L_vs_ey: L vs e_y over 0.5-100 nm for four tuned/frozen series (paper style, PNG+PDF):
  GP++ tuned   -- data/gp_exports/gp_luminosity_for_wx.csv: 'conservative' block for 1-20 nm, 'frozen_extension' block
                  for 40-100 nm (the only GP data there)
  WarpX tuned  -- the frozen production locus (plot_comparison 'wx_cal'): phase PN at 0.5-16 nm, PC/PD at 20 nm,
                  and the extrapolated extension locus (phase PE) at 40-100 nm
  GP++ frozen  -- 40-100 nm at the GP parameters of 20 nm ('frozen_extension' block)
  WarpX frozen -- 40-100 nm at the WarpX 20 nm tuned parameters (512x256x128, n_m = 1e4; phase PF)
15 seeds per point, mean +/- seed STD; L_geom dashed.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nmreq as Q
import plot_comparison as PC
from paper_figures import PRL, save_fig, _ticks_in

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
GP_EXT = PC.GP_CSV


def gp_series(path, dataset):
    """{e_y: (L_mean, L_std ddof=1, n_seeds)} for one block of gp_luminosity_for_wx.csv (dataset names via PC.GP_BLOCK)."""
    return {float(r["eps_y_nm"]): (float(r["L_mean_1e34"]), float(r["L_std_1e34"]), int(r["n_seeds"]))
            for r in csv.DictReader(l for l in open(path) if not l.startswith("#")) if r["block"] == PC.GP_BLOCK[dataset]}


def series():
    C = PC.curves()
    wx = lambda key: {float(e): (l, s, int(n)) for e, l, s, n in zip(*C[key])}
    gp_t = gp_series(PC.GP_CSV, "calibrated"); gp_t.update(gp_series(GP_EXT, "tuned"))     # 1-20 nm + 20-100 nm
    gp_f = gp_series(GP_EXT, "frozen")
    return {"gp_tuned": gp_t, "wx_tuned": wx("wx_cal"), "gp_frozen": gp_f, "wx_frozen": wx("wx_frz20")}


STYLE = {"wx_tuned": (r"WarpX tuned", "C0", "s", "--"), "gp_tuned": (r"GP++ tuned", "C3", "s", "--")}   # frozen series not drawn


def draw():
    S = series(); eys = sorted(set(Q.EYS + Q.EYS_EXT))
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.plot(eys, [PC.L_geom(x) for x in eys], "--", color="k", lw=1.0, zorder=3, label=r"$\mathscr{L}_{\rm geom}$")
        for key, (lab, col, mk, ls) in STYLE.items():
            d = S[key]; es = np.array(sorted(d))
            if len(es) == 0:
                continue
            ax.errorbar(es, [d[e][0] for e in es], yerr=[d[e][1] for e in es], fmt=mk, ls=ls, color=col, ecolor=col, mfc="white",
                        capsize=1.5, elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        tks = [2, 4, 8, 12, 16, 20, 40, 60, 80, 100]
        ax.xaxis.set_major_locator(mticker.FixedLocator(tks)); ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in tks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(r"$\varepsilon_y$ [nm]"); ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        ax.legend(loc="upper right", frameon=False, fontsize=5.6); _ticks_in(ax)
        save_fig(fig, "WX_extension_L_vs_ey"); plt.close(fig)
    return S


def draw_code_ratio():
    """WX_code_ratio_vs_ey: WarpX / GP++ mean luminosity vs e_y, as two separately drawn comparisons
    (error bar: standard error of the ratio of means):
      1-20 nm   -- each simulator at its recommended configuration (WarpX tuned / GP++ tuned)
      40-100 nm -- both simulators frozen at their recommended 20 nm configurations (WarpX frozen / GP++ frozen)."""
    S = series()

    def ratio(a, b, emax=None, emin=None):
        es = np.array(sorted(e for e in a if e in b and (emax is None or e <= emax) and (emin is None or e >= emin)))
        r = np.array([a[e][0] / b[e][0] for e in es])
        # standard error of the ratio of means: each relative seed scatter divided by sqrt(n_seeds), in quadrature
        re = r * np.sqrt(np.array([(a[e][1] / a[e][0]) ** 2 / a[e][2] + (b[e][1] / b[e][0]) ** 2 / b[e][2] for e in es]))
        return es, r, re

    rec = ratio(S["wx_tuned"], S["gp_tuned"], emax=20)
    frz = ratio(S["wx_frozen"], S["gp_frozen"], emin=40)
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.axhline(1.0, color="k", ls="--", lw=0.8, zorder=2)
        ax.errorbar(*rec[:2], yerr=rec[2], fmt="s", ls="-", color="C0", ecolor="C0", mfc="white", capsize=1.5, elinewidth=0.6,
                    lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=r"recommended configurations")
        ax.errorbar(*frz[:2], yerr=frz[2], fmt="D", ls=":", color="C1", ecolor="C1", mfc="white", capsize=1.5, elinewidth=0.6,
                    lw=0.9, ms=3.2, markeredgewidth=0.8, zorder=5, label=r"both frozen at 20 nm configurations")
        ax.set_xscale("log")                              # 1-100 nm: log axis so every tick label has room
        tks = [1, 2, 4, 8, 12, 20, 40, 100]                # thinned so the rotated labels never touch
        ax.xaxis.set_major_locator(mticker.FixedLocator(tks)); ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in tks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_xlabel(r"$\varepsilon_y$ [nm]"); ax.set_ylabel(r"$\mathscr{L}_{\rm WarpX}/\mathscr{L}_{\rm GP++}$")
        ax.legend(loc="upper right", frameon=False, fontsize=6.0); _ticks_in(ax)
        save_fig(fig, "WX_code_ratio_vs_ey"); plt.close(fig)
    return [tuple(zip(*rec)), tuple(zip(*frz))]


if __name__ == "__main__":
    S = draw(); draw_code_ratio()
    print(f"{'e_y':>5} | " + " | ".join(f"{k:>22}" for k in STYLE))
    for e in sorted(set(Q.EYS + Q.EYS_EXT)):
        print(f"{e:>5g} | " + " | ".join((f"{S[k][e][0]:.3f} ± {S[k][e][1]:.3f} ({S[k][e][2]})" if e in S[k] else "—").rjust(22) for k in STYLE))


