#!/usr/bin/env python3
"""WX_extension_L_vs_ey: L vs e_y over 0.5-100 nm for four tuned/frozen series (paper style, PNG+PDF):
  GP++ tuned   -- conservative-formula parameters (data/gp/lumi_nominal_vs_calibrated.csv 'calibrated' for 1-20 nm,
                  data/gp/lumi_tuned_vs_frozen_highemit.csv 'tuned' for 20-100 nm)
  WarpX tuned  -- conservative-formula parameters (NM_CONS/NY_CONS incl. the 40-100 nm extension, phase PE)
  GP++ frozen  -- 40-100 nm at the GP parameters of 20 nm ('frozen' in the same file)
  WarpX frozen -- 40-100 nm at the WarpX 20 nm tuned parameters (512x256x128, n_m = 1e4; phase PF)
15 seeds per point, mean +/- seed STD; L_geom dashed.
"""
import csv
from collections import defaultdict
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
GP_EXT = Q.GP_LUMI_EXT_CSV


def gp_series(path, dataset):
    runs = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["dataset"] == dataset:
            runs[float(r["eps_y_nm"])].append(float(r["lumi_ee"]) / 1e34)
    return {e: (np.mean(v), np.std(v, ddof=1), len(v)) for e, v in runs.items()}


def series():
    C = PC.curves()
    wx = lambda key: {float(e): (l, s, int(n)) for e, l, s, n in zip(*C[key])}
    gp_t = gp_series(PC.GP_CSV, "calibrated"); gp_t.update(gp_series(GP_EXT, "tuned"))     # 1-20 nm + 20-100 nm
    gp_f = gp_series(GP_EXT, "frozen")
    return {"gp_tuned": gp_t, "wx_tuned": wx("wx_cal"), "gp_frozen": gp_f, "wx_frozen": wx("wx_frz20")}


STYLE = {"wx_tuned": (r"WarpX tuned", "C0", "s", "--"), "gp_tuned": (r"GP++ tuned", "C3", "s", "--")}   # frozen series dropped (verification only)


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
    """WX_code_ratio_vs_ey: WarpX tuned / GP++ tuned vs e_y over 1-100 nm (error bar: seed STDs in quadrature)."""
    S = series(); wx, gp = S["wx_tuned"], S["gp_tuned"]
    es = np.array(sorted(e for e in wx if e in gp))
    r = np.array([wx[e][0] / gp[e][0] for e in es])
    re = r * np.sqrt(np.array([(wx[e][1] / wx[e][0]) ** 2 + (gp[e][1] / gp[e][0]) ** 2 for e in es]))
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.axhline(1.0, color="k", ls="--", lw=0.8, zorder=2)
        ax.errorbar(es, r, yerr=re, fmt="s", ls="-", color="C0", ecolor="C0", mfc="white", capsize=1.5, elinewidth=0.6,
                    lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=r"WarpX tuned / GP++ tuned")
        ax.set_xscale("log")                              # 1-100 nm: log axis so every tick label has room
        tks = [1, 2, 4, 8, 12, 20, 40, 100]                # thinned so the rotated labels never touch
        ax.xaxis.set_major_locator(mticker.FixedLocator(tks)); ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in tks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        import matplotlib.pyplot as _plt
        _plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_xlabel(r"$\varepsilon_y$ [nm]"); ax.set_ylabel(r"$\mathscr{L}_{\rm WarpX}/\mathscr{L}_{\rm GP++}$")
        ax.legend(loc="upper right", frameon=False, fontsize=6.0); _ticks_in(ax)
        save_fig(fig, "WX_code_ratio_vs_ey"); plt.close(fig)
    return list(zip(es, r, re))


if __name__ == "__main__":
    S = draw(); draw_code_ratio()
    print(f"{'e_y':>5} | " + " | ".join(f"{k:>22}" for k in STYLE))
    for e in sorted(set(Q.EYS + Q.EYS_EXT)):
        print(f"{e:>5g} | " + " | ".join((f"{S[k][e][0]:.3f} ± {S[k][e][1]:.3f} ({S[k][e][2]})" if e in S[k] else "—").rjust(22) for k in STYLE))


