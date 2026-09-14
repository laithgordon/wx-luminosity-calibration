#!/usr/bin/env python3
"""Comparison figures (paper style, PNG+PDF in plots/):

  WX_comparison_L_vs_ey   L vs e_y for four curves -- WX untuned (512x512x256, n_m = 1e5),
                       WX tuned (512 x n_y_cons(e_y) x 128, n_m = n_m_cons(e_y), the frozen production locus in nmreq),
                       GP untuned (512x512x25, n_m = 1e5) and GP tuned (GP's conservative block) -- plus the
                       geometric luminosity L_geom(e_y) as a black dashed line.
  WX_comparison_HD_vs_ey  H_D = L / L_geom for the same four curves.
  WX_comparison_STD_vs_ey relative seed scatter STD(L)/L [%] for the same four curves.
  WX_comparison_ratio_vs_ey  L_tuned / L_untuned per code (error bar: seed STDs combined in quadrature).

GP data: data/gp_exports/gp_luminosity_for_wx.csv (per-emittance L_mean/L_std(ddof=1)/n_seeds in 1e34 cm^-2 s^-1, 15 seeds
per point; blocks nominal, conservative, frozen_extension).
WX data: data/results.csv rows on the two comparison grids (any row matching the grid+n_m is used; same-seed repeats
averaged). Points are mean +/- seed STD. L in 1e34 cm^-2 s^-1;
L_geom = f_coll N^2 / (4 pi sigma_x sigma_y), f_coll = 133 x 120 Hz.
"""
import csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
GP_CSV = Q.GP_LUMI_CSV
GP_BLOCK = {"nominal": "nominal", "calibrated": "conservative", "tuned": "frozen_extension", "frozen": "frozen_extension"}
F_COLL = 133 * 120.0

# curve definitions: (label, color, marker, ls)
STYLE = {
    "wx_uncal": (r"WarpX untuned", "C0", "o", "-"),
    "wx_cal":   (r"WarpX tuned", "C0", "s", "--"),
    "gp_uncal": (r"GP++ untuned", "C3", "o", "-"),
    "gp_cal":   (r"GP++ tuned", "C3", "s", "--"),
    "wx_cal_2d": (r"WarpX tuned, 2D-slice solver", "C2", "^", "--"),
    "wx_frz20": (r"WarpX, parameters frozen at 20 nm", "C1", "D", ":"),
}


def L_geom(e_y_nm):
    sy = Q.sigma_y(e_y_nm); sx = Q.SIGMA_X
    return F_COLL * Q.N_PART ** 2 / (4 * math.pi * sx * sy * 1e4) / 1e34   # cm^-2 s^-1 / 1e34


ALL_EYS = sorted(Q.EYS + Q.EYS_EXT)          # tuned set extended to 100 nm (phase PE); GP/untuned stop at 20 nm


def wx_curve(match, solver="3d", eys=None):
    raw = defaultdict(list)
    eys = Q.EYS if eys is None else eys
    for r in Q.read_results(solver):
        e = float(r["e_y_nm"])
        if e in eys and match(e, (int(r["nx"]), int(r["ny"]), int(r["nz"])), float(r["nm"])):
            raw[e].append((int(r["seed"]), float(r["L"])))
    runs = {k: Q.dedupe_seeds(v) for k, v in raw.items()}
    es = sorted(runs)
    return (np.array(es), np.array([np.mean(runs[e]) for e in es]),
            np.array([np.std(runs[e], ddof=1) if len(runs[e]) > 1 else 0.0 for e in es]),
            np.array([len(runs[e]) for e in es]))


def gp_curve(dataset, extra=None):
    """extra = (csv_path, dataset) merged in for e_y values the main file lacks (the 40-100 nm extension).
    Reads the per-emittance aggregates of gp_luminosity_for_wx.csv (L_mean_1e34, L_std_1e34 with ddof=1, n_seeds);
    the dataset names map onto its blocks through GP_BLOCK."""
    def _agg(block):
        return {float(r["eps_y_nm"]): (float(r["L_mean_1e34"]), float(r["L_std_1e34"]), int(r["n_seeds"]))
                for r in csv.DictReader(l for l in open(GP_CSV) if not l.startswith("#")) if r["block"] == block}
    agg = _agg(GP_BLOCK[dataset])
    if extra:
        path, ds = extra
        have = set(agg)                          # e_y values already covered by the main block
        agg.update({e: v for e, v in _agg(GP_BLOCK[ds]).items() if e not in have})
    es = sorted(agg)
    return (np.array(es), np.array([agg[e][0] for e in es]),
            np.array([agg[e][1] for e in es]), np.array([agg[e][2] for e in es]))


def curves():
    out = {}
    out["wx_uncal"] = wx_curve(lambda e, g, nm: g == (512, 512, 256) and abs(nm - 1e5) < 1, eys=ALL_EYS)
    out["wx_cal"] = wx_curve(lambda e, g, nm: g == (512, Q.n_y_cons(e), 128) and abs(nm / Q.n_m_cons(e) - 1) < 0.02, eys=ALL_EYS)
    out["gp_uncal"] = gp_curve("nominal")
    out["gp_cal"] = gp_curve("calibrated", extra=(GP_CSV, "tuned"))
    out["wx_cal_2d"] = wx_curve(lambda e, g, nm: g == (512, Q.NY_CONS[e], 128) and abs(nm / Q.NM_CONS[e] - 1) < 0.02, solver="2d")
    # extension batch B (phase PF): the 20 nm tuned grid and n_m held fixed (512x256x128, n_m = 1e4) out to 100 nm;
    # at 20 nm this is the tuned point itself, so the curve starts there
    out["wx_frz20"] = wx_curve(lambda e, g, nm: g == (512, 256, 128) and abs(nm / 1.0e4 - 1) < 0.02, eys=[20] + Q.EYS_EXT)
    return out


def draw(kind):
    C = curves()
    ee = np.array(Q.EYS, float)
    lg = np.array([L_geom(x) for x in ee])
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        if kind == "L":
            ax.plot(ee, lg, "--", color="k", lw=1.0, zorder=3, label=r"$\mathscr{L}_{\rm geom}$")
        elif kind == "HD":
            ax.axhline(1.0, color="k", ls="--", lw=0.8, zorder=2)
        for key, (lab, col, mk, ls) in STYLE.items():
            e, L, sd, N = C[key]
            if kind in ("L", "HD"):                       # the L and H_D panels show the calibrated range only
                if key in ("wx_frz20", "wx_cal_2d"):      # extension-only / solver-test series: shown in their own figures
                    continue
                m = e <= 20; e, L, sd, N = e[m], L[m], sd[m], N[m]
            if len(e) == 0:
                continue
            ref = np.array([L_geom(x) for x in e])
            if kind == "STD":
                y = sd / L * 100; ye = None
            elif kind == "L":
                y, ye = L, sd
            else:
                y, ye = L / ref, sd / ref
            ax.errorbar(e, y, yerr=ye, fmt=mk, ls=ls, color=col, ecolor=col, mfc=(col if "uncal" in key else "white"),
                        capsize=1.5, elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        have_ext = kind == "STD" and len(C["wx_cal"][0]) and C["wx_cal"][0].max() > 20
        tks = [v for v in Q.EYS if v not in (0.5, 1)] + [0.5] + (Q.EYS_EXT if have_ext else [])   # thin the crowded low end
        if kind in ("L", "HD"):
            ax.set_xlim(0, 21)
        ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(tks)))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in sorted(tks)]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(r"$\varepsilon_y$ [nm]")
        if kind == "L":
            ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
            ax.legend(loc="upper right", frameon=False, fontsize=5.4)
            name = "WX_comparison_L_vs_ey"
        elif kind == "HD":
            ax.set_ylabel(r"$H_D = \mathscr{L}/\mathscr{L}_{\rm geom}$")
            ax.legend(loc="upper right", frameon=False, fontsize=5.4)
            name = "WX_comparison_HD_vs_ey"
        else:
            ax.set_ylabel(r"$\mathrm{STD}(\mathscr{L})/\mathscr{L}$  [%]")
            ax.legend(loc="upper right", frameon=False, fontsize=5.4)
            name = "WX_comparison_STD_vs_ey"
        _ticks_in(ax)
        save_fig(fig, name)
        plt.close(fig)
    return C


RATIO_STYLE = {("wx_cal", "wx_uncal"): (r"WarpX tuned / untuned", "C0", "s", "-"),
               ("gp_cal", "gp_uncal"): (r"GP++ tuned / untuned", "C3", "s", "-")}


def draw_ratio():
    C = curves()
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.axhline(1.0, color="k", ls="--", lw=0.8, zorder=2)
        for (knum, kden), (lab, col, mk, ls) in RATIO_STYLE.items():
            en, Ln, sn, _ = C[knum]; ed, Ld, sd_, _ = C[kden]
            md = {x: (l, s_) for x, l, s_ in zip(ed, Ld, sd_)}
            ee = np.array([x for x in en if x in md])
            r = np.array([Ln[list(en).index(x)] / md[x][0] for x in ee])
            re = r * np.sqrt(np.array([(sn[list(en).index(x)] / Ln[list(en).index(x)]) ** 2
                                       + (md[x][1] / md[x][0]) ** 2 for x in ee]))
            ax.errorbar(ee, r, yerr=re, fmt=mk, ls=ls, color=col, ecolor=col, mfc="white",
                        capsize=1.5, elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        emax = max((max(C[k][0]) for k in ("wx_uncal", "wx_cal") if len(C[k][0])), default=20)
        if emax > 20:                                       # extended to 100 nm: log axis, extension tick style
            ax.set_xscale("log")
            tks = [0.5, 1, 2, 4, 8, 12, 20, 40, 100]        # thinned so the rotated labels never touch
        else:
            tks = sorted([v for v in Q.EYS if v not in (0.5, 1)] + [0.5])
        ax.xaxis.set_major_locator(mticker.FixedLocator(tks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in tks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_xlabel(r"$\varepsilon_y$ [nm]")
        ax.set_ylabel(r"$\mathscr{L}_{\rm tuned}/\mathscr{L}_{\rm untuned}$")
        ax.legend(loc="upper right", frameon=False, fontsize=5.4)
        _ticks_in(ax)
        save_fig(fig, "WX_comparison_ratio_vs_ey")
        plt.close(fig)


SOLVER_STYLE = {"wx_cal": (r"WarpX tuned, 3D solver", "C0", "s", "--"),
                "wx_cal_2d": (r"WarpX tuned, 2D-slice solver", "C2", "^", "--"),
                "gp_cal": (r"GP++ tuned", "C3", "s", "--")}


def draw_solver():
    """WX_solver_L_vs_ey: L vs e_y for the tuned WarpX set with the 3D and the 2D-slice field solver, and tuned GP++."""
    C = curves()
    ee = np.array(Q.EYS, float)
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.plot(ee, [L_geom(x) for x in ee], "--", color="k", lw=1.0, zorder=3, label=r"$\mathscr{L}_{\rm geom}$")
        for key, (lab, col, mk, ls) in SOLVER_STYLE.items():
            e, L, sd, N = C[key]
            if len(e):
                ax.errorbar(e, L, yerr=sd, fmt=mk, ls=ls, color=col, ecolor=col, mfc="white", capsize=1.5, elinewidth=0.6,
                            lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        tks = [v for v in Q.EYS if v not in (0.5, 1)] + [0.5]
        ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(tks)))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in sorted(tks)]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(r"$\varepsilon_y$ [nm]"); ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        ax.legend(loc="upper right", frameon=False, fontsize=5.6)
        _ticks_in(ax)
        save_fig(fig, "WX_solver_L_vs_ey"); plt.close(fig)
    return C


if __name__ == "__main__":
    C = draw("L"); draw("HD"); draw("STD"); draw_ratio(); draw_solver()
    print(f"{'curve':>9} | " + " | ".join(f"{v:g}" for v in Q.EYS))
    for key in STYLE:
        e, L, sd, N = C[key]
        m = {x: (l, n) for x, l, n in zip(e, L, N)}
        print(f"{key:>9} | " + " | ".join(f"{m[v][0]:.3f}({m[v][1]})" if v in m else "—" for v in Q.EYS))
    print("L_geom    | " + " | ".join(f"{L_geom(v):.3f}" for v in Q.EYS))
