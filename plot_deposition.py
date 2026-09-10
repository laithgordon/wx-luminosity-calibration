#!/usr/bin/env python3
"""WX_deposition_L_vs_ey (paper style, PNG+PDF): the charge-deposition / field-solver comparison at the tuned
WarpX points (512 x NY_CONS x 128, n_m = NM_CONS), seeds 1-3 for the test configurations:

  (a) L vs e_y for  WarpX 3D solver + 3rd-order deposition (the calibration configuration, phase PB/PD seeds),
                    WarpX 2D-slice solver + 3rd-order deposition (phase PS),
                    WarpX 2D-slice solver + 1st-order (CIC) deposition (phase PC2, algo.particle_shape=1),
                    and GP++ tuned (lumi_ee, 15 seeds);
  (b) each WarpX configuration divided by the GP++ luminosity at the same e_y (GP++ has no 0.5 nm point).
Error bars: seed STD in (a); in (b) the two relative seed STDs combined in quadrature.
Phase PC3 (3D solver + CIC) is loaded and tabulated but not drawn: it is identical to PC2 to < 0.1 %.
WarpX cannot set a per-axis shape order, so "1st order" is (1,1,1), against GP++'s (1,1,0)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in
import plot_comparison as PC

STYLE = {   # key: (label, color, marker, ls)
    "wx_3d_s3":  (r"WarpX, 3D solver, 3rd-order deposition", "C0", "s", "-"),
    "wx_2d_s3":  (r"WarpX, 2D-slice solver, 3rd-order deposition", "C2", "^", "--"),
    "wx_2d_cic": (r"WarpX, 2D-slice solver, 1st-order (CIC) deposition", "C1", "D", ":"),
    "gp_cal":    (r"GP++ tuned (CIC, 2D slices)", "C3", "o", "-"),
}


def curves():
    tuned = lambda e, g, nm: g == (512, Q.NY_CONS[e], 128) and abs(nm / Q.NM_CONS[e] - 1) < 0.02
    out = {"wx_3d_s3": PC.wx_curve(tuned, "3d"), "wx_2d_s3": PC.wx_curve(tuned, "2d"),
           "wx_2d_cic": PC.wx_curve(tuned, "2d_cic"), "wx_3d_cic": PC.wx_curve(tuned, "3d_cic"),
           "gp_cal": PC.gp_curve("calibrated")}
    return out


def draw():
    C = curves()
    gp = {x: (l, s) for x, l, s in zip(*C["gp_cal"][:3])}
    with plt.rc_context(PRL):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
        ee = np.array(Q.EYS, float)
        a.plot(ee, [PC.L_geom(x) for x in ee], "--", color="k", lw=1.0, zorder=3, label=r"$\mathscr{L}_{\rm geom}$")
        for key, (lab, col, mk, ls) in STYLE.items():
            e, L, sd, N = C[key]
            if not len(e):
                continue
            a.errorbar(e, L, yerr=sd, fmt=mk, ls=ls, color=col, ecolor=col, mfc="white", capsize=1.5, elinewidth=0.6,
                       lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
            if key != "gp_cal":
                m = np.array([x in gp for x in e])
                r = np.array([L[i] / gp[e[i]][0] for i in range(len(e)) if m[i]])
                re_ = np.array([r[j] * np.hypot(sd[i] / L[i], gp[e[i]][1] / gp[e[i]][0]) for j, i in enumerate(np.where(m)[0])])
                b.errorbar(e[m], r, yerr=re_, fmt=mk, ls=ls, color=col, ecolor=col, mfc="white", capsize=1.5,
                           elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        b.axhline(1.0, color="k", ls="--", lw=0.8, zorder=2)
        for ax, tks in ((a, [v for v in Q.EYS if v not in (0.5, 1)] + [0.5]), (b, [v for v in Q.EYS if v != 0.5])):
            ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(tks)))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in sorted(tks)]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xlabel(r"$\varepsilon_y$ [nm]"); _ticks_in(ax)
        a.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        b.set_ylabel(r"$\mathscr{L}_{\rm WarpX}\,/\,\mathscr{L}_{\rm GP++}$")
        a.legend(loc="upper right", frameon=False, fontsize=5.4); a.set_ylim(0, None)
        b.legend(loc="upper right", frameon=False, fontsize=5.4)
        a.text(0.16, 0.95, "(a)", transform=a.transAxes, va="top"); b.text(0.03, 0.06, "(b)", transform=b.transAxes, va="bottom")
        fig.tight_layout(w_pad=1.5)
        save_fig(fig, "WX_deposition_L_vs_ey"); plt.close(fig)
    return C


def paired_ratio(C, num, den, seeds=(1, 2, 3)):
    """seed-paired ratio num/den per e_y from results.csv rows: {e_y: (mean %, SEM %, n)}"""
    from collections import defaultdict
    def per_seed(solver):
        d = defaultdict(dict)
        for r in Q.read_results(solver):
            e = float(r["e_y_nm"])
            if e in Q.NM_CONS and (int(r["nx"]), int(r["ny"]), int(r["nz"])) == (512, Q.NY_CONS[e], 128) \
                    and abs(float(r["nm"]) / Q.NM_CONS[e] - 1) < 0.02 and int(r["seed"]) in seeds:
                d[e].setdefault(int(r["seed"]), []).append(float(r["L"]))
        return {e: {s: float(np.mean(v)) for s, v in ss.items()} for e, ss in d.items()}
    A, B = per_seed(num), per_seed(den)
    out = {}
    for e in Q.EYS:
        ss = sorted(set(A.get(e, {})) & set(B.get(e, {})))
        if ss:
            v = np.array([A[e][s] / B[e][s] - 1 for s in ss]) * 100
            out[e] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v))
    return out


if __name__ == "__main__":
    C = draw()
    print(f"{'config':>10} | " + " | ".join(f"{v:g}" for v in Q.EYS))
    for key in ("wx_3d_s3", "wx_2d_s3", "wx_2d_cic", "wx_3d_cic", "gp_cal"):
        e, L, sd, N = C[key]; m = {x: (l, n) for x, l, n in zip(e, L, N)}
        print(f"{key:>10} | " + " | ".join(f"{m[v][0]:.3f}({m[v][1]})" if v in m else "—" for v in Q.EYS))
    for num, den in (("2d_cic", "2d"), ("3d_cic", "3d"), ("2d_cic", "3d_cic")):
        pr = paired_ratio(C, num, den)
        print(f"paired {num}/{den} - 1 [%]: " + ", ".join(f"{e:g}: {pr[e][0]:+.2f}±{pr[e][1]:.2f}" for e in Q.EYS if e in pr))
