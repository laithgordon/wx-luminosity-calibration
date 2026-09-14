#!/usr/bin/env python3
"""WX_coarse_grid_L_vs_nm: L vs n_m on a deliberately under-resolved 32x32x32 grid (e_y = 8 nm, 3D IGF solver,
n_m = 1e3..1e7, 15 seeds each; phase PT). Purpose: does the seed STD shrink while L converges to the WRONG value?
Left axis: mean L +/- seed STD with the converged nominal-grid L_inf(8 nm) and its +/-5 % band for reference.
Right axis: relative STD [%]. Paper style, PNG+PDF.
"""
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in

GRID, EY = (32, 32, 32), 8.0


def ladder():
    raw = defaultdict(list)
    for r in Q.read_results("3d"):
        if abs(float(r["e_y_nm"]) - EY) < 1e-9 and (int(r["nx"]), int(r["ny"]), int(r["nz"])) == GRID:
            raw[int(float(r["nm"]))].append((int(r["seed"]), float(r["L"])))
    runs = {k: Q.dedupe_seeds(v) for k, v in raw.items()}
    nm = np.array(sorted(runs), float)
    L = np.array([np.mean(runs[int(v)]) for v in nm]); sd = np.array([np.std(runs[int(v)], ddof=1) if len(runs[int(v)]) > 1 else 0 for v in nm])
    N = np.array([len(runs[int(v)]) for v in nm])
    return nm, L, sd, N


def draw():
    nm, L, sd, N = ladder()
    Linf_nom, _ = Q.plateau_stat(Q.ladders()[EY])          # converged luminosity on the nominal (512,256,128) grid
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.axhspan(0.95 * Linf_nom, 1.05 * Linf_nom, color="C2", alpha=0.13, lw=0, zorder=1)
        ax.axhline(Linf_nom, color="C2", lw=0.9, zorder=2, label=r"$\mathscr{L}_\infty$ nominal grid $\pm5\%$")
        if len(nm):
            ax.errorbar(nm, L, yerr=sd, fmt="o", ls="-", color="C0", ecolor="C0", capsize=1.5, elinewidth=0.6, lw=0.9, ms=3.4,
                        zorder=5, label=r"$32{\times}32{\times}32$, $\varepsilon_y=8$ nm (mean $\pm$ STD)")
            ax2 = ax.twinx()
            ax2.plot(nm, sd / L * 100, "s--", color="C3", ms=3.0, lw=0.8, mfc="white", zorder=4, label=r"STD/$\mathscr{L}$ [%]")
            ax2.set_ylabel(r"$\mathrm{STD}(\mathscr{L})/\mathscr{L}$  [%]", color="C3"); ax2.tick_params(axis="y", colors="C3", direction="in", length=2.5, width=0.5)
            ax2.set_ylim(0, max(1.0, (sd / L * 100).max() * 1.25))
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="center left", frameon=False, fontsize=5.6)
        ax.set_xscale("log"); ax.set_xlabel(r"$n_m$"); ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        _ticks_in(ax)
        save_fig(fig, "WX_coarse_grid_L_vs_nm"); plt.close(fig)
    return nm, L, sd, N, Linf_nom


if __name__ == "__main__":
    nm, L, sd, N, Linf = draw()
    print(f"nominal-grid L_inf(8 nm) = {Linf:.4f}")
    for v, l, s, n in zip(nm, L, sd, N):
        print(f"n_m={v:.0e}: L = {l:.4f} ± {s:.4f} ({s/l*100:.2f} %), {n} seeds, vs L_inf: {(l/Linf-1)*100:+.1f} %")
