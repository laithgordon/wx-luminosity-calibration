#!/usr/bin/env python3
"""WX_depo_correlation: deposition-error decoherence vs e_y (paper style, one column).

Two stacked panels, shared x = e_y (log, DESCENDING 20 -> 0.5 so disruption grows rightward).
(a) mean pairwise pass correlation C_bar (10-seed mean ± SEM) with reference lines at 0
    ("independent passes") and 1 ("identical passes"); N_osc annotated along the top axis;
    the excluded emittances (1, 0.5 nm) shaded.
(b) n_m^cons actually run at each point (log y): the confound shown, not hidden — the rise of
    C_bar toward 2 nm tracks n_m (shot noise diluting), the BREAK at 1-0.5 nm does not.
Data: Data/depo_corr_stat.json (depo_corr_stat.py: PQ dumps, 10 seeds, deposited on the
calibrated n_y = NY_CONS grid vs 8x refined reference; waists = prominent core minima, 5 %)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from paper_figures import PRL, save_fig, _ticks_in

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)


def draw():
    D = json.loads((ROOT / "data" / "depo_corr_stat.json").read_text())
    es = sorted((float(k) for k in D), reverse=True)         # 20 ... 0.5
    g = lambda k: np.array([D[f"{e:g}"][k] for e in es])
    cb, sem, N, nm = g("cbar"), g("sem"), g("N"), g("nm")
    with plt.rc_context(PRL):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.08))
        for a in (ax, ax2):
            a.axvspan(1.45, 0.36, color="C3", alpha=0.08, lw=0)
        ax.axhline(0.0, color="0.4", lw=0.7, ls=":")
        ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
        ax.text(21.5, 0.02, "independent passes", fontsize=5.2, color="0.4", va="bottom")
        ax.text(0.62, 1.02, "identical passes", fontsize=5.2, color="0.4", va="bottom", ha="left")
        ax.errorbar(es, cb, yerr=sem, fmt="o", color="C0", ecolor="C0", ms=3.8, capsize=1.5,
                    elinewidth=0.7, lw=0.9, ls="-", zorder=5)
        ax.text(0.03, 0.62, r"10 seeds, $\pm$ SEM", transform=ax.transAxes, fontsize=5.2, color="C0")
        ax.text(0.72, 0.32, "excluded", transform=ax.transAxes, fontsize=5.4, color="C3", alpha=0.8)
        ax.set_ylim(-0.22, 1.12)
        ax.set_ylabel(r"$\bar C_{i\neq j}$ (pass correlation)")
        # N_osc along the top
        for e, n in zip(es, N):
            ax.annotate(f"{n:.1f}", (e, 1.0), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=5.0, color="0.35", annotation_clip=False)
        ax.annotate(r"$N_{\rm osc}$:", (es[0] * 1.22, 1.0), xytext=(0, 10), textcoords="offset points",
                    ha="right", fontsize=5.0, color="0.35", annotation_clip=False)
        _ticks_in(ax)

        ax2.plot(es, nm, "s-", color="0.35", ms=3.2, lw=0.9, mfc="white", zorder=5)
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$n_m^{\rm cons}$")
        ax2.set_xscale("log")
        ax2.set_xlim(25, 0.4)                                # descending: disruption grows rightward
        ax2.xaxis.set_major_locator(mticker.FixedLocator(es))
        ax2.xaxis.set_major_formatter(mticker.FixedFormatter([f"{e:g}" for e in es]))
        ax2.xaxis.set_minor_locator(mticker.NullLocator())
        ax2.set_xlabel(r"$\varepsilon_y$ [nm]  ($D_y$ increases $\rightarrow$)")
        _ticks_in(ax2)
        save_fig(fig, "WX_depo_correlation"); plt.close(fig)
    return {e: D[f"{e:g}"] for e in es}


if __name__ == "__main__":
    for e, d in draw().items():
        print(f"e_y={e:>4g}: C̄={d['cbar']:+.3f}±{d['sem']:.3f} N={d['N']:.1f}±{d['sN']:.1f} "
              f"n_m={d['nm']:.1e} adj_last={d['adj_last']:+.2f}±{d['adj_last_sem']:.2f} ({d['nseeds']} seeds)")
