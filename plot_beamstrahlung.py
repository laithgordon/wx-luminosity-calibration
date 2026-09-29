#!/usr/bin/env python3
"""WX_beamstrahlung_L_vs_ey (paper style, PNG+PDF in plots/) and beamstrahlung_numbers.md.

The recommended (conservative) WarpX configuration run twice, with beamstrahlung on and off:

  (a) L vs e_y for both, error bars = seed STD;
  (b) the same-seed difference (L_off/L_on - 1) in percent, error bar = standard error over the paired seeds.

Panel (b) is the measurement: the effect is a few tenths of a percent while the seed scatter is 1-3 %, so only
the seed-paired difference resolves it. Pairing is by (e_y, seed); seeds without a partner are dropped.

Beamstrahlung ON : data/results.csv, solver 3d, on the frozen production locus (512 x n_y^cons x 128,
                   n_m within 2 % of n_m^cons) -- the 'conservative' series of WX_comparison_L_vs_ey.
Beamstrahlung OFF: data/results_bsoff.csv, the same points with beam1/beam2.do_qed_quantum_sync = 0 and
                   nothing else changed (phase NB). Kept in its own file so no calibration figure or number
                   can pick it up.
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
BSOFF = ROOT / "data" / "results_bsoff.csv"
STYLE = {"on": ("beamstrahlung on", "C0", "s", "--"), "off": ("beamstrahlung off", "C3", "o", "-")}


def per_seed():
    """{'on'|'off': {e_y: {seed: L}}}, same-seed repeat runs averaged."""
    raw = {"on": defaultdict(lambda: defaultdict(list)), "off": defaultdict(lambda: defaultdict(list))}
    for r in Q.read_results("3d"):
        e = float(r["e_y_nm"])
        if e in Q.EYS and (int(r["nx"]), int(r["ny"]), int(r["nz"])) == (512, Q.n_y_cons(e), 128) \
                and abs(float(r["nm"]) / Q.n_m_cons(e) - 1) < 0.02:
            raw["on"][e][int(r["seed"])].append(float(r["L"]))
    for r in csv.DictReader(open(BSOFF)):
        if r["beamstrahlung"] != "off":
            raise ValueError(f"{BSOFF.name} carries a row with beamstrahlung={r['beamstrahlung']!r}")
        raw["off"][float(r["e_y_nm"])][int(r["seed"])].append(float(r["L"]))
    return {k: {e: {s: float(np.mean(v)) for s, v in d.items()} for e, d in raw[k].items()} for k in raw}


def series():
    """[(e_y, mean, std, n)] per setting, and the seed-paired difference [(e_y, pct, se, n_pairs)]."""
    P = per_seed()
    out, paired = {}, []
    for k, d in P.items():
        out[k] = [(e, float(np.mean(list(d[e].values()))),
                   float(np.std(list(d[e].values()), ddof=1)) if len(d[e]) > 1 else 0.0, len(d[e]))
                  for e in sorted(d)]
    for e in sorted(set(P["on"]) & set(P["off"])):
        seeds = sorted(set(P["on"][e]) & set(P["off"][e]))
        if not seeds:
            continue
        v = np.array([P["off"][e][s] / P["on"][e][s] - 1 for s in seeds]) * 100
        paired.append((e, float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, len(seeds)))
    return out, paired


def draw():
    S, paired = series()
    with plt.rc_context(PRL):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
        for key, (lab, col, mk, ls) in STYLE.items():
            e, L, sd, n = (np.array(x) for x in zip(*S[key]))
            ax.errorbar(e, L, yerr=sd, fmt=mk, ls=ls, color=col, ecolor=col, mfc="white", capsize=1.5,
                        elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5, label=lab)
        ax.set_ylabel(r"$\mathscr{L}\ [10^{34}\,\mathrm{cm^{-2}\,s^{-1}}]$")
        ax.legend(loc="upper right", frameon=False, fontsize=6.0)
        ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, va="top", fontsize=7)
        _ticks_in(ax)
        e, p, se, n = (np.array(x) for x in zip(*paired))
        ax2.axhline(0.0, color="k", ls="--", lw=0.8, zorder=2)
        ax2.errorbar(e, p, yerr=se, fmt="o", ls="-", color="C3", ecolor="C3", mfc="white", capsize=1.5,
                     elinewidth=0.6, lw=0.9, ms=3.4, markeredgewidth=0.8, zorder=5)
        w = np.average(p, weights=1 / np.where(se > 0, se, np.inf) ** 2) if np.any(se > 0) else p.mean()
        ax2.axhline(w, color="0.35", lw=0.8, ls=":", zorder=3)
        ax2.text(0.97, 0.06, fr"weighted mean ${w:+.3f}\,\%$", transform=ax2.transAxes, ha="right", fontsize=5.8)
        ax2.set_xlim(0, 21)
        tks = [v for v in Q.EYS if v not in (0.5, 1)] + [0.5]
        ax2.xaxis.set_major_locator(mticker.FixedLocator(sorted(tks)))
        ax2.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in sorted(tks)]))
        ax2.xaxis.set_minor_locator(mticker.NullLocator())
        ax2.set_xlabel(r"$\varepsilon_y$ [nm]")
        ax2.set_ylabel(r"$\mathscr{L}_{\rm off}/\mathscr{L}_{\rm on}-1$  [%]")
        ax2.text(0.03, 0.95, "(b)", transform=ax2.transAxes, va="top", fontsize=7)
        _ticks_in(ax2)
        fig.tight_layout()
        save_fig(fig, "WX_beamstrahlung_L_vs_ey")
        plt.close(fig)
    return S, paired


def table(S, paired):
    on = {e: (L, sd, n) for e, L, sd, n in S["on"]}
    off = {e: (L, sd, n) for e, L, sd, n in S["off"]}
    w = sum(p / se ** 2 for _, p, se, _ in paired if se > 0) / sum(1 / se ** 2 for _, p, se, _ in paired if se > 0)
    lines = [
        "# Beamstrahlung on and off at the recommended configuration",
        "",
        "Generated by `plot_beamstrahlung.py`; do not edit. L in 10³⁴ cm⁻² s⁻¹, *std* = sample standard deviation",
        "over seeds (ddof = 1). The last column is the same-seed difference (L_off/L_on − 1), the only column in",
        "which the effect is resolved: it is a few tenths of a percent while the seed scatter is 1–3 %.",
        "",
        "| ε_y [nm] | on: L ± std (seeds) | off: L ± std (seeds) | off/on | paired Δ ± se [%] (pairs) |",
        "|---|---|---|---|---|",
    ]
    for e, p, se, npair in paired:
        Lon, son, non = on[e]; Loff, soff, noff = off[e]
        lines.append(f"| {e:g} | {Lon:.6f} ± {son:.6f} ({non}) | {Loff:.6f} ± {soff:.6f} ({noff}) | "
                     f"{Loff / Lon:.5f} | {p:+.3f} ± {se:.3f} ({npair}) |")
    lines += ["", f"Weighted mean of the paired difference: **{w:+.3f} %**.", "",
              "Beamstrahlung on: `data/results.csv`, solver 3d, on the frozen production locus. Beamstrahlung off:",
              "`data/results_bsoff.csv`, the same points with `beam1/beam2.do_qed_quantum_sync = 0`.", ""]
    (ROOT / "beamstrahlung_numbers.md").write_text("\n".join(lines))
    print(f"Wrote beamstrahlung_numbers.md (weighted mean paired difference {w:+.3f} %)")
    return w


if __name__ == "__main__":
    S, paired = draw()
    table(S, paired)
    for e, p, se, n in paired:
        print(f"e_y={e:>4g} nm  paired {p:+.3f} ± {se:.3f} % ({n} seeds)")
