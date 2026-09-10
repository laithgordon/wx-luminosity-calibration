#!/usr/bin/env python3
"""WX_slice_pinch_vs_Dy: central-slice minimum beam size vs D_y against the envelope-model 1/R(D_y).

Data: phase PP runs (tuned 3D parameters, 1 seed per e_y) with a per-step openPMD particle dump of the
two beams (diag `pd`). Per step and beam: select the central slice |z - z_centroid| < SLICE_HW * sigma_z,
compute the weighted rms of y about the slice's own centroid (so slice-centroid offsets and the unpinched
head/tail do not dilute it), take the minimum over steps, average the two beams. Overlay: the whole-beam
minimum from the BeamRelevant runs (phase PD, faded) and the theory curve 1/R(D_y).
"""
import csv, glob, sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W

SLICE_HW = 0.25          # half-width of the central slice in units of sigma_z
SIGMA_Z = 100e-6         # m, deck value


def _comp(g, name):
    """openPMD record component with unitSI applied; constant components (groups carrying a
    `value` attribute, e.g. WarpX's positionOffset) come back as a scalar that broadcasts."""
    r = g[name]
    if hasattr(r, "keys"):                                  # constant component
        return float(r.attrs["value"]) * float(r.attrs.get("unitSI", 1.0))
    return r[...] * float(r.attrs.get("unitSI", 1.0))


def beam_yzw(h5, it, sp):
    p = h5[f"data/{it}/particles/{sp}"]
    y = _comp(p, "position/y") + _comp(p, "positionOffset/y")
    z = _comp(p, "position/z") + _comp(p, "positionOffset/z")
    w = p["weighting"][...] if "weighting" in p else np.ones_like(y)
    return y, z, w


def slice_sigma_y(y, z, w):
    zc = np.average(z, weights=w)
    m = np.abs(z - zc) < SLICE_HW * SIGMA_Z
    if m.sum() < 50:
        return np.nan
    yc = np.average(y[m], weights=w[m])
    return float(np.sqrt(np.average((y[m] - yc) ** 2, weights=w[m])))


def run_min(run_dir):
    """(sig_y_min_slice averaged over beams, n_steps) for one PP run."""
    mins = {"beam1": np.inf, "beam2": np.inf}
    allf = sorted(glob.glob(str(run_dir / "diags" / "pd" / "*.h5")))
    files = W.drop_bad_ost(allf)
    if len(files) < len(allf):
        print(f"note: {run_dir.name}: {len(allf)-len(files)} dump(s) on a hung OST skipped", flush=True)
    n = 0
    for f in files:
        with h5py.File(f, "r") as h5:
            for it in h5["data"]:
                n += 1
                for sp in mins:
                    try:
                        s = slice_sigma_y(*beam_yzw(h5, it, sp))
                    except KeyError:
                        s = np.nan
                    if np.isfinite(s):
                        mins[sp] = min(mins[sp], s)
    if not all(np.isfinite(v) for v in mins.values()):
        return None, n
    return 0.5 * (mins["beam1"] + mins["beam2"]), n


def slice_points():
    out = {}
    for r in csv.DictReader(open(W.JOBLIST)):
        if r["phase"] != "PP" or r["status"] != "done":
            continue
        e = float(r["e_y_nm"])
        v, n = run_min(W.RUN_ROOT / r["label"])
        if v is not None:
            out[e] = (v, n)
    return out


def wholebeam_points():
    import plot_sigmin as PSm
    R = PSm.runs()
    return {e: (np.mean([v[0] for v in R[e].values()]),
                np.std([v[0] for v in R[e].values()], ddof=1) if len(R[e]) > 1 else 0.0) for e in R}


def draw():
    S = slice_points(); B = wholebeam_points()
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        Dall = [Q.D_y(e) for e in set(S) | set(B)] or [20, 130]
        dx = np.geomspace(min(Dall) / 1.15, max(Dall) * 1.15, 120)
        ax.plot(dx, [1.0 / Q.R_of_D(d) for d in dx], "-", color="0.35", lw=1.0, zorder=3,
                label=fr"theory: $1/R(D_y)$ ($R\simeq{Q.LAMBDA}\,D_y^{{{Q.Q_PRED}}}$)")
        if B:
            eb = np.array(sorted(B), float); Db = np.array([Q.D_y(e) for e in eb])
            rb = np.array([B[e][0] / Q.sigma_y(e) for e in eb]); sb = np.array([B[e][1] / Q.sigma_y(e) for e in eb])
            ax.errorbar(Db, rb, yerr=sb, fmt="s", ls="--", color="C0", ecolor="C0", mfc="white", capsize=1.5,
                        elinewidth=0.6, lw=0.8, ms=3.0, markeredgewidth=0.7, alpha=0.45, zorder=4,
                        label=r"whole beam: $\sigma_{y,\min}/\sigma_{y,0}$")
        rows = []
        if S:
            es = np.array(sorted(S), float); Ds = np.array([Q.D_y(e) for e in es])
            rs = np.array([S[e][0] / Q.sigma_y(e) for e in es])
            ax.plot(Ds, rs, "o", ls="--", color="C3", lw=0.9, ms=3.6, zorder=6,
                    label=fr"central slice ($|z-\bar z|<{SLICE_HW:g}\,\sigma_z$): $\sigma_{{y,\min}}^{{\rm slice}}/\sigma_{{y,0}}$")
            for e, d, r in zip(es, Ds, rs):
                ax.annotate(fr"${e:g}\,$nm", (d, r), textcoords="offset points", xytext=(5, -3), fontsize=5.6)
            rows = [(e, d, S[e][0] * 1e9, r, 1.0 / Q.R_of_D(d), S[e][1]) for e, d, r in zip(es, Ds, rs)]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$D_y(\varepsilon_y)$"); ax.set_ylabel(r"$\sigma_{y,\min}/\sigma_{y,0}$")
        ax.legend(loc="best", frameon=False, fontsize=5.6); _ticks_in(ax)
        save_fig(fig, "WX_slice_pinch_vs_Dy"); plt.close(fig)
    return rows


if __name__ == "__main__":
    for e, d, sm, r, th, n in draw():
        print(f"e_y={e:>4g} nm  D_y={d:6.2f}  sigma_y,min^slice={sm:.3f} nm  ratio={r:.3f}  theory 1/R={th:.3f}  ({n} dumps)")
