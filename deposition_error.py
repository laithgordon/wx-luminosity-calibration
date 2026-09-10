#!/usr/bin/env python3
"""Measurement 2: per-pass deposition error correlation (offline, from the PP calibrated-grid dumps).

At each waist of the central slice (local minima of the per-step Gaussian core width), the slice's
charge is deposited along y with WarpX's shape-3 (cubic B-spline) onto (a) the run's own grid and
(b) an 8x refined grid block-averaged back down. delta_i(y) = rho_dep - rho_ref isolates the
deposition error committed at pass i. Output: delta_i(y) overlays, the correlation matrix
C_ij = <delta_i delta_j>/(|delta_i||delta_j|) over core bins (rho_ref > 5% of peak), and the mean
off-diagonal correlation per emittance."""
import glob, sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in
from plot_slice_pinch import beam_yzw, SIGMA_Z
from plot_core_width import run_widths, HW

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W

RUNS = {e: (f"PP_slice_ey{ {0.5:'0p5'}.get(e, f'{e:g}') }_s1", Q.NY_CONS[e]) for e in (20.0, 16.0, 12.0, 8.0, 4.0, 2.0, 1.0, 0.5)}
REFINE = 8


def bspline3(x):
    ax = np.abs(x); w = np.zeros_like(ax)
    m = ax < 1;  w[m] = 2/3 - ax[m]**2 + ax[m]**3/2
    m = (ax >= 1) & (ax < 2); w[m] = (2 - ax[m])**3 / 6
    return w


def deposit(y, w, dy, n, y0):
    """shape-3 deposition of weights w at positions y onto n cells of width dy starting at y0."""
    u = (y - y0) / dy - 0.5                      # position in cell-center units
    rho = np.zeros(n)
    k = np.floor(u).astype(int)
    for off in (-1, 0, 1, 2):
        idx = k + off
        m = (idx >= 0) & (idx < n)
        np.add.at(rho, idx[m], w[m] * bspline3(u[m] - idx[m]))
    return rho / dy


def waist_steps(label):
    """Waists = prominent local minima of the per-step core width (prominence >= 5 % of the
    median width) — the same criterion as the N_osc census; shallow late dips do not count."""
    from scipy.signal import find_peaks
    d = run_widths(label)
    g = 0.5 * (d["gauss1"] + d["gauss2"])
    steps = d["step"].astype(int)
    m = np.isfinite(g)
    pk, _ = find_peaks(-g[m], prominence=0.05 * np.nanmedian(g[m]))
    return [int(steps[m][i]) for i in pk]


def deltas(e):
    label, ny = RUNS[e]
    sy0 = Q.sigma_y(e); dy = 32 * sy0 / ny
    Ly = 32 * sy0; y0 = -Ly / 2
    out = []
    for st in waist_steps(label):
        f = W.RUN_ROOT / label / "diags" / "pd" / f"openpmd_{st:06d}.h5"
        with h5py.File(f, "r") as h5:
            y, z, w = beam_yzw(h5, str(st), "beam1")
        zc = np.average(z, weights=w); m = np.abs(z - zc) < HW * SIGMA_Z
        y, w = y[m], w[m]
        rho = deposit(y, w, dy, ny, y0)
        rho_f = deposit(y, w, dy / REFINE, ny * REFINE, y0)
        rho_ref = rho_f.reshape(ny, REFINE).mean(axis=1)
        out.append((st, rho, rho_ref, rho - rho_ref))
    return out, dy, sy0, ny


def analyse():
    res = {}
    for e in RUNS:
        d, dy, sy0, ny = deltas(e)
        peak = max(r[2].max() for r in d)
        core = np.zeros(ny, bool)
        for _, _, rr, _ in d:
            core |= rr > 0.05 * peak
        V = np.array([dd[core] for _, _, _, dd in d])
        norm = np.sqrt((V ** 2).sum(axis=1))
        C = (V @ V.T) / np.outer(norm, norm)
        off = C[np.triu_indices(len(V), 1)]
        res[e] = dict(steps=[r[0] for r in d], d=d, C=C, off=off, core=core, dy=dy, sy0=sy0, ny=ny,
                      relerr=[float(np.sqrt((dd[core]**2).mean()) / peak) for _, _, _, dd in d])
    return res


def draw(res):
    with plt.rc_context(PRL):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
        for ax, e in zip(axs, sorted(RUNS, reverse=True)):
            r = res[e]; ny, dy, sy0 = r["ny"], r["dy"], r["sy0"]
            yc = (np.arange(ny) + 0.5) * dy - 16 * sy0
            for j, (st, _, rr, dd) in enumerate(r["d"]):
                ax.plot(yc / sy0, dd / rr.max(), lw=0.8, label=f"pass {j+1} (step {st})")
            ax.set_xlim(-4, 4)
            ax.set_xlabel(r"$y/\sigma_{y,0}$"); ax.set_ylabel(r"$\delta\rho/\rho_{\rm ref}^{\rm peak}$")
            ax.legend(frameon=False, fontsize=5.2, loc="upper right",
                      title=fr"$\varepsilon_y={e:g}$ nm ($n_y={ny}$), $\bar C_{{i\neq j}}={np.mean(r['off']):.2f}$",
                      title_fontsize=5.4)
            _ticks_in(ax)
        fig.tight_layout(); save_fig(fig, "WX_depo_error_passes"); plt.close(fig)


def dump_json(res):
    import json
    out = {str(e): dict(ny=r["ny"], npass=len(r["steps"]), mean_offdiag=float(np.mean(r["off"])),
                        rms_first=float(r["relerr"][0]), rms_max=float(max(r["relerr"])))
           for e, r in res.items()}
    (ROOT / "data" / "depo_error.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    res = analyse()
    dump_json(res)
    for e, r in sorted(res.items(), reverse=True):
        print(f"e_y = {e:g} nm (grid n_y = {r['ny']}): waists at steps {r['steps']}")
        print("  C_ij =")
        for row in r["C"]:
            print("   ", " ".join(f"{v:+.2f}" for v in row))
        print(f"  mean off-diagonal correlation: {np.mean(r['off']):+.3f}")
        print(f"  per-pass rms deposition error / peak: {[f'{v:.4f}' for v in r['relerr']]}")
    draw(res)
