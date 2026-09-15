#!/usr/bin/env python3
"""WX_pinch_evolution: vertical beam/slice width vs time through the collision (e_y = 8 nm, tuned 3D run).

From the phase-PP per-step dumps of beam1: whole-beam rms; the central slice (|z - z_bar| < 0.1 sigma_z)
as rms and as Gaussian-core width; and off-centre slices (z_bar +/- 0.5, +/- 1.0 sigma_z, rms) showing that
each slice pinches at its own time. Widths normalised by sigma_y0; time as c*t/sigma_z (shifted so the
central slice's minimum is at 0). Horizontal line: envelope-model 1/R(D_y)."""
import glob
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in
from plot_slice_pinch import beam_yzw, SIGMA_Z
from plot_core_width import wquantile, core_widths

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W

EY, HW = 8.0, 0.10          # nm; slice half-width in sigma_z (converged, see methods)
OFFS = (-1.0, -0.5, 0.5, 1.0)


def series(label):
    t, wb, s_rms, s_core, offs = [], [], [], [], {o: [] for o in OFFS}
    for f in sorted(glob.glob(str(W.RUN_ROOT / label / "diags" / "pd" / "*.h5"))):
        with h5py.File(f, "r") as h5:
            for it in h5["data"]:
                t.append(float(h5[f"data/{it}"].attrs["time"]))
                y, z, w = beam_yzw(h5, it, "beam1")
                yc = np.average(y, weights=w)
                wb.append(float(np.sqrt(np.average((y - yc) ** 2, weights=w))))
                zc = np.average(z, weights=w)
                for o in (0.0,) + OFFS:
                    m = np.abs(z - (zc + o * SIGMA_Z)) < HW * SIGMA_Z
                    if m.sum() < 100:
                        v = (np.nan, np.nan, np.nan)
                    else:
                        v = core_widths(np.where(m, y, np.nan)[m] * 0 + y[m], z[m] * 0 + zc + o * SIGMA_Z, w[m]) \
                            if False else None
                    if o == 0.0:
                        if m.sum() < 100:
                            s_rms.append(np.nan); s_core.append(np.nan); continue
                        d = y[m] - np.average(y[m], weights=w[m]); ww = w[m]
                        s_rms.append(float(np.sqrt(np.average(d ** 2, weights=ww))))
                        iqr = (wquantile(d, ww, .75) - wquantile(d, ww, .25)) / 1.349
                        h_, e_ = np.histogram(d, bins=80, range=(-5 * iqr, 5 * iqr), weights=ww)
                        c_ = 0.5 * (e_[1:] + e_[:-1]); sel = h_ > 0.2 * h_.max()
                        try:
                            from scipy.optimize import curve_fit
                            popt, _ = curve_fit(lambda x, A, s: A * np.exp(-x * x / (2 * s * s)),
                                                c_[sel], h_[sel], p0=(h_.max(), iqr), maxfev=2000)
                            s_core.append(abs(float(popt[1])))
                        except Exception:
                            s_core.append(np.nan)
                    else:
                        if m.sum() < 100:
                            offs[o].append(np.nan); continue
                        d = y[m] - np.average(y[m], weights=w[m])
                        offs[o].append(float(np.sqrt(np.average(d ** 2, weights=w[m]))))
    return (np.array(t),) + tuple(np.array(v) for v in (wb, s_rms, s_core)) + (
        {o: np.array(v) for o, v in offs.items()},)


def draw():
    t, wb, s_rms, s_core, offs = series(f"PP_slice_ey{EY:g}_s1")
    s0 = Q.sigma_y(EY); D = Q.D_y(EY)
    x = 299792458.0 * t / SIGMA_Z
    x -= x[np.nanargmin(s_rms)]
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        for o in OFFS:
            ax.plot(x, offs[o] / s0, "-", color="0.72", lw=0.6, zorder=2)
            i = np.nanargmin(offs[o])
            ax.annotate(fr"${o:+g}\,\sigma_z$", (x[i], offs[o][i] / s0), textcoords="offset points",
                        xytext=(2, -7), fontsize=5.0, color="0.45")
        ax.plot(x, wb / s0, "--", color="C0", lw=1.0, zorder=4, label="whole beam (rms)")
        ax.plot(x, s_rms / s0, "-", color="C3", lw=1.0, zorder=5, label=fr"central slice rms ($|z-\bar z|<{HW:g}\,\sigma_z$)")
        ax.plot(x, s_core / s0, "-", color="C2", lw=1.0, zorder=6, label="central slice, Gaussian core")
        ax.axhline(1.0 / Q.R1_of_D(D), color="0.35", lw=0.8, ls=":", zorder=3)
        ax.text(0.985, 1.0 / Q.R1_of_D(D) * 1.05, r"$1/R(D_y)$", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=5.6, color="0.35")
        ax.set_yscale("log")
        ax.set_xlabel(r"$c\,(t-t_{\rm pinch})/\sigma_z$"); ax.set_ylabel(r"$\sigma_y(t)/\sigma_{y,0}$")
        ax.legend(loc="upper right", frameon=False, fontsize=5.2, title=fr"$\varepsilon_y={EY:g}\,$nm", title_fontsize=5.6)
        _ticks_in(ax)
        save_fig(fig, "WX_pinch_evolution"); plt.close(fig)
    print(f"minima/sigma_y0: whole beam {np.nanmin(wb)/s0:.3f}, slice rms {np.nanmin(s_rms)/s0:.3f}, "
          f"slice core {np.nanmin(s_core)/s0:.3f}, theory 1/R {1/Q.R1_of_D(D):.3f}")
    for o in OFFS:
        i = np.nanargmin(offs[o])
        print(f"  slice at {o:+g} sigma_z: min {offs[o][i]/s0:.3f} at c*dt/sigma_z = {x[i]:+.2f}")


def histogram(label=None, beam="beam1"):
    """WX_pinch_histogram: the central slice's y-distribution at the step of its core minimum (e_y = EY),
    with the Gaussian peak fit (bins above 20 % of the peak), the slice rms and IQR/1.349 marked."""
    import h5py as _h5
    from scipy.optimize import curve_fit
    label = label or f"PP_slice_ey{EY:g}_s1"
    t, wb, s_rms, s_core, offs = series(label)
    files = sorted(glob.glob(str(W.RUN_ROOT / label / "diags" / "pd" / "*.h5")))
    i_min = int(np.nanargmin(s_core)); s0 = Q.sigma_y(EY)
    with _h5.File(files[i_min], "r") as h5:
        it = list(h5["data"])[0]; y, z, w = beam_yzw(h5, it, beam)
    zc = np.average(z, weights=w); m = np.abs(z - zc) < HW * SIGMA_Z
    d = (y[m] - np.average(y[m], weights=w[m])) / s0; ww = w[m] / w[m].sum()
    rms = float(np.sqrt(np.average(d ** 2, weights=ww)))
    iqr = (wquantile(d, ww, .75) - wquantile(d, ww, .25)) / 1.349
    h_, e_ = np.histogram(d, bins=80, range=(-2, 2), weights=ww); c_ = 0.5 * (e_[1:] + e_[:-1])
    sel = h_ > 0.2 * h_.max()
    popt, _ = curve_fit(lambda x, A, s: A * np.exp(-x * x / (2 * s * s)), c_[sel], h_[sel], p0=(h_.max(), iqr), maxfev=4000)
    sg = abs(float(popt[1]))
    with plt.rc_context(PRL):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.bar(c_, h_, width=e_[1] - e_[0], color="C0", alpha=0.35, lw=0, label="slice particles")
        xx = np.linspace(-2, 2, 400)
        ax.plot(xx, popt[0] * np.exp(-xx * xx / (2 * sg * sg)), "-", color="C2", lw=1.2,
                label=fr"Gaussian peak fit: $\sigma={sg:.3f}\,\sigma_{{y,0}}$")
        for v, col, lab in ((rms, "C3", fr"rms $={rms:.3f}\,\sigma_{{y,0}}$"), (iqr, "0.3", fr"IQR/1.349 $={iqr:.3f}\,\sigma_{{y,0}}$")):
            ax.axvline(v, color=col, ls="--", lw=0.8, label=lab); ax.axvline(-v, color=col, ls="--", lw=0.8)
        ax.set_xlabel(r"$(y-\bar y_{\rm slice})/\sigma_{y,0}$"); ax.set_ylabel("charge fraction / bin")
        ax.set_xlim(-2, 2); ax.set_ylim(0, None)
        ax.legend(loc="upper right", frameon=False, fontsize=5.4, title=fr"$\varepsilon_y={EY:g}\,$nm, step of core minimum", title_fontsize=5.6)
        _ticks_in(ax)
        save_fig(fig, "WX_pinch_histogram"); plt.close(fig)
    print(f"histogram at step {i_min}: core sigma {sg:.3f}, rms {rms:.3f}, IQR/1.349 {iqr:.3f} (sigma_y0 units)")


if __name__ == "__main__":
    draw(); histogram()
