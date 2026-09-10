#!/usr/bin/env python3
"""Measurement 2 with statistics: per-pass deposition-error correlation per emittance, from the
PQ dumps (10 seeds), particles deposited offline onto the CALIBRATED grid (n_y = NY_CONS) vs an
8x refined reference. Waists = prominent minima (5% of median) of each run's cached core series
(the dump window 30-85 confines everything to the collision). Output: mean pairwise correlation
C_bar per e_y (seed mean +/- STD) and adjacent-pass correlation; cached to Data/depo_corr_stat.json."""
import glob, json, sys
from pathlib import Path
import numpy as np
import h5py
from scipy.signal import find_peaks
import nmreq as Q
from plot_slice_pinch import beam_yzw, SIGMA_Z
from plot_core_width import run_widths, HW
from deposition_error import deposit, REFINE

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
import wxcal as W


def run_cbar(label, e):
    ny = Q.NY_CONS[e]; sy0 = Q.sigma_y(e)
    dy = 32 * sy0 / ny; y0 = -16 * sy0
    d = run_widths(label)
    g = 0.5 * (d["gauss1"] + d["gauss2"]); steps = d["step"].astype(int)
    m = np.isfinite(g)
    pk, _ = find_peaks(-g[m], prominence=0.05 * np.nanmedian(g[m]))
    ws = [int(steps[m][i]) for i in pk]
    if len(ws) < 2:
        return None
    deltas = []
    for st in ws:
        f = W.RUN_ROOT / label / "diags" / "pd" / f"openpmd_{st:06d}.h5"
        with h5py.File(f, "r") as h5:
            y, z, w = beam_yzw(h5, str(st), "beam1")
        zc = np.average(z, weights=w); msk = np.abs(z - zc) < HW * SIGMA_Z
        y, w = y[msk], w[msk]
        rho = deposit(y, w, dy, ny, y0)
        rho_ref = deposit(y, w, dy / REFINE, ny * REFINE, y0).reshape(ny, REFINE).mean(axis=1)
        deltas.append((rho - rho_ref, rho_ref))
    peak = max(r.max() for _, r in deltas)
    core = np.zeros(ny, bool)
    for _, r in deltas:
        core |= r > 0.05 * peak
    V = np.array([dd[core] for dd, _ in deltas])
    nn = np.sqrt((V ** 2).sum(axis=1))
    C = (V @ V.T) / np.outer(nn, nn)
    off = C[np.triu_indices(len(V), 1)]
    adj = np.array([C[i, i + 1] for i in range(len(V) - 1)])
    return dict(n=len(ws), cbar=float(np.mean(off)), adj=float(np.mean(adj)),
                adj_last=float(adj[-1]))                     # last adjacent pair = late in the crossing


def main():
    import csv
    out = {}
    for e in (20, 16, 12, 8, 4, 2, 1, 0.5):
        labs = sorted({r["label"] for r in csv.DictReader(open(W.JOBLIST))
                       if r["phase"] == "PQ" and (W.RUN_ROOT / r["label"] / "job_status.txt").exists()
                       and abs(float(r["e_y_nm"]) - e) < 1e-9})
        rs = [run_cbar(l, e) for l in labs]
        rs = [r for r in rs if r]
        cb = np.array([r["cbar"] for r in rs])
        out[str(e)] = dict(N=float(np.mean([r["n"] for r in rs])), sN=float(np.std([r["n"] for r in rs])),
                           cbar=float(cb.mean()), scbar=float(cb.std(ddof=1)),
                           sem=float(cb.std(ddof=1) / np.sqrt(len(cb))),
                           adj=float(np.mean([r["adj"] for r in rs])),
                           adj_last=float(np.mean([r["adj_last"] for r in rs])),
                           adj_last_sem=float(np.std([r["adj_last"] for r in rs], ddof=1) / np.sqrt(len(rs))),
                           nm=float(Q.NM_CONS[e]), nseeds=len(rs))
        print(f"e_y={e:>4g}: N_osc={out[str(e)]['N']:.2f}±{out[str(e)]['sN']:.2f}  "
              f"C_bar={out[str(e)]['cbar']:+.3f}±{out[str(e)]['sem']:.3f}(SEM)  adj_last={out[str(e)]['adj_last']:+.3f}  ({len(rs)} seeds)", flush=True)
    (ROOT / "data" / "depo_corr_stat.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
