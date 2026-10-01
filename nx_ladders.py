#!/usr/bin/env python3
"""n_m ladders at fixed n_x, eps_y = 20 nm, for WarpX and GUINEA-PIG++ (appendix figure WX_GP_nx_nm_ladders).

WarpX  -- data/nm_vs_nx_ladders.csv: one seed-1 run per (n_x, n_m); n_y = 256, n_z = 128, 3D solver, 3rd-order
          deposition (inputs/input_calib_C3_250.txt). Rungs reused from the tuning ladders are the seed-1 rows of
          data/results.csv (solver '3d'); same-seed re-runs are averaged as in nmreq.dedupe_seeds.
GP++   -- data/gp_exports/gp_nx_nm_ladders_20nm.csv: single common seed 100805; n_y = 512, n_z = 64, n_t = 6, FFT solver.
          A GP++ ladder counts as converged by the GP++ rule (paper_numbers 'nx_convergence_20nm' of that repo): the
          relative change from n_m = 5e6 to 1e7 is below GP_CONV_TOL.

Figure: each panel is normalized to the top rung of its n_x = 512 ladder; ladders that do not converge by their
code's rule are drawn open and dashed. WarpX rungs below WX_NM_MIN, where the single-run scatter exceeds 2 %, are
not drawn (they still enter the reduction).

Reduction per ladder (written to data/nm_vs_nx_summary.csv for WarpX):
  L_inf    mean over the highest window of >= 3 consecutive rungs spanning a decade with max/min - 1 <= PLATEAU_REL;
           none where no such window exists
  n_m_req  last-crossing bracket of the +/-5 % band about L_inf (nmreq.bracket)
  n_m_req_lower_bound  for ladders with no plateau, still falling at the top rung: L_inf <= L(top), so n_m_req is at
           least the largest rung with L > (1 + TOL) L(top)
  n_m_predicted  eq:law_nm on the frozen locus, 0.440 D_y^3.269 n_x / 512
  approach  side from which the ladder reaches L_inf (or L(top)): sign of the last rung outside PLATEAU_REL

    python3 nx_ladders.py        # writes the summary and plots/WX_GP_nx_nm_ladders.{pdf,png}
"""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nmreq as Q
from paper_figures import PRL, save_fig, _ticks_in

ROOT = Path(__file__).resolve().parent
WX_CSV = ROOT / "data" / "nm_vs_nx_ladders.csv"
GP_CSV = ROOT / "data" / "gp_exports" / "gp_nx_nm_ladders_20nm.csv"
SUMMARY = ROOT / "data" / "nm_vs_nx_summary.csv"
EY = 20.0
PLATEAU_REL = 0.002
GP_CONV_TOL = 0.001
WX_NM_MIN = 1e4
NXS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
COL = dict(zip(NXS, ["#7f7f7f", "#9467bd", "#1f77b4", "#000000", "#2ca02c", "#ff7f0e", "#d62728", "#8c564b"]))
MK = dict(zip(NXS, ["v", "^", "s", "o", "D", "o", "s", "D"]))


def _rows(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def ladders(code):
    """{n_x: dict(nm, L)} sorted by n_m."""
    if code == "wx":
        rows = [(int(r["nx"]), float(r["nm"]), float(r["L"])) for r in _rows(WX_CSV)]
    else:
        rows = [(int(r["n_x"]), float(r["n_m"]), float(r["L"])) for r in _rows(GP_CSV)]
    out = {}
    for nx in sorted({r[0] for r in rows}):
        pts = sorted((nm, L) for x, nm, L in rows if x == nx)
        out[nx] = dict(nm=np.array([p[0] for p in pts]), L=np.array([p[1] for p in pts]))
    return out


def plateau(d):
    """Mean over the highest window of >= 3 consecutive rungs spanning one decade within PLATEAU_REL, else None."""
    nm, L = d["nm"], d["L"]
    for j in range(len(nm) - 1, 1, -1):
        i = next((i for i in range(j - 2, -1, -1) if nm[j] / nm[i] >= 9.99), None)   # shortest decade window
        if i is not None and L[i:j + 1].max() / L[i:j + 1].min() - 1 <= PLATEAU_REL:
            return float(L[i:j + 1].mean()), (i, j)
    return None, None


def reduce(d, nx):
    Linf, win = plateau(d)
    nm, L = d["nm"], d["L"]
    pred = Q.LOCUS_NM_PREFACTOR * Q.D_y(EY) ** Q.LOCUS_NM_EXPONENT * nx / 512
    row = dict(n_x=nx, L_inf=Linf, L_top_rung=float(L[-1]), nm_top=float(nm[-1]), n_m_predicted=pred,
               n_m_req=None, n_m_req_lower_bound=None)
    ref = Linf if Linf is not None else L[-1]
    if Linf is not None:
        b = Q.bracket(d, Linf)
        row["n_m_req"] = b["nreq"]
        row["ratio"] = b["nreq"] / pred if b["nreq"] else None
    else:
        above = [nm[i] for i in range(len(nm)) if L[i] > (1 + Q.TOL) * L[-1]]
        row["n_m_req_lower_bound"] = float(max(above)) if above else None
        row["ratio"] = row["n_m_req_lower_bound"] / pred if above else None
    lo = win[0] if win else len(nm) - 1
    out = [i for i in range(lo) if abs(L[i] / ref - 1) > PLATEAU_REL]
    row["approach"] = ("from_above" if L[out[-1]] > ref else "from_below") if out else "flat"
    return row


def write_summary(rows):
    head = ["# nm_vs_nx_summary.csv -- WarpX n_m ladders at fixed n_y=256, n_z=128, eps_y=20 nm, one per n_x, seed 1; "
            "written by nx_ladders.py (rules in its docstring)",
            "n_x,L_inf,L_top_rung,n_m_req,n_m_req_lower_bound,n_m_predicted,ratio,approach,nm_top"]
    f = lambda v, fmt: "" if v is None else fmt.format(v)
    lines = [",".join([str(r["n_x"]), f(r["L_inf"], "{:.6f}"), f(r["L_top_rung"], "{:.6f}"), f(r["n_m_req"], "{:.0f}"),
                       f(r["n_m_req_lower_bound"], "{:.0f}"), f(r["n_m_predicted"], "{:.0f}"), f(r["ratio"], "{:.3f}"),
                       r["approach"], f(r["nm_top"], "{:.0f}")]) for r in rows]
    SUMMARY.write_text("\n".join(head + lines) + "\n")


def numbers():
    """WarpX paper numbers: each ladder relative to the n_x = 512 plateau."""
    red = {nx: reduce(d, nx) for nx, d in ladders("wx").items()}
    L0 = red[512]["L_inf"]
    per = [dict(n_x=nx, L_inf=r["L_inf"], plateau_over_512=(r["L_inf"] / L0 if r["L_inf"] else None),
                L_top_rung=r["L_top_rung"], top_rung_over_512=r["L_top_rung"] / L0, nm_top=r["nm_top"],
                approach=r["approach"], n_m_req=r["n_m_req"], n_m_req_lower_bound=r["n_m_req_lower_bound"],
                n_m_predicted=r["n_m_predicted"], ratio_to_law=r["ratio"])
           for nx, r in red.items()]
    conv = [p["L_inf"] for p in per if p["n_x"] in (256, 512, 1024)]
    return dict(L_inf_512=L0, per_n_x=per, spread_256_to_1024=max(conv) / min(conv) - 1)


def converged(code, d, nx):
    if code == "wx":
        return plateau(d)[0] is not None
    i5, i7 = list(d["nm"]).index(5e6), list(d["nm"]).index(1e7)
    return abs(d["L"][i7] / d["L"][i5] - 1) < GP_CONV_TOL


def draw():
    data = {c: ladders(c) for c in ("gp", "wx")}
    with plt.rc_context(PRL):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
        for ax, code, title in ((axs[0], "gp", "GUINEA-PIG++"), (axs[1], "wx", "WarpX")):
            lad = data[code]
            L0 = lad[512]["L"][-1]
            ax.axhspan(0.99, 1.01, color="0.85", alpha=0.6, lw=0, zorder=0)
            ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=1)
            for nx, d in lad.items():
                nc = not converged(code, d, nx)
                k = d["nm"] >= (WX_NM_MIN if code == "wx" else 0)
                ax.plot(d["nm"][k], d["L"][k] / L0, marker=MK[nx], ls="--" if nc else "-", color=COL[nx],
                        mfc="white" if nc else COL[nx], ms=3.0, lw=0.9, markeredgewidth=0.8, zorder=3, label=rf"$n_x={nx}$")
            ax.set_xscale("log"); ax.set_xlim(*((6e4, 1.6e7) if code == "gp" else (6e3, 2e8))); ax.set_ylim(0.92, 1.40)
            ax.set_xlabel(r"$n_m$"); ax.set_title(title, fontsize=8)
            _ticks_in(ax)
        h = {}
        for ax in axs:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                h[ll] = hh
        order = [rf"$n_x={nx}$" for nx in NXS if rf"$n_x={nx}$" in h]
        fig.legend([h[k] for k in order], order, loc="upper center", ncol=len(order), frameon=False, fontsize=6.2,
                   bbox_to_anchor=(0.5, 1.07), handletextpad=0.3, columnspacing=1.0)
        axs[0].set_ylabel(r"$\mathscr{L}\,/\,\mathscr{L}_{\rm top}(n_x=512)$")
        fig.tight_layout(w_pad=0.6)
        save_fig(fig, "WX_GP_nx_nm_ladders"); plt.close(fig)


if __name__ == "__main__":
    write_summary([reduce(d, nx) for nx, d in ladders("wx").items()])
    draw()
    b = numbers()
    print(f"WarpX L_inf(512) = {b['L_inf_512']:.4f}, spread 256-1024 = {b['spread_256_to_1024']:.4f}")
    for p in b["per_n_x"]:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
