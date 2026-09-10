#!/usr/bin/env python3
"""n_x / n_m coupling test at e_y = 8 nm, n_y = 256, n_z = 128.

Question: is the +22% "n_x convergence" seen in the pegged overview a genuine
resolution effect, or a finite-n_m artefact of the fine x grid? Two rules:

  R1 (grid independence at converged n_m): for each n_x, the n_m ladder is
     'converged' where two successive decades agree within PLATEAU_TOL (2%,
     ~seed scatter). If the plateaus L_inf(n_x) agree within TOL across n_x,
     the smallest such n_x is justified.
  R2 (does the n_x sensitivity vanish as n_m grows?):
     Delta(n_m) = L(n_x, n_m) / L(512, n_m) - 1.  Monotone decrease toward 0
     with n_m => artefact; a finite floor => the fine grid resolves real physics.

Left panel: L vs n_m, one curve per n_x (bars = seed STD where >1 seed).
Right panel: Delta(n_m) per n_x.
Uses every results.csv row with e_y = 8, n_y = 256, n_z = 128 (any n_x, any n_m).

Output: plots/nx_nm_coupling.png
"""
import csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent            # repository root (flat layout: scripts, data/, plots/)
OUT  = ROOT / "plots" / "nx_nm_coupling.png"
EY, NY, NZ = 8.0, 256, 128
PLATEAU_TOL, TOL = 0.02, 0.05
COLORS = {512: "#2f6fb3", 1024: "#5aa469", 2048: "#e0a03c", 4096: "#c9562b", 8192: "#7b3f9e"}

runs = defaultdict(list)                       # (nx, nm) -> [L]
for r in csv.DictReader(open(ROOT / "data" / "results.csv")):
    if abs(float(r["e_y_nm"]) - EY) > 1e-9 or int(r["ny"]) != NY or int(r["nz"]) != NZ:
        continue
    runs[(int(r["nx"]), int(float(r["nm"])))].append(float(r["L"]))
nxs = sorted({k[0] for k in runs})
if not nxs:
    raise SystemExit("no data")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
plateau = {}
print(f"{'nx':>5} {'nm':>7} {'N':>2} {'L':>7} {'STD':>6}   decade change")
for nx in nxs:
    nm = np.array(sorted(k[1] for k in runs if k[0] == nx), float)
    L  = np.array([np.mean(runs[(nx, int(v))]) for v in nm])
    N  = np.array([len(runs[(nx, int(v))]) for v in nm])
    sd = np.array([np.std(runs[(nx, int(v))], ddof=1) if len(runs[(nx, int(v))]) > 1 else np.nan for v in nm])
    c = COLORS.get(nx, "k")
    ax1.errorbar(nm, L, yerr=np.where(np.isfinite(sd), sd, 0), fmt="o-", color=c, ms=5.5,
                 lw=1.4, capsize=3, label=rf"$n_x={nx}$")
    # R1: plateau where consecutive decade rungs agree within PLATEAU_TOL
    for i, (v, l, n, s) in enumerate(zip(nm, L, N, sd)):
        ch = ""
        if i > 0:
            ch = f"{(l / L[i-1] - 1) * 100:+.1f}% vs {nm[i-1]:.0e}"
        print(f"{nx:>5} {v:>7.0e} {n:>2d} {l:>7.3f} {s:>6.3f}   {ch}")
    conv = [i for i in range(1, len(nm)) if abs(L[i] / L[i-1] - 1) < PLATEAU_TOL and nm[i] / nm[i-1] >= 9]
    if conv:
        plateau[nx] = (nm[conv[0]], L[conv[0]:].mean())
        ax1.axhline(plateau[nx][1], color=c, ls=":", lw=1, alpha=0.7)
ax1.set_xscale("log"); ax1.set_xlabel(r"$n_m$"); ax1.set_ylabel(r"$L$  [$10^{34}$cm$^{-2}$s$^{-1}$]")
ax1.set_title(rf"$L$ vs $n_m$ per $n_x$  ($e_y=8$ nm, $n_y={NY}$, $n_z={NZ}$)")
ax1.grid(True, which="both", alpha=0.25); ax1.legend(fontsize=8)

# R2: Delta(nm) relative to the coarsest grid present (ideally 512)
ref = nxs[0]
for nx in nxs[1:]:
    common = sorted(v for (x, v) in runs if x == nx and (ref, v) in runs)
    if not common:
        continue
    d = [np.mean(runs[(nx, v)]) / np.mean(runs[(ref, v)]) - 1 for v in common]
    ax2.plot(common, np.array(d) * 100, "o-", color=COLORS.get(nx, "k"), ms=5.5, lw=1.4,
             label=rf"$n_x={nx}$ vs {ref}")
    print(f"Delta(nx={nx} vs {ref}): " + ", ".join(f"{v:.0e}: {x*100:+.1f}%" for v, x in zip(common, d)))
ax2.axhline(0, color="k", lw=0.8); ax2.axhspan(-TOL * 100, TOL * 100, color="grey", alpha=0.12, lw=0,
                                                label=r"$\pm5\%$")
ax2.set_xscale("log"); ax2.set_xlabel(r"$n_m$"); ax2.set_ylabel(rf"$L(n_x)/L({ref}) - 1$  [%]")
ax2.set_title(r"R2: does the $n_x$ sensitivity vanish as $n_m\to\infty$?")
ax2.grid(True, which="both", alpha=0.25); ax2.legend(fontsize=8)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nsaved {OUT}")
if plateau:
    print("R1 plateaus (nx: first converged nm, L_inf):")
    for nx, (v, l) in plateau.items():
        print(f"  {nx}: from nm={v:.0e}, L_inf={l:.3f}")
    vals = np.array([l for _, l in plateau.values()])
    print(f"  spread across plateaued nx: {(vals.max()/vals.min()-1)*100:.1f}%  (tol {TOL*100:.0f}%)")
else:
    print("R1: no nx has a plateau yet")
