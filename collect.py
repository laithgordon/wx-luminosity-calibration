#!/usr/bin/env python3
"""Refresh results.csv from completed runs and mark jobs done in joblist.csv.

For every job in joblist.csv with status submitted/done, integrate the
differential luminosity spectrum diags/reducedfiles/DiffLum_beam1_beam2.txt
(final timestep, trapezoidal) and convert it to the luminosity L in units of
1e34 cm^-2 s^-1 (same maths as analysis/extract_lumi.py). results.csv lives in
../Analysis/Data/ and is fully rebuilt on each run; joblist rows whose result parsed with L > 0 are marked
'done'. Jobs whose SLURM job finished but produced no usable DiffLum are
flagged 'check' (look at slurm.err / output.err in the run dir).

Usage:  python3 collect.py
"""
import csv, subprocess
import wxcal as W

rows = W.read_joblist()
results, done, checked, waiting = [], 0, 0, 0

# one sacct call for all known slurm ids -> state
# only rows whose fate is still open need an sacct lookup (querying every historic
# ID makes slurmdbd crawl); a slow/failed sacct just leaves them 'submitted'.
ids = [r["slurm_id"] for r in rows if r["slurm_id"] and r["status"] == "submitted"]
state = {}
if ids:
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State",
                              "-j", ",".join(ids)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        print("warning: sacct timed out; leaving unresolved rows as 'submitted'")
        out = ""
    for line in out.splitlines():
        p = line.split("|")
        if len(p) >= 2:
            state[p[0]] = p[1]

for r in rows:
    if r["status"] in ("pending", "cancelled"):
        continue                      # never submitted / withdrawn by hand: nothing to collect
    L_rate = None
    if r["status"] == "done" and r["L_rate"]:
        # already collected once: reuse the stored value instead of re-reading
        # pscratch (faster, and immune to files stranded on a hung OST)
        L_rate = float(r["L_rate"])
    else:
        f = W.RUN_ROOT / r["label"] / "diags" / "reducedfiles" / "DiffLum_beam1_beam2.txt"
        if W.on_bad_ost(f):
            print(f"skipped (hung OST): {r['label']}")
            waiting += 1
            continue                  # unreadable, not broken: leave the row as is
        elif f.exists():
            try:
                L_m2 = W.integrate_difflum(f)
                L_rate = W.lumi_to_rate(L_m2)
            except Exception as e:
                print(f"parse error {r['label']}: {e}")
    if L_rate is not None and L_rate > 0:
        r["status"], r["L_rate"] = "done", f"{L_rate:.6f}"
        emity, sigmay = W.emity_sigmay(r["e_y_nm"])
        results.append(dict(e_y_nm=r["e_y_nm"], sigmay_nm=f"{sigmay*1e9:.4f}",
                            nx=r["nx"], ny=r["ny"], nz=r["nz"], nm=r["nm"],
                            seed=r["seed"], L=f"{L_rate:.6f}",
                            solver=W.SOLVER_TAG.get(r["phase"], "3d")))   # PS = 2d slices; PC3/PC2 = CIC deposition (algo.particle_shape=1), 3d / 2d
        done += 1
    else:
        st = state.get(r["slurm_id"], "")
        if r["status"] == "check":
            checked += 1
        elif st and not st.startswith(("RUNNING", "PENDING")):
            r["status"] = "check"     # job over, but no usable luminosity
            checked += 1
        else:
            waiting += 1

# results.csv: beam parameters (e_y, sigma_y), grid parameters, seed, and the
# luminosity L in units of 1e34 cm^-2 s^-1 (the DiffLum spectrum integrated at
# the final step, converted to a collider rate with 133 bunches x 120 Hz —
# the same convention as wx_calib_lumi.csv and the convergence notebook).
W.RESULTS.parent.mkdir(parents=True, exist_ok=True)
with open(W.RESULTS, "w", newline="") as fo:
    fields = ["e_y_nm","sigmay_nm","nx","ny","nz","nm","seed","L","solver"]
    w = csv.DictWriter(fo, fieldnames=fields)
    w.writeheader(); w.writerows(results)
W.write_joblist(rows)
print(f"results.csv: {done} results | still running/queued: {waiting} | "
      f"needs attention ('check'): {checked}")
