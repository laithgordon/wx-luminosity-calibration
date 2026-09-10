#!/usr/bin/env python3
"""Append new jobs to joblist.csv (e.g. reserve draws: extra ladder points,
extra seeds, truth upgrades).

Usage examples:
  # 5 seeds of an extra nm ladder point at e_y=2
  python3 add_jobs.py --phase P2 --stage A_nm --sim nm_xhi \\
      --ey 2 --nx 8192 --ny 256 --nz 128 --nm 5000000 --seeds 5 \\
      --nodes 1 --walltime 1.5

  # extra seeds 6-10 on an existing point
  python3 add_jobs.py --phase P1 --stage A_nm --sim nm_mid --ey 12 \\
      --nx 8192 --ny 256 --nz 128 --nm 100000 --seeds 10 --first-seed 6 \\
      --nodes 1 --walltime 0.5
"""
import argparse
import wxcal as W

ap = argparse.ArgumentParser()
for a in ("phase", "stage", "sim"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--ey", required=True, help="e_y in nm (e.g. 0.5)")
for a in ("nx", "ny", "nz", "nm"):
    ap.add_argument(f"--{a}", required=True, type=float)
ap.add_argument("--seeds", type=int, default=1, help="last seed number")
ap.add_argument("--first-seed", type=int, default=1)
ap.add_argument("--nodes", type=int, required=True)
ap.add_argument("--walltime", type=float, required=True, help="hours (set ~2x expected)")
ap.add_argument("--qos", default=None, help="default: overrun if walltime<2h else regular")
args = ap.parse_args()

rows = W.read_joblist()
have = {r["label"] for r in rows}
qos = args.qos or ("overrun" if args.walltime < 2.0 else "regular")
added = 0
for seed in range(args.first_seed, args.seeds + 1):
    lab = W.label_for(args.phase, args.stage, args.sim, args.ey, seed)
    if lab in have:
        print(f"skip (exists): {lab}")
        continue
    rows.append(dict(label=lab, phase=args.phase, stage=args.stage, sim=args.sim,
                     e_y_nm=args.ey, nx=int(args.nx), ny=int(args.ny),
                     nz=int(args.nz), nm=f"{args.nm:.0f}", seed=seed,
                     nodes=args.nodes, walltime=W.walltime_str(args.walltime),
                     qos=qos, status="pending", slurm_id="", L_rate=""))
    added += 1
W.write_joblist(rows)
print(f"added {added} job(s); joblist now {len(rows)}")
