#!/usr/bin/env python3
"""Submit pending jobs from joblist.csv, in phases.

Usage examples:
  python3 submit.py --phase P0 --dry-run     # show what would be submitted
  python3 submit.py --phase P0               # submit all pending P0 jobs
  python3 submit.py --phase P1 --limit 10    # first 10 pending P1 jobs
  python3 submit.py --label P3_nm_hi_ey0p5_s1
  python3 submit.py --phase P5-gated --qos regular   # gated truths, when quota allows

Only 'pending' jobs are ever submitted (safe to re-run). Each job gets its own
run directory under /pscratch/.../wx_calibration/<label>/ with the rendered
sbatch kept alongside the outputs.
"""
import argparse, subprocess, re, datetime
import wxcal as W

ap = argparse.ArgumentParser()
ap.add_argument("--phase", nargs="*", default=None)
ap.add_argument("--label", nargs="*", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--account", default="m4272_g")
ap.add_argument("--qos", default=None, help="override the joblist qos for this batch")
ap.add_argument("--extra", default="", help="extra WarpX args appended to the command line")
ap.add_argument("--numprocs", default=None, help='override decomposition, e.g. "2 2 2"')
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

rows = W.read_joblist()
template = open(W.TEMPLATE).read()
picked = [r for r in rows if r["status"] == "pending"
          and (args.phase is None or r["phase"] in args.phase)
          and (args.label is None or r["label"] in args.label)]
if args.limit:
    picked = picked[:args.limit]
if not picked:
    print("nothing to submit (no matching pending jobs)")
    raise SystemExit

total_nh = 0.0
n_ok = 0
for r in picked:
    emity, sigmay = W.emity_sigmay(r["e_y_nm"])
    nodes = int(r["nodes"]); ntasks = nodes * 4
    qos = args.qos or r["qos"]
    rundir = W.RUN_ROOT / r["label"]
    sub = (template
           .replace("__JOBNAME__", f"wxcal_{r['label']}")
           .replace("__WALL__", r["walltime"]).replace("__NODES__", str(nodes))
           .replace("__NTASKS__", str(ntasks)).replace("__ACCOUNT__", args.account)
           .replace("__QOS__", qos).replace("__RUNDIR__", str(rundir))
           .replace("__INPUT__", str(W.INPUT))
           .replace("__NX__", r["nx"]).replace("__NY__", r["ny"])
           .replace("__NZ__", r["nz"]).replace("__NM__", r["nm"])
           .replace("__EMITY__", f"{emity:.6e}").replace("__SIGMAY__", f"{sigmay:.6e}")
           .replace("__SEED__", r["seed"])
           .replace("__NUMPROCS__", args.numprocs or W.numprocs(nodes, r["nx"], r["ny"]))
           .replace("__EXTRA__", args.extra))
    wall_h = sum(float(x) * f for x, f in zip(r["walltime"].split(":"), (1, 1/60, 1/3600)))
    total_nh += wall_h * nodes
    if args.dry_run:
        print(f"DRY: {r['label']:34s} {nodes}N {r['walltime']} q={qos} "
              f"numprocs={W.numprocs(nodes, r['nx'], r['ny'])}")
        continue
    rundir.mkdir(parents=True, exist_ok=True)
    script = rundir / "job.sbatch"
    script.write_text(sub)
    out = subprocess.run(["sbatch", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    m = re.search(r"Submitted batch job (\d+)", out.stdout)
    if not m:
        print(f"FAILED {r['label']}: {out.stderr.strip() or out.stdout.strip()}")
        continue
    r["status"], r["slurm_id"] = "submitted", m.group(1)
    n_ok += 1
    print(f"submitted {r['label']:34s} -> {m.group(1)} ({nodes}N, q={qos})")

if not args.dry_run:
    W.write_joblist(rows)
if args.dry_run:
    print(f"would submit {len(picked)} job(s); walltime-cap node-h upper bound: {total_nh:.1f}")
else:
    print(f"submitted {n_ok} of {len(picked)} job(s)" + (f" ({len(picked)-n_ok} FAILED at sbatch)" if n_ok < len(picked) else "")
          + f"; walltime-cap node-h upper bound: {total_nh:.1f}")
