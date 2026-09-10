#!/usr/bin/env python3
"""Shared helpers for the WX_Calibration pipeline (joblist I/O, model rules)."""
import csv, math, re
from pathlib import Path

HERE      = Path(__file__).resolve().parent          # repository root
ROOT      = HERE
JOBLIST   = HERE / "data" / "joblist.csv"
RESULTS   = HERE / "data" / "results.csv"
PLAN_CSV  = ROOT / "Plan" / "WX Calibration Plan.csv"
TEMPLATE  = HERE / "template.sbatch"
INPUT     = HERE / "data" / "input_calib_C3_250.txt"
RUN_ROOT  = Path("/pscratch/sd/l/laithg/simulations/wx_calibration")

FIELDS = ["label","phase","stage","sim","e_y_nm","nx","ny","nz","nm","seed",
          "nodes","walltime","qos","status","slurm_id","L_rate"]

# luminosity conversion (identical to analysis/extract_lumi.py)
NUM_BUNCHES, TRAIN_REP = 133, 120

def ey_tag(e):
    s = f"{float(e):g}".replace(".", "p")
    return f"ey{s}"

def label_for(phase, stage, sim, e_y, seed):
    return f"{phase.replace('-','')}_{sim}_{ey_tag(e_y)}_s{int(seed)}"

# fixed optics/beam parameters
BETA_Y = 0.12e-3                       # m  (beta_y = 0.12 mm)
GAMMA  = 125e9 / 0.51099895e6          # beam_energy / m_e c^2 = 244618.9

def ny_theory(e_y_nm):
    """n_y(e_y) theory pegging: nearest power of 2 to 256*(8/e_y)^(1/4),
    ties rounding UP (log-2 half-up), per the plan document.
    e.g. e_y=2 -> 362 -> 512 (NOT 256); e_y=4 -> 304 -> 256."""
    import math as _m
    x = 256.0 * (8.0 / float(e_y_nm)) ** 0.25
    return 2 ** int(_m.floor(_m.log2(x) + 0.5))

def emity_sigmay(e_y_nm):
    """SI values for the e_y override: emity = e_y (normalized);
    sigma_y = sqrt(beta_y * e_y / gamma) with beta_y and gamma fixed.
    (e_y = 8 nm -> sigma_y = 1.9806 nm, matching the calib deck's 1.98 nm.)"""
    emity = float(e_y_nm) * 1e-9
    return emity, math.sqrt(BETA_Y * emity / GAMMA)

def pow2_nodes(n):
    """Round node count up to a power of 2 so numprocs divides the grid."""
    p = 1
    while p < int(n):
        p *= 2
    return p

def numprocs(nodes, nx, ny):
    """Decomposition. 1 node -> "4 1 1"; 2+ nodes -> "px 2 2".
    NOTE (2026-08-15): the earlier "py=2, pz=1 hangs" theory was wrong -- every
    hang was a GPU OOM in the FFT Poisson solver (see Plan §8). The decomposition
    is irrelevant to that; keeping the z split for 2+ nodes only because it
    balances the two beams' particles when nm is large (Plan §8 rule 2)."""
    g = int(nodes) * 4
    if g <= 4:
        return f"{g} 1 1"
    return f"{g // 4} 2 2"

def walltime_str(hours):
    h = max(0.5, float(hours))
    hh = int(h); mm = int(round((h - hh) * 60))
    if mm == 60: hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}:00"

def read_joblist():
    if not JOBLIST.exists():
        return []
    return list(csv.DictReader(open(JOBLIST)))

def write_joblist(rows):
    with open(JOBLIST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

def integrate_difflum(path):
    """Rectangle-sum integral of dL/dE_com at the final timestep: sum of bin
    values times the uniform bin width — the exact inverse of the histogram
    binning, robust to a peak in any bin (unlike the trapezoid rule, which
    half-weights edge bins and biased the pre-2.1x results low by up to ~26%).
    Returns L in m^-2."""
    import numpy as np
    with open(path) as f:
        header = f.readline()
    E_com = np.array([float(x) for x in re.findall(r"=([\d.eE+\-]+)\(eV\)", header)])
    data = np.loadtxt(path)
    row = data[2:] if data.ndim == 1 else data[-1, 2:]
    return float(np.sum(row[:len(E_com)]) * (E_com[1] - E_com[0]))

def lumi_to_rate(L_m2):
    return L_m2 * 1e-4 * NUM_BUNCHES * TRAIN_REP / 1e34

# pscratch OST outage (2026-09-01, NERSC ticket pending): several OSTs are hung
# (122 and 18 confirmed) — any data read or stat() of a file striped there
# blocks forever in the kernel (cl_sync_io_wait), with no error. `lfs getstripe
# -i` only queries the metadata server, so it never blocks; a 5 s `timeout dd`
# read probe then classifies the file's OST as hung or healthy. Hung verdicts
# are persisted in bad_osts.txt so each OST is probed at most once ever.
# DELETE bad_osts.txt once NERSC restores the filesystem, and rebuild the
# analysis caches listed in Analysis/Data/caches_built_during_ost_outage.txt.
BAD_OSTS_FILE = HERE / "bad_osts.txt"
_ost_good = set()
_ost_bad = None

def _bad_osts():
    global _ost_bad
    if _ost_bad is None:
        _ost_bad = set(BAD_OSTS_FILE.read_text().split()) if BAD_OSTS_FILE.exists() else set()
    return _ost_bad

def _ost_hung(idx, probe_file):
    """dd-read 64k of probe_file with a 5 s timeout; only a timeout (rc 124)
    means the OST is hung — any other failure is the caller's problem."""
    import subprocess
    bad = _bad_osts()
    if idx in bad:
        return True
    if idx in _ost_good:
        return False
    r = subprocess.run(["timeout", "5", "dd", f"if={probe_file}", "of=/dev/null",
                        "bs=64k", "count=1"], capture_output=True)
    if r.returncode == 124:
        bad.add(idx)
        with open(BAD_OSTS_FILE, "a") as f:
            f.write(idx + "\n")
        print(f"pscratch OST {idx} is hung: skipping files striped there", flush=True)
        return True
    _ost_good.add(idx)
    return False

def _stripe_indices(files):
    import subprocess
    try:
        out = subprocess.run(["lfs", "getstripe", "-i"] + files,
                             capture_output=True, text=True, timeout=300)
        idx = out.stdout.split()
        if len(idx) != len(files) or out.returncode != 0:
            raise ValueError("unpaired getstripe output")
        return idx
    except Exception:
        pass
    idx = []
    for f in files:
        try:
            o = subprocess.run(["lfs", "getstripe", "-i", f], capture_output=True,
                               text=True, timeout=60).stdout.strip()
        except Exception:
            o = ""
        idx.append(o or "?")            # missing file / not Lustre: harmless to keep
    return idx

def drop_bad_ost(files):
    """Filter a list of existing pscratch files down to the ones not striped on a
    hung OST. Call this on any batch of files before reading or stat()ing them."""
    files = [str(f) for f in files]
    if not files:
        return files
    return [f for f, o in zip(files, _stripe_indices(files))
            if o == "?" or not _ost_hung(o, f)]

def on_bad_ost(path):
    """True if `path` exists and its Lustre objects sit on a hung OST. Probe this
    BEFORE any exists()/stat()/open() of a single pscratch file."""
    return not drop_bad_ost([path])


# results.csv 'solver' tag per phase (anything not listed is the 3D IGF calibration configuration).
# PS  = warpx.use_2d_slices_fft_solver=1                         (2026-08, field-solver test)
# PC3 = algo.particle_shape=1 (CIC, all axes), 3D solver         (2026-09-08, deposition test mimicking GP++'s (1,1,0))
# PC2 = algo.particle_shape=1 + warpx.use_2d_slices_fft_solver=1 (2026-09-08)
SOLVER_TAG = {"PS": "2d", "PC3": "3d_cic", "PC2": "2d_cic"}
