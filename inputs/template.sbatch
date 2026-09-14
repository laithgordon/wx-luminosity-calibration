#!/bin/bash -l

# WX_Calibration standard submit script (rendered by submit.py — do not edit
# the rendered copies; edit template.sbatch instead).
# Pattern: whole GPU nodes, 4 ranks/node, gpu-bind=none (the verified June-sweep
# pattern). QOS/account/walltime filled per job from the joblist.

#SBATCH -t __WALL__
#SBATCH -N __NODES__
#SBATCH -J __JOBNAME__
#SBATCH -A __ACCOUNT__
#SBATCH -q __QOS__
#SBATCH -C gpu&hbm80g
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -c 32
#SBATCH -o __RUNDIR__/slurm.out
#SBATCH -e __RUNDIR__/slurm.err

EXE=__EXE__
INPUT=__INPUT__

cd __RUNDIR__ || exit 1

# runtime environment for this build (site-specific; wxcal.SITE_ENV)
source __SITE_ENV__

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export SLURM_CPU_BIND=cores

# Clear stale outputs from any previous attempt in this run dir (the watchdog
# below greps output.err, so it must not see an old abort message).
rm -f output.txt output.err Backtrace.* job_status.txt

# Overrides: study variables + luminosity-only diagnostics (this build has no
# ADIOS2, so the Full particle dumps are disabled and any openPMD path forced
# to h5 — same as the June sweep wrapper).
srun --ntasks=__NTASKS__ --ntasks-per-node=4 --gpu-bind=none -c 32 \
    "$EXE" "$INPUT" \
    my_constants.nx=__NX__ \
    my_constants.ny=__NY__ \
    my_constants.nz=__NZ__ \
    my_constants.nmacropart=__NM__ \
    my_constants.emity=__EMITY__ \
    my_constants.sigmay=__SIGMAY__ \
    warpx.random_seed=__SEED__ \
    warpx.numprocs="__NUMPROCS__" \
    diagnostics.diags_names=particles_in \
    particles_in.intervals=0 \
    particles_in.openpmd_backend=h5 \
    qed_qs.load_table_from=__QED_QS_TABLE__ \
    qed_bw.load_table_from=__QED_BW_TABLE__ __EXTRA__ \
    > output.txt 2> output.err &
SRUN_PID=$!

# Watchdog. Every OOM so far (44 jobs) printed "Arena out of memory" to output.err
# within ~1-2 min and then HUNG until the walltime cap: the aborting rank re-enters
# AMReX's signal handler inside MPI_Abort and never exits, so slurm's
# KillOnBadExit (already the site default) never fires. Cancel the job ourselves.
sleep 30
while kill -0 "$SRUN_PID" 2>/dev/null; do
    if grep -qE "out of memory|amrex::Abort|Segfault|SIGABRT|SIGSEGV|SIGFPE" output.err 2>/dev/null; then
        echo "watchdog: abort detected in output.err at $(date); cancelling job" >> job_status.txt
        sleep 60          # let the other ranks flush their Backtrace files
        scancel "$SLURM_JOB_ID"
        break
    fi
    sleep 30
done
wait "$SRUN_PID"
echo "exit=$? $(date)" >> job_status.txt
