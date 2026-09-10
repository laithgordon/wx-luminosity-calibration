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

EXE=/pscratch/sd/l/laithg/WarpX/build/bin/warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.QED
INPUT=__INPUT__

cd __RUNDIR__ || exit 1

# runtime environment for this build
source /opt/cray/pe/lmod/lmod/init/bash
module load PrgEnv-gnu/8.6.0
module load cray-mpich/9.0.1
module load cudatoolkit/12.9

# 2026-08-20 NERSC maintenance moved the CUDA stack: the binary needs the CUDA 12.9
# libraries (cufft 11, nvJitLink 12) while the rebuilt cray-mpich GPU layer
# (libmpi_gtl_cuda) needs libcudart.so.13 -- expose both.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64

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
    particles_in.openpmd_backend=h5 __EXTRA__ \
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
