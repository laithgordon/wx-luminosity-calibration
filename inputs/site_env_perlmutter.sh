# Runtime environment for the WarpX build used in the study (NERSC Perlmutter, sourced by template.sbatch).
# Replace this file, or point WX_SITE_ENV at another one, on a different machine.
source /opt/cray/pe/lmod/lmod/init/bash
module load PrgEnv-gnu/8.6.0
module load cray-mpich/9.0.1
module load cudatoolkit/12.9

# 2026-08-20 NERSC maintenance moved the CUDA stack: the binary needs the CUDA 12.9
# libraries (cufft 11, nvJitLink 12) while the rebuilt cray-mpich GPU layer
# (libmpi_gtl_cuda) needs libcudart.so.13 -- expose both.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64
