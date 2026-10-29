#!/bin/bash
# Slurm job: hybrid MPI + OpenMP run of the two-stream PIC code, plus a small strong-scaling sweep.
# Edit the #SBATCH lines and the module names for your cluster, then:  sbatch scripts/slurm_pic.sh
#SBATCH --job-name=pic_two_stream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # MPI ranks per node
#SBATCH --cpus-per-task=8            # OpenMP threads per rank
#SBATCH --time=00:15:00
#SBATCH --output=pic_%j.out
##SBATCH --partition=<your-partition>
##SBATCH --account=<your-project>

set -euo pipefail

# ---- environment: adjust to your site (examples for common module systems)
module purge 2>/dev/null || true
module load cmake gcc openmpi 2>/dev/null || echo "module load failed: load a compiler, MPI and CMake by hand"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- build (once) and run the physics tests on a compute node
if [ ! -x build/hpc/pic_two_stream ]; then
  cmake --preset hpc
  cmake --build --preset hpc -j "${SLURM_CPUS_PER_TASK:-4}"
fi
ctest --preset hpc

# ---- production run: 64 million particles, phase-space snapshots every 100 steps
srun --cpu-bind=cores build/hpc/pic_two_stream \
     --particles 64000000 --grid 256 --tend 40 --snapshots 100 --out "run_${SLURM_JOB_ID:-local}"

# ---- strong scaling: fixed problem, 1..ntasks ranks (threads per rank fixed)
echo "ranks,threads,seconds" > "scaling_${SLURM_JOB_ID:-local}.csv"
for n in 1 2 4; do
  [ "$n" -le "${SLURM_NTASKS:-4}" ] || continue
  secs=$( { /usr/bin/time -f "%e" srun -n "$n" --cpu-bind=cores build/hpc/pic_two_stream \
             --particles 16000000 --grid 256 --tend 10 --out "scaling_tmp_$n" > /dev/null; } 2>&1 | tail -1 )
  echo "$n,$OMP_NUM_THREADS,$secs" >> "scaling_${SLURM_JOB_ID:-local}.csv"
done
cat "scaling_${SLURM_JOB_ID:-local}.csv"
