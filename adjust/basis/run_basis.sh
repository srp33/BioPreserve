#!/bin/bash
#SBATCH --job-name=basis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/basis_%j.log
#SBATCH --requeue

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -f "${SUBMIT_DIR}/__main__.py" ] && [ -f "${SUBMIT_DIR}/pipeline.py" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
    ADJUST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    PROJECT_ROOT="$(cd "${ADJUST_DIR}/.." && pwd)"
elif [ -d "${SUBMIT_DIR}/basis" ] && [ -f "${SUBMIT_DIR}/basis/__main__.py" ]; then
    ADJUST_DIR="${SUBMIT_DIR}"
    SCRIPT_DIR="${ADJUST_DIR}/basis"
    PROJECT_ROOT="$(cd "${ADJUST_DIR}/.." && pwd)"
elif [ -d "${SUBMIT_DIR}/adjust/basis" ] && [ -f "${SUBMIT_DIR}/adjust/basis/__main__.py" ]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
    ADJUST_DIR="${PROJECT_ROOT}/adjust"
    SCRIPT_DIR="${ADJUST_DIR}/basis"
else
    echo "Could not determine basis/ directory from submission directory: ${SUBMIT_DIR}" >&2
    echo "Submit this script from basis/, adjust/, or the repository root." >&2
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
DEFAULT_ARGS=(
    --combined-path basis/data/log_transformed-2_studies-test_metabric.csv
    --test-source metabric
    --output-dir out/
)

mkdir -p "${LOG_DIR}"

if [ "$#" -eq 0 ]; then
    BASIS_ARGS=("${DEFAULT_ARGS[@]}")
else
    BASIS_ARGS=("$@")
fi

cd "${PROJECT_ROOT}"

# Keep NumPy / BLAS aligned with the SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date)] Starting BASIS job"
echo "Project root: ${PROJECT_ROOT}"
echo "Basis directory: ${SCRIPT_DIR}"
echo "Working directory: ${ADJUST_DIR}"
echo "Submission directory: ${SUBMIT_DIR}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
printf 'Arguments:'
printf ' %q' "${BASIS_ARGS[@]}"
printf '\n'

cd "${ADJUST_DIR}"
pixi run python -m basis "${BASIS_ARGS[@]}"

echo "[$(date)] BASIS job completed"
