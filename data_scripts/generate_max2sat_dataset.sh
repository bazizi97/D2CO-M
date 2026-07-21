#!/usr/bin/env bash
# ==============================================================================
# Script to generate Weighted MAX-2-SAT Datasets for D2CO-M Training/Val/Test
# Uses akmaxsat (pyakmaxsat) solver and outputs JSON graph representations.
# ==============================================================================

set -e

# Default output base directory
DATA_DIR="${1:-./data/max2sat}"
N_WORKERS="${2:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_SCRIPT="${SCRIPT_DIR}/process_dataset.py"

echo "======================================================================"
echo "Generating Weighted MAX-2-SAT Dataset with Akmaxsat solver"
echo "Base Output Directory: ${DATA_DIR}"
echo "Parallel Workers: ${N_WORKERS}"
echo "Max Soft Weight: 10"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Training Set (90,000 problems, seed 100)
# ------------------------------------------------------------------------------
TRAIN_DIR="${DATA_DIR}/train"
echo ""
echo "[1/3] Generating Training Dataset (90,000 instances) in ${TRAIN_DIR}..."
mkdir -p "${TRAIN_DIR}"
python3 "${PROCESS_SCRIPT}" \
    --mode all \
    --problem_dir "${TRAIN_DIR}" \
    --seed 100 \
    --min_n 50 \
    --max_n 150 \
    --min_alpha 2 \
    --max_alpha 8 \
    --n_problem 90000 \
    --solver akmaxsat \
    --weighted \
    --max_weight 10 \
    --n_workers "${N_WORKERS}"

# ------------------------------------------------------------------------------
# 2. Validation Set (5,000 problems, seed 101)
# ------------------------------------------------------------------------------
VAL_DIR="${DATA_DIR}/val"
echo ""
echo "[2/3] Generating Validation Dataset (5,000 instances) in ${VAL_DIR}..."
mkdir -p "${VAL_DIR}"
python3 "${PROCESS_SCRIPT}" \
    --mode all \
    --problem_dir "${VAL_DIR}" \
    --seed 101 \
    --min_n 50 \
    --max_n 150 \
    --min_alpha 2 \
    --max_alpha 8 \
    --n_problem 5000 \
    --solver akmaxsat \
    --weighted \
    --max_weight 10 \
    --n_workers "${N_WORKERS}"

# ------------------------------------------------------------------------------
# 3. Test Set (5,000 problems, seed 101)
# ------------------------------------------------------------------------------
TEST_DIR="${DATA_DIR}/test"
echo ""
echo "[3/3] Generating Test Dataset (5,000 instances) in ${TEST_DIR}..."
mkdir -p "${TEST_DIR}"
python3 "${PROCESS_SCRIPT}" \
    --mode all \
    --problem_dir "${TEST_DIR}" \
    --seed 101 \
    --min_n 50 \
    --max_n 150 \
    --min_alpha 2 \
    --max_alpha 8 \
    --n_problem 5000 \
    --solver akmaxsat \
    --weighted \
    --max_weight 10 \
    --n_workers "${N_WORKERS}"

echo ""
echo "======================================================================"
echo "MAX-2-SAT Dataset Generation Complete!"
echo "Train: ${TRAIN_DIR}"
echo "Val:   ${VAL_DIR}"
echo "Test:  ${TEST_DIR}"
echo "======================================================================"
