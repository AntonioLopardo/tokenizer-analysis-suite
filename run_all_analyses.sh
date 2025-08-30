#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
CONFIG_DIR="${DATA_DIR}/configs"
RESULTS_DIR="${DATA_DIR}/results"

TOK_CONFIG="${CONFIG_DIR}/tok_config.json"
LANG_CONFIG="${CONFIG_DIR}/lang_eng.json"
ENG_DATA="${DATA_DIR}/eng_Latn/eng_latn_300mb.txt"
MORPHSCORE_DATA="${DATA_DIR}/morphscore"

TOK_SUITE_OUT="${RESULTS_DIR}/tok_suite"
MORPHSCORE_OUT="${RESULTS_DIR}/morphscore"

# Print all path variables
echo "DATA_DIR=${DATA_DIR}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "TOK_CONFIG=${TOK_CONFIG}"
echo "LANG_CONFIG=${LANG_CONFIG}"
echo "ENG_DATA=${ENG_DATA}"
echo "MORPHSCORE_DATA=${MORPHSCORE_DATA}"
echo "TOK_SUITE_OUT=${TOK_SUITE_OUT}"

mkdir -p "${RESULTS_DIR}" "${TOK_SUITE_OUT}"

echo "[1/4] Running notebook_metrics/metrics.py"
python notebook_metrics/metrics.py \
  --config "${TOK_CONFIG}" \
  --dataset "${ENG_DATA}" \
  --outdir "${RESULTS_DIR}" \
  --subset-n 1000

echo "[2/4] Running scripts/run_tokenizer_analysis.py"
python scripts/run_tokenizer_analysis.py \
  --tokenizer-config "${TOK_CONFIG}" \
  --language-config "${LANG_CONFIG}" \
  --output-dir "${TOK_SUITE_OUT}" \
  --verbose \
  --samples-per-lang 1000 \
  --morphscore \
  --morphscore-data "${MORPHSCORE_DATA}"

# echo "[3/4] Running notebook_metrics/run_morphscore_analysis.py"
# python notebook_metrics/run_morphscore_analysis.py \
#   --tok-config "${TOK_CONFIG}" \
#   --data-dir "${MORPHSCORE_DATA}" \
#   --language-subset eng_Latn \
#   --output-dir "${MORPHSCORE_OUT}"

echo "[4/4] Consolidating CSV metrics"
python notebook_metrics/consolidate.py \
  --dir "${RESULTS_DIR}" \
  --out "${RESULTS_DIR}/final_results.csv"

echo "All analyses completed successfully."


