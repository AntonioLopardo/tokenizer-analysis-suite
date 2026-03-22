#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
CONFIG_DIR="${DATA_DIR}/configs"
RESULTS_DIR="${DATA_DIR}/results/fineweb_eng_split_metrics"
MORPHSCORE_DATA="${DATA_DIR}/morphscore"

TOK_CONFIG="${CONFIG_DIR}/tok_config_fineweb_eng.json"
LANG_CONFIG="${CONFIG_DIR}/lang_eng_corrected.json"

echo "DATA_DIR=${DATA_DIR}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "TOK_CONFIG=${TOK_CONFIG}"
echo "LANG_CONFIG=${LANG_CONFIG}"

mkdir -p "${RESULTS_DIR}"

python scripts/run_tokenizer_analysis.py \
  --tokenizer-config "${TOK_CONFIG}" \
  --language-config "${LANG_CONFIG}" \
  --output-dir "${RESULTS_DIR}" \
  --verbose \
  --samples-per-lang 25000 \
  --morphscore \
  --morphscore-data "${MORPHSCORE_DATA}"

echo "Analysis completed. Results in ${RESULTS_DIR}"
