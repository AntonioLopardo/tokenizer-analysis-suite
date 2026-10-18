#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
CONFIG_DIR="${DATA_DIR}/configs"
RESULTS_DIR="${DATA_DIR}/multi_lang_results_test"
MORPHSCORE_DATA="${DATA_DIR}/morphscore"

# Print static path variables
echo "DATA_DIR=${DATA_DIR}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "MORPHSCORE_DATA=${MORPHSCORE_DATA}"

mkdir -p "${RESULTS_DIR}"

# Hard-coded set of languages to run
LANGS=("amh")

run_idx=1
for lang_code in "${LANGS[@]}"; do
  LANG_CONFIG="${CONFIG_DIR}/lang_${lang_code}.json"
  TOK_CONFIG="${CONFIG_DIR}/tok_config_${lang_code}.json"

  if [[ ! -f "${LANG_CONFIG}" ]]; then
    echo "[SKIP] Missing language config: ${LANG_CONFIG}"
    continue
  fi
  if [[ ! -f "${TOK_CONFIG}" ]]; then
    echo "[SKIP] Missing tokenizer config: ${TOK_CONFIG}"
    continue
  fi

  OUT_DIR="${RESULTS_DIR}/${lang_code}"
  TOK_SUITE_OUT="${OUT_DIR}/tok_suite"
  mkdir -p "${TOK_SUITE_OUT}"

  echo "[${run_idx}] Running analysis for ${lang_code}"
  echo "  LANG_CONFIG=${LANG_CONFIG}"
  echo "  TOK_CONFIG=${TOK_CONFIG}"
  echo "  TOK_SUITE_OUT=${TOK_SUITE_OUT}"

  python scripts/run_tokenizer_analysis.py \
    --tokenizer-config "${TOK_CONFIG}" \
    --language-config "${LANG_CONFIG}" \
    --output-dir "${TOK_SUITE_OUT}" \
    --verbose \
    --samples-per-lang 25000 \
    --morphscore \
    --morphscore-data "${MORPHSCORE_DATA}"

  echo "Consolidating CSV metrics across: ${OUT_DIR}"
  python notebook_metrics/consolidate.py \
    --dir "${OUT_DIR}" \
    --out "${OUT_DIR}/final_results.csv"

  run_idx=$((run_idx + 1))
done

echo "All analyses completed successfully."