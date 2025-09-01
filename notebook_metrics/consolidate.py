import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


COLUMNS_TO_KEEP: List[str] = [
    "tokenizer",
    "vocab_size",
    "fertility",
    "renyi_efficiency_a_1",
    "renyi_efficiency_a_2",
    "renyi_efficiency_a_2_5",
    "unigram_entropy",
    "compression_ratio_global_mean",
    "bigrams_entropy",
    "average_token_rank",
    "corr_chunkability_rt_mean",
    "corr_chunkability_rt_z",
    "corr_chunkability_accuracy",
    "encoding_length",
    #"chars_total",
    "avg_chars_per_token",
    "n_tokens",
    "u_mean",
    "u_median",
    "final_tokens_count",
    "used_in_merges",
    "leading_space_tokens_count",
    "leading_space_tokens_share",
    "AUC_zipf",
    "SLOPE_zipf",
    "POWER_LAW_MAE_zipf",
    "macro_precision",
    "macro_recall",
]


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


# Map legacy column names to the current schema
ALIASES = {
    "u_stats_n_tokens": "n_tokens",
    "macro_precision_morphscore": "macro_precision",
    "macro_recall_morphscore": "macro_recall",
}


def consolidate_csvs(input_dir: Path, output_path: Path) -> Path:
    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        logging.info("No CSV files found in %s; will attempt to include JSON results if available", input_dir)

    logging.info("Found %d CSV files to consolidate", len(csv_files))

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s due to read error: %s", csv_file, exc)
            continue

        # Apply aliasing: rename legacy columns to the new schema if needed
        renaming_map = {old: new for old, new in ALIASES.items() if old in df.columns and new not in df.columns}
        if renaming_map:
            df = df.rename(columns=renaming_map)

        if "tokenizer" not in df.columns:
            logging.warning("Skipping %s because required column 'tokenizer' is missing", csv_file)
            continue

        # Keep only requested columns that are present; add missing ones as NaN for consistent schema
        present_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
        missing_cols = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
        working = df[present_cols].copy()
        for mc in missing_cols:
            working[mc] = pd.NA

        # Reorder columns to the canonical order
        working = working[COLUMNS_TO_KEEP]
        frames.append(working)
        logging.info(
            "Loaded %s with %d rows; present=%d, missing=%d",
            csv_file.name,
            len(working),
            len(present_cols),
            len(missing_cols),
        )

    # Try to include metrics from analysis JSON under <input_dir>/tok_suite
    try:
        json_df = extract_metrics_from_input_dir(input_dir)
        # Ensure canonical order and presence
        present_cols = [c for c in COLUMNS_TO_KEEP if c in json_df.columns]
        missing_cols = [c for c in COLUMNS_TO_KEEP if c not in json_df.columns]
        working = json_df[present_cols].copy()
        for mc in missing_cols:
            working[mc] = pd.NA
        working = working[COLUMNS_TO_KEEP]
        frames.append(working)
        logging.info("Loaded analysis_results.json with %d rows; present=%d, missing=%d", len(working), len(present_cols), len(missing_cols))
    except FileNotFoundError:
        logging.warning("analysis_results.json not found under %s/tok_suite; continuing without JSON", input_dir)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping analysis JSON due to error: %s", exc)

    if not frames:
        raise RuntimeError("No usable CSV files with the required schema were found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Align results by keeping all tokenizers across files and taking first non-null per metric
    aggregated = (
        combined.groupby("tokenizer", as_index=False)
        .agg({col: _first_non_null for col in COLUMNS_TO_KEEP if col != "tokenizer"})
    )

    # Ensure canonical column order
    aggregated = aggregated[[*COLUMNS_TO_KEEP]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    logging.info(
        "Wrote consolidated CSV to %s with %d tokenizers and %d columns",
        output_path,
        len(aggregated),
        len(aggregated.columns),
    )
    return output_path


def extract_metrics_from_analysis_json(json_path: Path) -> pd.DataFrame:
    """
    Parse a tokenizer analysis JSON (analysis_results.json) and return a DataFrame
    with one row per tokenizer containing the metrics in COLUMNS_TO_KEEP.

    Only metrics present in the JSON are populated; others are set to NA.
    """
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r") as f:
        data: Dict[str, Any] = json.load(f)

    def get_nested(dct: Dict[str, Any], keys: List[Any], default: Any = pd.NA):
        cur: Any = dct
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def pick_language(per_language_block: Optional[Dict[str, Any]], preferred: str = "eng_Latn") -> Optional[str]:
        if not isinstance(per_language_block, dict) or not per_language_block:
            return None
        if preferred in per_language_block:
            return preferred
        # Fall back to the first available language key
        return next(iter(per_language_block.keys()), None)

    # Collect the union of tokenizer names appearing under known sections
    sections = [
        "fertility",
        "renyi_efficiency",
        "vocabulary_utilization",
        "compression_ratio",
        "unigram_distribution_metrics",
        "morphscore",
        "cognitive_plausibility",
        "encoding_length",
        "token_length",
        "zipf",
        "vocabulary_tokens",
        "ustat",
    ]
    tokenizer_names = set()
    for section in sections:
        per_tok = get_nested(data, [section, "per_tokenizer"], default={})
        if isinstance(per_tok, dict):
            tokenizer_names.update(per_tok.keys())

    if not tokenizer_names:
        raise ValueError("No per-tokenizer entries found in the provided JSON.")

    rows: List[Dict[str, Any]] = []
    for tok in sorted(tokenizer_names):
        row: Dict[str, Any] = {col: pd.NA for col in COLUMNS_TO_KEEP}
        row["tokenizer"] = tok

        # vocab_size (from vocabulary_utilization per-language block)
        vu_per_lang = get_nested(data, ["vocabulary_utilization", "per_tokenizer", tok, "per_language"], default=None)
        lang_key = pick_language(vu_per_lang)
        if lang_key is not None:
            row["vocab_size"] = get_nested(vu_per_lang, [lang_key, "vocab_size"], default=pd.NA)

        # fertility (global mean)
        row["fertility"] = get_nested(data, ["fertility", "per_tokenizer", tok, "global", "mean"], default=pd.NA)

        # Rényi efficiencies (overall)
        row["renyi_efficiency_a_1"] = get_nested(
            data, ["renyi_efficiency", "per_tokenizer", tok, "renyi_1.0", "overall"], default=pd.NA
        )
        row["renyi_efficiency_a_2"] = get_nested(
            data, ["renyi_efficiency", "per_tokenizer", tok, "renyi_2.0", "overall"], default=pd.NA
        )
        row["renyi_efficiency_a_2_5"] = get_nested(
            data, ["renyi_efficiency", "per_tokenizer", tok, "renyi_2.5", "overall"], default=pd.NA
        )

        # Unigram distribution metrics (per-language)
        udm_per_lang = get_nested(
            data, ["unigram_distribution_metrics", "per_tokenizer", tok, "per_language"], default=None
        )
        udm_lang_key = pick_language(udm_per_lang)
        if udm_lang_key is not None:
            row["unigram_entropy"] = get_nested(udm_per_lang, [udm_lang_key, "unigram_entropy"], default=pd.NA)
            row["average_token_rank"] = get_nested(udm_per_lang, [udm_lang_key, "avg_token_rank"], default=pd.NA)
            row["n_tokens"] = get_nested(udm_per_lang, [udm_lang_key, "total_tokens"], default=pd.NA)

        # Compression ratio (global mean)
        row["compression_ratio_global_mean"] = get_nested(
            data, ["compression_ratio", "per_tokenizer", tok, "global", "mean"], default=pd.NA
        )

        # Cognitive plausibility correlations
        row["corr_chunkability_rt_mean"] = get_nested(
            data, ["cognitive_plausibility", "per_tokenizer", tok, "corr_chunkability_rt_mean"], default=pd.NA
        )
        row["corr_chunkability_rt_z"] = get_nested(
            data, ["cognitive_plausibility", "per_tokenizer", tok, "corr_chunkability_rt_z"], default=pd.NA
        )
        row["corr_chunkability_accuracy"] = get_nested(
            data, ["cognitive_plausibility", "per_tokenizer", tok, "corr_chunkability_accuracy"], default=pd.NA
        )

        # Encoding length (mean)
        row["encoding_length"] = get_nested(
            data, ["encoding_length", "per_tokenizer", tok, "encoding_length", "mean"], default=pd.NA
        )

        # Token length -> avg chars per token (mean)
        row["avg_chars_per_token"] = get_nested(
            data, ["token_length", "per_tokenizer", tok, "character_length", "mean"], default=pd.NA
        )

        # N-gram entropy metrics (global bigram entropy)
        row["bigrams_entropy"] = get_nested(
            data, ["ngram_entropy_metrics", "per_tokenizer", tok, "global_2gram_entropy"], default=pd.NA
        )

        # Zipf metrics (global)
        row["AUC_zipf"] = get_nested(data, ["zipf", "per_tokenizer", tok, "global", "AUC"], default=pd.NA)
        row["SLOPE_zipf"] = get_nested(data, ["zipf", "per_tokenizer", tok, "global", "SLOPE"], default=pd.NA)
        row["POWER_LAW_MAE_zipf"] = get_nested(
            data, ["zipf", "per_tokenizer", tok, "global", "POWER_LAW_MAE"], default=pd.NA
        )

        # Vocabulary tokens stats
        vt_block = get_nested(data, ["vocabulary_tokens", "per_tokenizer", tok, "vocabulary_tokens"], default=None)
        if isinstance(vt_block, dict):
            row["final_tokens_count"] = vt_block.get("final_tokens_count", pd.NA)
            row["used_in_merges"] = vt_block.get("used_in_merges", pd.NA)
            row["leading_space_tokens_count"] = vt_block.get("leading_space_tokens_count", pd.NA)
            row["leading_space_tokens_share"] = vt_block.get("leading_space_tokens_share", pd.NA)

        # U-stat summary
        row["u_mean"] = get_nested(data, ["ustat", "per_tokenizer", tok, "summary", "u_mean"], default=pd.NA)
        row["u_median"] = get_nested(data, ["ustat", "per_tokenizer", tok, "summary", "u_median"], default=pd.NA)

        # chars_total not found in JSON -> remain NA if absent

        # MorphScore macro metrics (per-language)
        ms_per_lang = get_nested(data, ["morphscore", "per_tokenizer", tok, "per_language"], default=None)
        ms_lang_key = pick_language(ms_per_lang)
        if ms_lang_key is not None:
            row["macro_precision"] = get_nested(ms_per_lang, [ms_lang_key, "macro_precision"], default=pd.NA)
            row["macro_recall"] = get_nested(ms_per_lang, [ms_lang_key, "macro_recall"], default=pd.NA)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure canonical column order and presence
    for col in COLUMNS_TO_KEEP:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS_TO_KEEP]
    return df


def extract_metrics_from_input_dir(input_dir: Path) -> pd.DataFrame:
    """
    Convenience wrapper: locate analysis JSON at <input_dir>/tok_suite/analysis_results.json
    and return the metrics DataFrame with COLUMNS_TO_KEEP.
    """
    base = Path(input_dir).expanduser().resolve()
    json_path = (base / "tok_suite" / "analysis_results.json").resolve()
    return extract_metrics_from_analysis_json(json_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate CSVs in a directory, keeping only specified columns and "
            "aligning rows by tokenizer (outer-union across files)."
        )
    )
    parser.add_argument(
        "--dir",
        dest="input_dir",
        type=str,
        default=".",
        help="Directory containing CSV files to consolidate",
    )
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        default=None,
        help="Output CSV file path (default: <dir>/consolidated.csv)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (input_dir / "final_results.csv").resolve()
    )

    consolidate_csvs(input_dir, output_path)


if __name__ == "__main__":
    main()
