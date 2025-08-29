import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from utils import compute_results_for_tokenizers_config, subset_dataset

# Metric implementations
from metrics_implementations.vocab_metrics import (
    calculate_measure_vocab,
)
from metrics_implementations.minimum_description import (
    calculate_measure_encoding_length,
)
from metrics_implementations.zipfiness import (
    calculate_zipf_metrics_for_dataset,
)
from metrics_implementations.ustat import (
    calculate_u_stats_for_tokenizer,
)
from metrics_implementations.cognitive_plausibility import (
    calculate_measure_chunkability_keuleers,
    keu_wide,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("metrics")

# Ensure Rust tokenizers use internal parallelism by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def _load_text_dataset(text_path: str):
    """Load a HuggingFace text dataset from a local file path."""
    text_path = str(text_path)
    return load_dataset("text", data_files={"train": text_path}, split="train")


def _run_vocab_metrics(config_path: str, out_dir: Path):
    out_csv = out_dir / "vocab_metrics.csv"

    def _calc(tok, name):
        logger.info("[vocab] %s: start", name)
        res = calculate_measure_vocab(tok, name)
        logger.info("[vocab] %s: done", name)
        return res

    df = compute_results_for_tokenizers_config(
        config_path=config_path,
        calculate_measure=_calc,
        sort_by=["vocab_size"] if "vocab_size" in getattr(calculate_measure_vocab, "__annotations__", {}) else None,
        output_csv_path=str(out_csv),
        add_vocab_size=True,
    )
    logger.info("[vocab] group -> %s", out_csv)
    return df


def _run_zipf_metrics(
    config_path: str,
    dataset,
    out_dir: Path,
    *,
    batch_size: int = 8192,
    max_lines: Optional[int] = None,
    base: str = "e",
):
    out_csv = out_dir / "zipf_metrics.csv"

    def _calc(tok, name, *, dataset, batch_size: int, max_lines: Optional[int], base: str):
        logger.info("[zipf] %s: start", name)
        res = calculate_zipf_metrics_for_dataset(
            tok, name, dataset=dataset, batch_size=batch_size , max_lines=max_lines, base=base
        )
        logger.info("[zipf] %s: done", name)
        return res

    df = compute_results_for_tokenizers_config(
        config_path=config_path,
        calculate_measure=_calc,
        sort_by=["vocab_size"],
        output_csv_path=str(out_csv),
        add_vocab_size=True,
        dataset=dataset,
        batch_size=batch_size,
        max_lines=max_lines,
        base=base,
    )
    logger.info("[zipf] group -> %s", out_csv)
    return df


def _run_min_description(
    config_path: str,
    dataset,
    out_dir: Path,
    *,
    batch_size: int = 8192,
):
    out_csv = out_dir / "minimum_description_encoding_lengths.csv"

    def _calc(tok, name, *, dataset, batch_size: int):
        logger.info("[mdl] %s: start", name)
        res = calculate_measure_encoding_length(tok, name, dataset=dataset, batch_size=batch_size)
        logger.info("[mdl] %s: done", name)
        return res

    df = compute_results_for_tokenizers_config(
        config_path=config_path,
        calculate_measure=_calc,
        sort_by=["vocab_size"],
        output_csv_path=str(out_csv),
        add_vocab_size=True,
        dataset=dataset,
        batch_size=batch_size,
    )
    logger.info("[mdl] group -> %s", out_csv)
    return df


def _run_cognitive_plausibility(config_path: str, out_dir: Path):
    out_csv = out_dir / "chunkability_keuleers_correlations.csv"
    forms = keu_wide["Form"].dropna().astype(str).tolist()

    def _calc(tok, name, *, forms):
        logger.info("[chunk] %s: start", name)
        res = calculate_measure_chunkability_keuleers(tok, name, forms=forms)
        logger.info("[chunk] %s: done", name)
        return res

    df = compute_results_for_tokenizers_config(
        config_path=config_path,
        calculate_measure=_calc,
        sort_by=["vocab_size"],
        output_csv_path=str(out_csv),
        add_vocab_size=True,
        forms=forms,
    )
    logger.info("[chunk] group -> %s", out_csv)
    return df


def _run_u_stats(
    config_path: str,
    dataset,
    out_dir: Path,
):
    out_csv = out_dir / "u_stats_summary.csv"
    token_map_root = out_dir

    def _calc(tok, name, *, dataset, token_map_root):
        # Save token maps into a per-tokenizer subdirectory
        subdir = Path(token_map_root) / f"token_maps_{name}"
        logger.info("[u] %s: start", name)
        res = calculate_u_stats_for_tokenizer(
            tok,
            name,
            dataset=dataset,
            save_token_map=True,
            token_map_dir=subdir,
        )
        logger.info("[u] %s: done", name)
        return res

    df = compute_results_for_tokenizers_config(
        config_path=config_path,
        calculate_measure=_calc,
        sort_by=["vocab_size"],
        output_csv_path=str(out_csv),
        add_vocab_size=True,
        dataset=dataset,
        token_map_root=str(token_map_root),
    )
    logger.info("[u] group -> %s", out_csv)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run tokenizer metrics over a config and dataset, saving CSVs.")
    parser.add_argument("--config", "-c", required=True, help="Path to tokenizer config JSON (e.g., configs/tok_alt.json)")
    parser.add_argument("--dataset", "-d", required=True, help="Path to text dataset file (one example per line)")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for CSVs and artifacts")

    # Optional performance knobs
    parser.add_argument("--zipf-batch-size", type=int, default=4096, help="Batch size for Zipf tokenization")
    parser.add_argument("--zipf-max-lines", type=int, default=None, help="Optional cap on lines for Zipf metrics")
    parser.add_argument("--zipf-log-base", choices=["e", "10"], default="e", help="Log base for Zipf metrics")

    parser.add_argument("--mdl-batch-size", type=int, default=8192, help="Batch size for minimum description length")
    parser.add_argument("--subset-n", type=int, default=None, help="Globally shuffle and take only N lines for all metrics")

    args = parser.parse_args()

    config_path = str(Path(args.config).expanduser().resolve())
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and optional global subsampling
    dataset = _load_text_dataset(str(Path(args.dataset).expanduser().resolve()))
    if args.subset_n and args.subset_n > 0:
        logger.info("[global] using subset: %d lines (shuffled)", args.subset_n)
        dataset = subset_dataset(dataset, subset_n=args.subset_n, shuffle=True, seed=42)

    # Run all metric groups
    logger.info("[1/5] Vocab metrics …")
    _run_vocab_metrics(config_path, out_dir)

    logger.info("[2/5] Zipf metrics …")
    _run_zipf_metrics(
        config_path,
        dataset,
        out_dir,
        batch_size=args.zipf_batch_size,
        max_lines=args.zipf_max_lines,
        base=args.zipf_log_base,
    )

    logger.info("[3/5] Minimum description length …")
    _run_min_description(
        config_path,
        dataset,
        out_dir,
        batch_size=args.mdl_batch_size,
    )

    logger.info("[4/5] Cognitive plausibility (chunkability vs Keuleers) …")
    _run_cognitive_plausibility(config_path, out_dir)

    logger.info("[5/5] U-stats (this can take a while) …")
    _run_u_stats(
        config_path,
        dataset,
        out_dir,
    )

    logger.info("Done. Results saved under: %s", out_dir)


if __name__ == "__main__":
    main()


