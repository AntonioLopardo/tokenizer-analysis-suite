#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import csv

import importlib.util
from typing import Dict, Any, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

def import_morphscore_class() -> Any:
	here = Path(__file__).resolve()
	module_path = here.parent.parent / "morphscore" / "morphscore.py"
	spec = importlib.util.spec_from_file_location("morphscore_module", str(module_path))
	module = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module.MorphScore


def load_tok_config(tok_cfg_path: str) -> Dict[str, Dict[str, Any]]:
	with open(tok_cfg_path, "r") as f:
		return json.load(f)


def build_tokenizer(entry: Dict[str, Any]):
	cls = entry.get("class")
	if cls == "huggingface":
		model_path = entry["path"]
		return AutoTokenizer.from_pretrained(model_path)
	if cls == "huggingface-tokenizer":
		tokenizer_path = entry["path"]
		tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path,
                                      unk_token="<UNK>",
    sep_token="<SEP>",
    mask_token="<MASK>",
    cls_token="<CLS>")
		return tokenizer

	raise ValueError(f"Unsupported tokenizer class: {cls}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run MorphScore for multiple tokenizers defined in a config JSON.")
	parser.add_argument("--tok-config", required=True, help="Path to tokenizer config JSON (e.g., my_configs/tok.json).")
	parser.add_argument("--data-dir", required=True, help="Path to MorphScore dataset directory.")
	parser.add_argument("--output-dir", required=True, help="Directory to write results.")
	parser.add_argument("--language-subset", nargs="*", default=None, help="Languages or CSV filenames to evaluate. If omitted, uses contents of data_dir.")
	parser.add_argument("--splits", nargs="*", default=["train", "dev", "test"], help="Data splits to consider.")
	parser.add_argument("--by-split", action="store_true", default=False)
	parser.add_argument("--subword-prefix", default="#", help="Default subword prefix; can be overridden per tokenizer via tok-config entry `subword_prefix`.")
	return parser.parse_args()


def ensure_dir(path: str) -> None:
	Path(path).mkdir(parents=True, exist_ok=True)


def export_morphscore_summary_csv(output_dir: str, csv_path: Optional[str] = None, language: str = "english", recursive: bool = False) -> Path:
	"""
	Scan an output directory for per-tokenizer MorphScore result JSONs and
	write a CSV consolidating recall and precision for each tokenizer.

	Args:
		output_dir: Directory containing "*_results.json" files.
		csv_path: Optional explicit output CSV path. If None, writes to
			"morphscore_summary.csv" inside output_dir.
		language: Language key to read within each result JSON (default: "english").
		recursive: If True, search recursively under output_dir.

	Returns:
		Path to the written CSV file.
	"""
	base_dir = Path(output_dir)
	if not base_dir.exists():
		raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

	json_files: List[Path]
	json_files = sorted(base_dir.rglob("*_results.json")) if recursive else sorted(base_dir.glob("*_results.json"))
	if not json_files:
		print(f"[export_morphscore_summary_csv] No result JSONs found in {output_dir}", file=sys.stderr)

	rows: List[List[Any]] = []
	for jf in json_files:
		try:
			with jf.open("r", encoding="utf-8") as f:
				data: Dict[str, Any] = json.load(f)
		except Exception as e:
			print(f"[export_morphscore_summary_csv] Skipping {jf}: failed to read/parse ({e})", file=sys.stderr)
			continue

		name = jf.stem  # tokenizer name portion before .json
		lang_section = data.get(language, {}) if isinstance(data, dict) else {}
		recall = lang_section.get("morphscore_recall")
		precision = lang_section.get("morphscore_precision")
		rows.append([name, recall, precision])

	# Determine output CSV path
	out_csv = Path(csv_path) if csv_path is not None else (base_dir.parent / "morphscore_summary.csv")
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	# Write CSV
	with out_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		header = ["tokenizer_name", f"{language}_morphscore_recall", f"{language}_morphscore_precision"]
		writer.writerow(header)
		for row in rows:
			writer.writerow(row)

	return out_csv


def main():
	args = parse_args()
	ensure_dir(args.output_dir)

	MorphScore = import_morphscore_class()
	tok_cfg = load_tok_config(args.tok_config)

	for name, entry in tok_cfg.items():
		print(f"=== Running MorphScore for {name} ===")
		try:
			tokenizer = build_tokenizer(entry)
		except Exception as e:
			print(f"[{name}] Failed to build tokenizer: {e}")
			continue

		subword_prefix = entry.get("subword_prefix", args.subword_prefix)

		ms = MorphScore(
			data_dir=args.data_dir,
			language_subset=["english"],
			splits=args.splits,
			by_split=args.by_split,
			by_pos=False,
			subword_prefix=subword_prefix,
		)

		try:
			results = ms.eval(tokenizer, return_df=False)
		except Exception as e:
			print(f"[{name}] Evaluation failed: {e}")
			continue

		out_json = Path(args.output_dir) / f"{name}_results.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(results, f, indent=2, ensure_ascii=False)
		print(f"[{name}] Wrote {out_json}")

	# Export consolidated summary CSV across all tokenizer results
	try:
		out_csv = export_morphscore_summary_csv(args.output_dir)
		print(f"[summary] Wrote {out_csv}")
	except Exception as e:
		print(f"[summary] Failed to export summary CSV: {e}", file=sys.stderr)
if __name__ == "__main__":
	main()