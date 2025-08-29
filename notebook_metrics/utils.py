from collections import Counter
from datasets import load_dataset
from tokenizers import Tokenizer
import json
import re
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List, Union
import pandas as pd
from tqdm.auto import tqdm

eng_path = "/Users/antoniolopardo/Documents/SOAR-alt/data/eng_Latn/eng_latn_300mb.txt"
sample_tokenizer_path = "/Users/antoniolopardo/Documents/SOAR-alt/data/eng_montok/bpe_eng_latn_6144_300mb_unscaled.json"

# Load the local text file; each line becomes one example
eng_ds = load_dataset("text", data_files={"train": eng_path}, split="train")
    
    
def load_eng_dataset():
    """Load the English dataset."""
    return eng_ds

def load_sample_tokenizer():
    return Tokenizer.from_file(sample_tokenizer_path)

def load_tokenized_data(tokenizer, dataset, batch_size=4096):
    """Return a list of encodings for each line in the dataset."""
    encodings_out = []
    buffer = []
    for example in dataset:
        buffer.append(example["text"])
        if len(buffer) >= batch_size:
            encodings_out.extend(
                tokenizer.encode_batch(buffer, add_special_tokens=False)
            )
            buffer = []

    if buffer:
        encodings_out.extend(
            tokenizer.encode_batch(buffer, add_special_tokens=False)
        )

    return encodings_out
        
def compute_token_id_frequencies(encodings):
    """Given a list of Encoding objects, return a Counter of token IDs."""
    freq = Counter()
    for enc in encodings:
        freq.update(enc.ids)
    return freq


def subset_dataset(dataset, subset_n: Optional[int] = None, shuffle: bool = True, seed: int = 42):
    """Return a (optionally shuffled) subset of the dataset of at most subset_n lines.

    If subset_n is None or <= 0, returns dataset (shuffled if shuffle=True).
    """
    if subset_n is None or subset_n <= 0:
        if shuffle:
            tqdm.write("Subsampling disabled; returning shuffled full dataset")
            return dataset.shuffle(seed=seed)
        return dataset
    _ds = dataset.shuffle(seed=seed) if shuffle else dataset
    n = min(subset_n, len(_ds))
    tqdm.write(f"Subsampling: taking {n} lines (shuffle={shuffle}, seed={seed})")
    return _ds.select(range(n))


# -----------------------------
# Generic utilities for configs
# -----------------------------

def _load_tokenizer_from_config_entry(entry: Dict[str, Any]) -> Tokenizer:
    """Load a tokenizer from a config entry.

    Expected schema per entry:
        { "class": "huggingface-tokenizer", "path": "/abs/path/to/tokenizer.json" }
    """
    if entry.get("class") != "huggingface-tokenizer":
        raise ValueError(f"Unsupported tokenizer class: {entry.get('class')}")
    path = entry.get("path")
    if not path:
        raise ValueError("Tokenizer config entry is missing required 'path' field")
    return Tokenizer.from_file(path)


def _extract_vocab_size_from_name(name: str, pattern: str = r"_(\d+)_300mb") -> Optional[int]:
    """Extract an integer vocabulary size from a tokenizer name using regex pattern.

    Default pattern captures the number immediately before "_300mb".
    Returns None if no match is found.
    """
    m = re.search(pattern, name)
    return int(m.group(1)) if m else None


def compute_results_for_tokenizers_config(
    config_path: str,
    calculate_measure: Callable[[Tokenizer, str], Union[Dict[str, Any], pd.Series]],
    sort_by: Optional[List[str]] = None,
    output_csv_path: Optional[str] = None,
    add_vocab_size: bool = False,
    vocab_pattern: str = r"_(\d+)_300mb",
    **kwargs: Any,
) -> pd.DataFrame:
    """Run a measurement function for each tokenizer defined in a config.

    Args:
        config_path: Path to a JSON config mapping tokenizer names to entries.
        calculate_measure: Callable invoked as calculate_measure(tokenizer, name, **kwargs)
            and returning a dict-like row of results. If the returned object does
            not contain a 'tokenizer' key, one will be added automatically.
        sort_by: Optional list of columns to sort the resulting DataFrame by.
        output_csv_path: If provided, results are saved to this CSV path.
        add_vocab_size: If True, add a 'vocab_size' column parsed from the
            tokenizer name using 'vocab_pattern'.
        vocab_pattern: Regex to extract vocab size from tokenizer name.
        **kwargs: Extra arguments forwarded to calculate_measure.

    Returns:
        pd.DataFrame with one row per tokenizer and the returned metrics.
    """
    config_path = str(config_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    results: List[Dict[str, Any]] = []
    items = list(cfg.items())
    for name, entry in items:
        tqdm.write(f"[{name}] load tokenizer")
        tok = _load_tokenizer_from_config_entry(entry)
        tqdm.write(f"[{name}] calculate measure")
        row = calculate_measure(tok, name, **kwargs)
        # Normalize to dict
        if isinstance(row, pd.Series):
            row = row.to_dict()
        if not isinstance(row, dict):
            raise TypeError("calculate_measure must return a dict or pandas.Series")
        if "tokenizer" not in row:
            row["tokenizer"] = name
        if add_vocab_size and "vocab_size" not in row:
            vs = _extract_vocab_size_from_name(name, pattern=vocab_pattern)
            row["vocab_size"] = vs
        results.append(row)
        tqdm.write(f"[{name}] done")

    df = pd.DataFrame(results)

    if sort_by:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        df = df.sort_values(by=sort_by).reset_index(drop=True)

    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df