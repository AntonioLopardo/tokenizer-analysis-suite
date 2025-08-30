

from collections import Counter, defaultdict
import math
from typing import Dict, Tuple, List, Optional
# Compute U-statistics summaries for all tokenizers in config
from utils import compute_results_for_tokenizers_config
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd
import json, math, statistics, time
from typing import Any, Dict

def compute_char_and_bigram_counts(
    dataset,
    lowercase: bool = True,
    within_words_only: bool = True,
    valid_char_predicate = None,
) -> Tuple[Counter, Counter, int]:
    """
    Compute character unigrams and adjacent character bigrams from a dataset of text lines.

    Args:
        dataset: HF datasets split or iterable of dicts with key 'text'
        lowercase: if True, lowercase text before counting
        within_words_only: if True, only count adjacent pairs inside words (skip whitespace boundaries)
        valid_char_predicate: optional callable(ch) -> bool to filter characters (e.g., str.isalpha)

    Returns:
        (char_counts, bigram_counts, total_bigrams)
    """
    char_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    total_bigrams: int = 0

    print("[ustat] Counting chars+bigrams …")
    for ex in dataset:
        text = ex.get("text", "") if isinstance(ex, dict) else str(ex)
        if lowercase:
            text = text.lower()

        if within_words_only:
            words = text.split()
        else:
            # Treat the whole line as a sequence, including spaces
            words = [text]

        for w in words:
            if valid_char_predicate is not None:
                w = "".join([c for c in w if valid_char_predicate(c)])
            if not w:
                continue
            # update unigrams
            char_counts.update(w)
            # update bigrams
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                bigram_counts[pair] += 1
                total_bigrams += 1

    print(f"[ustat] Done counting: chars={len(char_counts)} bigrams={len(bigram_counts)} total_bigrams={total_bigrams}")
    return char_counts, bigram_counts, total_bigrams

def compute_char_pmi(
    c1: str,
    c2: str,
    char_counts: Counter,
    bigram_counts: Counter,
    total_bigrams: int,
    log_base: Optional[float] = math.e,
) -> float:
    """
    PMI(c1, c2) = log( f(c1 c2) * T / ( f(c1) * f(c2) ) )

    Args:
        log_base: base of the logarithm; default natural log.

    Returns:
        PMI value (float). Returns -inf if counts are zero.
    """
    f12 = bigram_counts.get((c1, c2), 0)
    f1 = char_counts.get(c1, 0)
    f2 = char_counts.get(c2, 0)

    numerator = f12 * max(total_bigrams, 1)
    denominator = f1 * f2

    if denominator <= 0 or numerator <= 0:
        return float("-inf")

    return math.log(numerator / denominator, log_base) if log_base else math.log(numerator / denominator)

def word_adjacent_pmis(
    word: str,
    char_counts: Counter,
    bigram_counts: Counter,
    total_bigrams: int,
    log_base: Optional[float] = math.e,
) -> List[Tuple[Tuple[str, str], float]]:
    """Return list of ((c_i, c_{i+1}), PMI) for adjacent pairs inside a word."""
    out = []
    if len(word) < 2:
        return []
    for i in range(len(word) - 1):
        c1, c2 = word[i], word[i + 1]
        pmi = compute_char_pmi(c1, c2, char_counts, bigram_counts, total_bigrams, log_base)
        out.append(((c1, c2), pmi))
    return out

def min_adjacent_pmi(
    word: str,
    char_counts: Counter,
    bigram_counts: Counter,
    total_bigrams: int,
    log_base: Optional[float] = math.e,
) -> float:
    """Minimum PMI among all adjacent pairs;"""
    # if len(word) < 2:
    #     return 1.0
    vals = [p for _, p in word_adjacent_pmis(word, char_counts, bigram_counts, total_bigrams, log_base)]
    return float("nan") if not vals else min(vals)

def _normalize_token_for_chars(tok: str, strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ")) -> str:
    for p in strip_prefixes:
        if tok.startswith(p):
            tok = tok[len(p):]
    # Common artifacts to drop
    return tok.replace("Ċ", "").replace("</w>", "")

def token_min_adjacent_pmi_map(
    tokenizer,
    dataset,
    lowercase: bool = False,
    within_words_only: bool = True,
    strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ"),
) -> Tuple[Dict[str, float], float]:
    """
    Returns:
      - dict: token -> U_stat(token) = min adjacent-char PMI inside token
      - float: min U_stat over all tokens (the tokenizer-level minimum)
    """
    char_c, bigram_c, T = compute_char_and_bigram_counts(dataset, lowercase=lowercase, within_words_only=within_words_only)

    vocab = tokenizer.get_vocab()  # token -> id
    token_to_min: Dict[str, float] = {}
    for tok in vocab.keys():
        norm = _normalize_token_for_chars(tok, strip_prefixes)
        val = min_adjacent_pmi(norm, char_c, bigram_c, T)
        token_to_min[tok] = val

    return token_to_min

def entropy_from_counts(counter: Counter, log_base: Optional[float] = math.e) -> float:
    total = sum(counter.values())
    if total <= 0:
        return float("nan")
    H = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            H -= p * (math.log(p, log_base) if log_base else math.log(p))
    return H

from collections import Counter, defaultdict

def token_left_right_entropy_map_set(
    tokenizer,
    dataset,
    lowercase: bool = False,
    within_words_only: bool = True,
    valid_char_predicate = None,
    strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ"),
    drop_if_len_lt1: bool = True,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    # Normalize tokens and build lookup
    vocab = tokenizer.get_vocab()  # token -> id
    norm_to_tokens: Dict[str, List[str]] = defaultdict(list)
    max_len = 1
    for t in vocab.keys():
        nt = _normalize_token_for_chars(t, strip_prefixes)
        if lowercase:
            nt = nt.lower()
        if drop_if_len_lt1 and len(nt) < 1:
            continue
        norm_to_tokens[nt].append(t)
        if len(nt) > max_len:
            max_len = len(nt)

    token_set = set(norm_to_tokens.keys())
    if not token_set:
        return {}, float("nan")

    # Accumulate neighbor counts
    left_counts: Dict[str, Counter] = {nt: Counter() for nt in token_set}
    right_counts: Dict[str, Counter] = {nt: Counter() for nt in token_set}

    def process_text(text: str):
        if lowercase:
            text = text.lower()
        words = text.split() if within_words_only else [text]
        for w in words:
            if valid_char_predicate is not None:
                w = "".join(ch for ch in w if valid_char_predicate(ch))
            if not w:
                continue
            n = len(w)
            for i in range(n):
                # Try all lengths up to max_len
                upto = min(max_len, n - i)
                for k in range(1, upto + 1):
                    s = w[i:i+k]
                    if s in token_set:
                        lch = w[i-1] if i > 0 else None
                        rch = w[i+k] if i + k < n else None
                        if lch is not None:
                            left_counts[s][lch] += 1
                        if rch is not None:
                            right_counts[s][rch] += 1

    print("[ustat] Scanning neighbors for left/right entropies …")
    for ex in dataset:
        text = ex.get("text", "") if isinstance(ex, dict) else str(ex)
        process_text(text)

    # Convert to entropies and map back to original tokens
    per_token: Dict[str, Tuple[float, float, float]] = {}
    global_min = float("inf")
    for nt, toks in norm_to_tokens.items():
        H_left = entropy_from_counts(left_counts[nt])
        H_right = entropy_from_counts(right_counts[nt])
        H_min = min(H_left, H_right) if not (math.isnan(H_left) or math.isnan(H_right)) else float("nan")
        for t in toks:
            per_token[t] = (H_left, H_right, H_min)
        if not math.isnan(H_min):
            global_min = min(global_min, H_min)

    if global_min == float("inf"):
        global_min = float("nan")
    print("[ustat] Finished computing left/right entropies")
    return per_token, global_min

import math
from typing import Dict, Tuple

def token_pmi_plus_entropy_stat_map(
    tokenizer,
    dataset,
    *,
    lam: float = 4.0,
    lowercase: bool = False,
    within_words_only: bool = True,
    strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ"),
    valid_char_predicate=None,
) -> Tuple[Dict[str, float], float]:
    """
    Compute U_stat(w) = min_{(c_i,c_{i+1})⊂w} PMI(c_i,c_{i+1}) + lam * min(H_left(w), H_right(w)),
    for every tokenizer token w.

    Returns:
      - dict: token -> U_stat(token)
      - float: min U_stat over all tokens
    """
    # 1) Dataset-wide character stats for PMI
    char_c, bigram_c, T = compute_char_and_bigram_counts(
        dataset,
        lowercase=lowercase,
        within_words_only=within_words_only,
        valid_char_predicate=valid_char_predicate,
    )

    # 2) Token-level left/right branching entropies
    ent_map, _ = token_left_right_entropy_map_set(
        tokenizer,
        dataset,
        lowercase=lowercase,
        within_words_only=within_words_only,
        valid_char_predicate=valid_char_predicate,
        strip_prefixes=strip_prefixes,
        drop_if_len_lt1=True,
    )

    # 3) Combine per token
    vocab = tokenizer.get_vocab()  # token -> id
    token_to_u: Dict[str, float] = {}
    global_min = float("inf")

    print(f"[ustat] Combining U per token for {len(vocab)} vocab entries …")
    for tok in vocab.keys():
        norm = _normalize_token_for_chars(tok, strip_prefixes)
        if lowercase:
            norm = norm.lower()

        pmi_min = min_adjacent_pmi(norm, char_c, bigram_c, T)
        H_tuple = ent_map.get(tok)
        if H_tuple is None:
            # If we never observed neighbors, leave entropy term as NaN to avoid biasing.
            H_min = float("nan")
        else:
            _, _, H_min = H_tuple

        if math.isnan(pmi_min) or math.isnan(H_min):
            # Skip tokens without well-defined components
            continue

        u_val = pmi_min + lam * H_min
        token_to_u[tok] = u_val
        global_min = min(global_min, u_val)

    if global_min == float("inf"):
        global_min = float("nan")

    print("[ustat] Finished combining U per token")
    return token_to_u, global_min

def _u_stats_from_per_token(per_token_u: Dict[str, float]) -> Dict[str, Any]:
    vals = list(per_token_u.values())
    n = len(vals)
    if n == 0:
        return {
            "n_tokens": 0,
            "u_mean": None,
            "u_median": None,
            "u_stdev": None,
            "u_min": None,
            "u_max": None,
            "u_p1": None,
            "u_p5": None,
            "u_p10": None,
            "u_p25": None,
            "u_p50": None,
            "u_p75": None,
            "u_p90": None,
            "u_p95": None,
            "u_p99": None,
        }
    vals_sorted = sorted(vals)
    def percentile(sorted_values, p):
        k = (len(sorted_values) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
    stats = {
        "n_tokens": n,
        "u_mean": statistics.fmean(vals) if hasattr(statistics, "fmean") else sum(vals) / n,
        "u_median": statistics.median(vals),
        "u_stdev": statistics.stdev(vals) if n > 1 else 0.0,
        "u_min": vals_sorted[0],
        "u_max": vals_sorted[-1],
        "u_p1": percentile(vals_sorted, 1),
        "u_p5": percentile(vals_sorted, 5),
        "u_p10": percentile(vals_sorted, 10),
        "u_p25": percentile(vals_sorted, 25),
        "u_p50": percentile(vals_sorted, 50),
        "u_p75": percentile(vals_sorted, 75),
        "u_p90": percentile(vals_sorted, 90),
        "u_p95": percentile(vals_sorted, 95),
        "u_p99": percentile(vals_sorted, 99),
    }
    return stats


def calculate_u_stats_for_tokenizer(tok: Tokenizer, name: str, *, dataset, save_token_map: bool = False, token_map_dir: Optional[Path] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    print(f"[ustat] {name}: start")
    ds = dataset
    token_map, _ = token_pmi_plus_entropy_stat_map(
        tok,
        ds,
        lam=4.0,
        lowercase=True,
        within_words_only=True,
    )
    # # Optionally save token_map to disk
    # if save_token_map:
    #     out_dir = token_map_dir if token_map_dir is not None else OUT_DIR / f"token_maps_{name}"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     out_path = (out_dir / name).with_suffix(".json")
    #     with open(out_path, "w") as jf:
    #         json.dump(token_map, jf)
    #     tqdm.write(f"[{name}] token_map saved to {out_path}")

    row = _u_stats_from_per_token(token_map)
    row["tokenizer"] = name
    dur = time.perf_counter() - start
    mean_str = "nan" if row["u_mean"] is None else f"{row['u_mean']:.3f}"
    print(f"[ustat] {name}: done in {dur:.1f}s, n_tokens={row['n_tokens']}, u_mean={mean_str}")
    return row