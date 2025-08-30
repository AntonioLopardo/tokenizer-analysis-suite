import numpy as np
from collections import Counter
from itertools import chain
from tokenizers import Tokenizer

try:
    # SciPy gives Simpson over non-uniform x; if unavailable, we will fallback
    from scipy.integrate import simpson as _scipy_simpson
except Exception:  # noqa: BLE001
    _scipy_simpson = None


def _log_vals(arr: np.ndarray, base: str = "e") -> np.ndarray:
    if base == "10":
        return np.log10(arr)
    return np.log(arr)


def rank_frequency_series(token_id_freq: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ranks (1..V) and counts sorted descending by count."""
    counts = np.array(sorted(token_id_freq.values(), reverse=True), dtype=np.int64)
    ranks = np.arange(1, counts.size + 1, dtype=np.int64)
    return ranks, counts


def auc_loglog(ranks: np.ndarray, counts: np.ndarray, base: str = "e") -> float:
    """Area under log(count) vs log(rank). Uses SciPy Simpson if available, else trapezoid."""
    x = _log_vals(ranks.astype(np.float64), base)
    y = _log_vals(counts.astype(np.float64), base)
    if _scipy_simpson is not None and x.size >= 2:
        return float(_scipy_simpson(y, x=x))
    # Fallback: trapezoid rule
    return float(np.trapz(y, x))


def slope_and_powerlaw_mae(ranks: np.ndarray, counts: np.ndarray, base: str = "e") -> tuple[float, float]:
    """Fit y = b0 + b1*x on log–log; return (slope=b1, MAE to the fitted line)."""
    x = _log_vals(ranks.astype(np.float64), base)
    y = _log_vals(counts.astype(np.float64), base)
    b1, b0 = np.polyfit(x, y, 1)
    pred = b0 + b1 * x
    mae = float(np.mean(np.abs(pred - y)))
    return float(b1), mae


def compute_zipf_metrics(token_id_freq: dict[int, int], base: str = "e") -> dict[str, float]:
    ranks, counts = rank_frequency_series(token_id_freq)
    auc = auc_loglog(ranks, counts, base=base)
    slope, mae = slope_and_powerlaw_mae(ranks, counts, base=base)
    return {"AUC": auc, "SLOPE": slope, "POWER_LAW_MAE": mae}

def count_token_ids_for_dataset(tokenizer: Tokenizer, dataset, batch_size: int = 16384, max_lines: int | None = None) -> Counter:
    vocab_size = tokenizer.get_vocab_size()
    counts = np.zeros(vocab_size, dtype=np.int64)
    buffer: list[str] = []
    seen = 0
    for example in dataset:
        buffer.append(example["text"])
        seen += 1
        if len(buffer) >= batch_size:
            encs = tokenizer.encode_batch(buffer, add_special_tokens=False)
            ids = np.fromiter(chain.from_iterable(enc.ids for enc in encs), dtype=np.int64)
            if ids.size:
                counts += np.bincount(ids, minlength=vocab_size)
            buffer = []
        if max_lines is not None and seen >= max_lines:
            break
    if buffer:
        encs = tokenizer.encode_batch(buffer, add_special_tokens=False)
        ids = np.fromiter(chain.from_iterable(enc.ids for enc in encs), dtype=np.int64)
        if ids.size:
            counts += np.bincount(ids, minlength=vocab_size)
    return Counter({i: int(c) for i, c in enumerate(counts) if c})


def calculate_zipf_metrics_for_dataset(
    tok: Tokenizer,
    name: str,
    *,
    dataset,
    batch_size: int = 4096,
    max_lines: int | None = None,
    base: str = "e",
) -> dict[str, float]:
    """Compute Zipf metrics for a tokenizer on a dataset with batching.

    Returns a dict with the tokenizer name and Zipf-related metrics.
    """
    print(f"[zipf] {name}: start (batch_size={batch_size}, max_lines={max_lines}, base={base})")
    counts = count_token_ids_for_dataset(
        tok, dataset, batch_size=batch_size, max_lines=max_lines
    )
    metrics = compute_zipf_metrics(counts, base=base)
    print(f"[zipf] {name}: done")
    return {
        "tokenizer": name,
        "AUC": float(metrics["AUC"]),
        "SLOPE": float(metrics["SLOPE"]),
        "POWER_LAW_MAE": float(metrics["POWER_LAW_MAE"]),
    }