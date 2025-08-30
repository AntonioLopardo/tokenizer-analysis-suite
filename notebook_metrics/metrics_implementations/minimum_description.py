
from utils import compute_results_for_tokenizers_config
from tokenizers import Tokenizer
from pathlib import Path

def calculate_measure_encoding_length(tok: Tokenizer, name: str, *, dataset, batch_size: int = 2048) -> dict:
    """Return total encoding length (sum of token counts) for the full dataset.
    Uses batching to avoid high memory usage.
    """
    print(f"[mdl] {name}: start (batch_size={batch_size})")
    total_tokens = 0
    total_chars = 0
    buffer = []
    for ex in dataset:
        text = ex.get("text", "")
        buffer.append(text)
        total_chars += len(text)
        if len(buffer) >= batch_size:
            encs = tok.encode_batch(buffer, add_special_tokens=False)
            total_tokens += sum(len(e.ids) for e in encs)
            buffer = []
    if buffer:
        encs = tok.encode_batch(buffer, add_special_tokens=False)
        total_tokens += sum(len(e.ids) for e in encs)
    print(f"[mdl] {name}: done (tokens={total_tokens}, chars={total_chars})")
    return {
        "tokenizer": name,
        "encoding_length": int(total_tokens),
        "chars_total": int(total_chars),
        "avg_chars_per_token": (total_chars/ max(1, total_tokens)) if total_tokens > 0 else None,
    }