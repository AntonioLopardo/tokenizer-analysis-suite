from utils import compute_results_for_tokenizers_config
from tokenizers import Tokenizer
from pathlib import Path
import json

def count_final_tokens_bpe(tokenizer: Tokenizer, exclude_special: bool = True):
    obj = json.loads(tokenizer.to_str())
    model = obj.get("model", {})
    if model.get("type") != "BPE" or "merges" not in model:
        raise ValueError("This metric applies only to BPE tokenizers (merges required).")

    merges = model.get("merges", [])
    used_in_merges = set()
    for m in merges:
        # in HF json, merges is usually a list of two-element lists
        if isinstance(m, list) and len(m) >= 2:
            used_in_merges.add(m[0]); used_in_merges.add(m[1])
        elif isinstance(m, str):
            parts = m.split()
            if len(parts) >= 2:
                used_in_merges.add(parts[0]); used_in_merges.add(parts[1])

    try:
        vocab = tokenizer.get_vocab(with_added_tokens=False)
    except TypeError:
        vocab = tokenizer.get_vocab()
    vocab_tokens = set(vocab.keys())

    special_tokens = set()
    if exclude_special:
        for at in obj.get("added_tokens", []):
            if at.get("special"):
                special_tokens.add(at.get("content"))

    final_tokens = vocab_tokens - used_in_merges - special_tokens
    return {
        "vocab_size": len(vocab_tokens),
        "used_in_merges": len(used_in_merges),
        "special_tokens": len(special_tokens),
        "final_tokens_count": len(final_tokens)
    }


def count_tokens_no_leading_space(tokenizer: Tokenizer):
    obj = json.loads(tokenizer.to_str())
    try:
        vocab = tokenizer.get_vocab(with_added_tokens=False)
    except TypeError:
        vocab = tokenizer.get_vocab()
        added = {at.get("content") for at in obj.get("added_tokens", []) if at.get("special")}
        vocab = {tok: idx for tok, idx in vocab.items() if tok not in added}

    tokens = list(vocab.keys())

    markers = set()
    if any(t.startswith("Ġ") for t in tokens):
        markers.add("Ġ")
    if any(t.startswith("▁") for t in tokens):
        markers.add("▁")
    markers.add(" ")

    def has_leading_space_marker(tok: str) -> bool:
        return any(tok.startswith(m) for m in markers)

    no_leading_space_tokens = [t for t in tokens if not has_leading_space_marker(t)]
    return {
        "vocab_size": len(tokens),
        "no_leading_space_tokens_count": len(no_leading_space_tokens),
        "no_leading_space_tokens_share": len(no_leading_space_tokens) / max(1, len(tokens))
    }


def calculate_measure_vocab(tok: Tokenizer, name: str) -> dict:
    print(f"[vocab] {name}: start")
    a = count_final_tokens_bpe(tok)
    b = count_tokens_no_leading_space(tok)
    print(f"[vocab] {name}: done")
    return {
        "tokenizer": name,
        "final_tokens_count": a["final_tokens_count"],
        "used_in_merges": a["used_in_merges"],
        "special_tokens": a["special_tokens"],
        "no_leading_space_tokens_count": b["no_leading_space_tokens_count"],
        "no_leading_space_tokens_share": b["no_leading_space_tokens_share"],
    }