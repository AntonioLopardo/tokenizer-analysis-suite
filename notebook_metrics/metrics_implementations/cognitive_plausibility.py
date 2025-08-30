from tokenizers import Tokenizer
import pandas as pd

KEULEERS_PATH = "data/lexicon/keuleers2012_lexicaldecision.csv"

keu_long = pd.read_csv(KEULEERS_PATH)
keu_wide = keu_long.copy()
keu_wide["Metric"] = keu_wide["Variable_ID"].str.replace("Keuleers-2012-LexicalDecision-", "", regex=False)
keu_wide = keu_wide.pivot_table(index="Form", columns="Metric", values="Value", aggfunc="mean").reset_index()
keu_wide = keu_wide.rename(columns={
    "ENGLISH_RT_MEAN": "rt_mean",
    "ENGLISH_RT_ZSCORE": "rt_z",
    "ENGLISH_ACCURACY_PERCENTAGE": "accuracy",
})

def compute_chunkability_for_word(tokenizer, word: str) -> float:
    if not isinstance(word, str) or len(word) == 0:
        return None
    enc = tokenizer.encode(word, add_special_tokens=False)
    num_tokens = len(enc.ids)
    num_chars = len(word)
    if num_chars == 0:
        return None
    return 1.0 - (num_tokens / num_chars)


def calculate_measure_chunkability_keuleers(tok: Tokenizer, name: str, *, forms: list[str]) -> dict:
    chunk_map = {w: compute_chunkability_for_word(tok, w) for w in forms}
    chunk_df = pd.DataFrame({"Form": list(chunk_map.keys()), "chunkability": list(chunk_map.values())})
    df = keu_wide.merge(chunk_df, on="Form", how="inner").dropna(subset=["chunkability","rt_mean","rt_z","accuracy"])[:]
    corr_rt = df[["chunkability","rt_mean"]].corr().iloc[0,1]
    corr_z  = df[["chunkability","rt_z"]].corr().iloc[0,1]
    corr_acc = df[["chunkability","accuracy"]].corr().iloc[0,1]
    return {
        "tokenizer": name,
        "corr_chunkability_rt_mean": float(corr_rt),
        "corr_chunkability_rt_z": float(corr_z),
        "corr_chunkability_accuracy": float(corr_acc),
        "n_words": int(len(df)),
    }