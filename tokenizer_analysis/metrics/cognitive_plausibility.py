"""
Cognitive plausibility metrics: chunkability vs human measures (Keuleers et al., 2012).

This module computes the correlation between tokenizer chunkability and human
lexical decision performance (reaction time and accuracy) using the Keuleers
2012 dataset. It integrates with the unified TokenizedData framework by
accessing tokenizer wrappers from the input provider to encode words.
"""

from typing import Dict, List, Any, Optional
import os
import math
import logging
import numpy as np
import pandas as pd

from .base import BaseMetrics
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider

logger = logging.getLogger(__name__)


class CognitivePlausibilityMetrics(BaseMetrics):
    """
    Compute correlations between tokenizer chunkability and human measures
    (RT mean, RT z-score, accuracy) from Keuleers et al. (2012).
    """

    def __init__(
        self,
        input_provider: InputProvider,
        normalization_config: Optional[Any] = None,
        language_metadata: Optional[Any] = None
    ):
        super().__init__(input_provider)
        self.language_metadata = language_metadata
        self._keuleers_df: Optional[pd.DataFrame] = None
        self._keuleers_path: Optional[str] = None
        
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        # We rely on tokenizer wrappers (encode) rather than pre-tokenized data here.
        keu = self._load_keuleers()
        forms = keu["Form"].astype(str).tolist()

        results: Dict[str, Any] = {
            "cognitive_plausibility": {
                "per_tokenizer": {},
                "metadata": {
                    "description": "Correlation between tokenizer chunkability and human lexical decision measures.",
                    "dataset": "Keuleers et al. (2012) Lexical Decision",
                    "file": self._keuleers_path,
                    "definitions": {
                        "chunkability": "1 - (# tokens produced by tokenizer for a word) / (number of characters in the word)",
                        "corr_chunkability_rt_mean": "Pearson correlation between chunkability and reaction time (mean).",
                        "corr_chunkability_rt_z": "Pearson correlation between chunkability and reaction time (z-score).",
                        "corr_chunkability_accuracy": "Pearson correlation between chunkability and accuracy percentage.",
                    },
                    "metrics": [
                        "corr_chunkability_rt_mean",
                        "corr_chunkability_rt_z",
                        "corr_chunkability_accuracy",
                        "n_words",
                    ],
                },
            }
        }

        get_tok = getattr(self.input_provider, "get_tokenizer", None)
        if not callable(get_tok):
            logger.warning(
                "Input provider does not expose get_tokenizer; cognitive plausibility requires raw encoding."
            )
            return results

        for tok_name in self.tokenizer_names:
            try:
                tok_wrapper = get_tok(tok_name)
            except Exception as e:
                logger.warning(f"Cannot fetch tokenizer '{tok_name}': {e}")
                continue

            # Ensure we can encode words
            can_encode = True
            try:
                can_encode = tok_wrapper.can_encode()
            except Exception:
                pass

            if not can_encode:
                logger.info(f"Tokenizer '{tok_name}' cannot encode raw text; skipping.")
                continue

            metrics = self._compute_for_tokenizer(tok_wrapper, tok_name, forms)
            results["cognitive_plausibility"]["per_tokenizer"][tok_name] = metrics

        return results

    def _get_keuleers_path(self) -> str:
        # Resolve repo root from this file: tokenizer_analysis/metrics/ -> ../..
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(repo_root, "data", "lexicon", "keuleers2012_lexicaldecision.csv")

    def _load_keuleers(self) -> pd.DataFrame:
        if self._keuleers_df is not None:
            return self._keuleers_df

        csv_path = self._get_keuleers_path()
        self._keuleers_path = csv_path

        keu_long = pd.read_csv(csv_path)
        df = keu_long.copy()
        df["Metric"] = df["Variable_ID"].str.replace(
            "Keuleers-2012-LexicalDecision-", "", regex=False
        )
        df = df.pivot_table(
            index="Form", columns="Metric", values="Value", aggfunc="mean"
        ).reset_index()
        df = df.rename(
            columns={
                "ENGLISH_RT_MEAN": "rt_mean",
                "ENGLISH_RT_ZSCORE": "rt_z",
                "ENGLISH_ACCURACY_PERCENTAGE": "accuracy",
            }
        )
        self._keuleers_df = df
        return df

    @staticmethod
    def _compute_chunkability(tokenizer_wrapper, word: str) -> Optional[float]:
        if not isinstance(word, str) or len(word) == 0:
            return None
        try:
            tokens = tokenizer_wrapper.encode(word)
            num_tokens = len(tokens)
            num_chars = len(word)
            if num_chars == 0:
                return None
            return 1.0 - (num_tokens / num_chars)
        except Exception as e:
            logger.debug(f"Failed to encode word '{word}': {e}")
            return None

    @staticmethod
    def _is_valid_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    def _compute_for_tokenizer(self, tok_wrapper, tok_name: str, forms: List[str]) -> Dict[str, Any]:
        chunk_map = {w: self._compute_chunkability(tok_wrapper, w) for w in forms}
        chunk_df = pd.DataFrame(
            {"Form": list(chunk_map.keys()), "chunkability": list(chunk_map.values())}
        )

        keu = self._load_keuleers()
        df = keu.merge(chunk_df, on="Form", how="inner").dropna(
            subset=["chunkability", "rt_mean", "rt_z", "accuracy"]
        )

        if df.empty:
            return {
                "tokenizer": tok_name,
                "corr_chunkability_rt_mean": np.nan,
                "corr_chunkability_rt_z": np.nan,
                "corr_chunkability_accuracy": np.nan,
                "n_words": 0,
            }

        corr_rt = float(df[["chunkability", "rt_mean"]].corr().iloc[0, 1])
        corr_z = float(df[["chunkability", "rt_z"]].corr().iloc[0, 1])
        corr_acc = float(df[["chunkability", "accuracy"]].corr().iloc[0, 1])

        return {
            "tokenizer": tok_name,
            "corr_chunkability_rt_mean": corr_rt,
            "corr_chunkability_rt_z": corr_z,
            "corr_chunkability_accuracy": corr_acc,
            "n_words": int(len(df)),
        }