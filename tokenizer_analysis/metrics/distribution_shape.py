"""
Zipfian distribution metrics using unified TokenizedData interface.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider

logger = logging.getLogger(__name__)

from scipy.integrate import simpson


class DistributionShapeMetrics(BaseMetrics):
    """
    Zipfian metrics over token rank-frequency distributions.

    Definitions (source: provided Rank-frequency metrics figure):
    - Rank-frequency AUC (AUC): Area under the curve of y=log(count) vs x=log(rank),
      computed using Simpson's rule.
    - Slope of linear function (SLOPE): The slope m from the linear fit
      f(x) = b + m x on the log-log points (x=log(rank), y=log(count)),
      approximating Zipf's law.
    - Deviation from linear function (POWER_LAW_MAE): The mean absolute error
      (1/n) ∑|b + m x_i − y_i| between the fitted line and the log-log points,
      quantifying closeness to a Zipfian power law.
    """

    def __init__(self,
                 input_provider: InputProvider,
                 normalization_config: Optional[Any] = None,
                 language_metadata: Optional[Any] = None):
        """
        Initialize Zipf metrics.
        Args:
            input_provider: InputProvider instance
            normalization_config: Unused here (kept for API compatibility)
            language_metadata: Optional language metadata for grouping
        """
        super().__init__(input_provider)
        self.language_metadata = language_metadata
        self._log_base = "e"  # "e" or "10"

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute Zipfian distribution metrics.
        Returns:
            {
              'zipf': {
                 'per_tokenizer': {
                    tok: {
                      'global': {'AUC': float, 'SLOPE': float, 'POWER_LAW_MAE': float},
                      'per_language': { lang: {... as above ...} }
                    }, ...
                 },
                 'pairwise_comparisons': {
                    'slope': {...}, 'auc': {...}, 'power_law_mae': {...}
                 },
                 'metadata': {...}
              }
            }
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results = {
            'zipf': {
                'per_tokenizer': {},
                'pairwise_comparisons': {
                    'slope': {},
                    'auc': {},
                    'power_law_mae': {}
                },
                'metadata': {
                    'description': 'Zipfian metrics over token frequency rank distributions',
                    'log_base': self._log_base,
                    'definitions': {
                        'source': 'See arXiv:2506.03101 (https://arxiv.org/pdf/2506.03101)',
                        'AUC': 'Area under log(count) vs log(rank) using Simpson’s rule.',
                        'SLOPE': 'Slope m of linear fit y = b + m x on log-log.',
                        'POWER_LAW_MAE': 'Mean absolute error to the fitted line on log-log.'
                    },
                    'metrics': ['AUC', 'SLOPE', 'POWER_LAW_MAE']
                }
            }
        }

        global_slopes: Dict[str, float] = {}
        global_aucs: Dict[str, float] = {}
        global_maes: Dict[str, float] = {}

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]

            # Global metrics
            token_id_freq = self._count_token_ids(tok_data)
            global_metrics = self._compute_zipf_metrics(token_id_freq, base=self._log_base)

            # Per-language metrics
            per_lang_metrics: Dict[str, Dict[str, float]] = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            for language, lang_data in lang_groups.items():
                lang_freq = self._count_token_ids(lang_data)
                lang_metrics = self._compute_zipf_metrics(lang_freq, base=self._log_base)
                per_lang_metrics[language] = lang_metrics

            results['zipf']['per_tokenizer'][tok_name] = {
                'global': global_metrics,
                'per_language': per_lang_metrics
            }

            global_slopes[tok_name] = float(global_metrics.get('SLOPE', 0.0))
            global_aucs[tok_name] = float(global_metrics.get('AUC', 0.0))
            global_maes[tok_name] = float(global_metrics.get('POWER_LAW_MAE', 0.0))

        # Pairwise comparisons across tokenizers for each metric
        if len(global_slopes) >= 2:
            results['zipf']['pairwise_comparisons']['slope'] = self.compute_pairwise_comparisons(global_slopes, 'zipf_slope')
        if len(global_aucs) >= 2:
            results['zipf']['pairwise_comparisons']['auc'] = self.compute_pairwise_comparisons(global_aucs, 'zipf_auc')
        if len(global_maes) >= 2:
            results['zipf']['pairwise_comparisons']['power_law_mae'] = self.compute_pairwise_comparisons(global_maes, 'zipf_power_law_mae')

        return results

    def _count_token_ids(self, tokenized_data: List[TokenizedData]) -> Dict[int, int]:
        """Flatten tokens and count frequencies."""
        all_tokens = TokenizedDataProcessor.flatten_all_tokens(tokenized_data)
        return dict(Counter(all_tokens))

    def _log_vals(self, arr: np.ndarray, base: str = "e") -> np.ndarray:
        if base == "10":
            return np.log10(arr)
        return np.log(arr)

    def _rank_frequency_series(self, token_id_freq: Dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ranks (1..V) and counts sorted descending by count."""
        if not token_id_freq:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        counts = np.array(sorted(token_id_freq.values(), reverse=True), dtype=np.int64)
        ranks = np.arange(1, counts.size + 1, dtype=np.int64)
        return ranks, counts

    def _auc_loglog(self, ranks: np.ndarray, counts: np.ndarray, base: str = "e") -> float:
        """Area under log(count) vs log(rank) using Simpson’s rule."""
        if ranks.size < 2 or counts.size < 2:
            return 0.0
        x = self._log_vals(ranks.astype(np.float64), base)
        y = self._log_vals(counts.astype(np.float64), base)
        return float(simpson(y, x=x))

    def _slope_and_powerlaw_mae(self, ranks: np.ndarray, counts: np.ndarray, base: str = "e") -> tuple[float, float]:
        """Fit y = b + m*x on log-log; return (slope=m, MAE to the fitted line)."""
        if ranks.size < 2 or counts.size < 2:
            return 0.0, 0.0
        x = self._log_vals(ranks.astype(np.float64), base)
        y = self._log_vals(counts.astype(np.float64), base)
        m, b = np.polyfit(x, y, 1)
        pred = b + m * x
        mae = float(np.mean(np.abs(pred - y)))
        return float(m), mae

    def _compute_zipf_metrics(self, token_id_freq: Dict[int, int], base: str = "e") -> Dict[str, float]:
        ranks, counts = self._rank_frequency_series(token_id_freq)
        if ranks.size == 0 or counts.size == 0:
            return {"AUC": 0.0, "SLOPE": 0.0, "POWER_LAW_MAE": 0.0}
        auc = self._auc_loglog(ranks, counts, base=base)
        slope, mae = self._slope_and_powerlaw_mae(ranks, counts, base=base)
        return {"AUC": auc, "SLOPE": slope, "POWER_LAW_MAE": mae}