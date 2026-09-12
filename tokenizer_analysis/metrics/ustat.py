"""
U-statistics metric (adjacent-char PMI + branching entropy) for tokenizer vocab.

This implements a tokenizer-level statistic inspired by the notebook implementation
in notebook_metrics/metrics_implementations/ustat.py. For each tokenizer token,
we compute U(token) = min_adjacent_char_PMI(token) + lam * min(H_left(token), H_right(token)),
where character statistics are estimated from the available input texts.

The metric summarizes per-token U values into aggregate statistics per tokenizer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from .base import BaseMetrics, TokenizedDataProcessor

logger = logging.getLogger(__name__)


class UStatMetrics(BaseMetrics):
    """
    Compute U-statistics summaries per tokenizer based on input texts and tokenizer vocab.

    Requirements:
    - Tokenizer must expose get_vocab() -> Dict[str, int]
    - Input data should include raw texts in TokenizedData.text (available in raw mode)
    """

    def __init__(self,
                 input_provider,
                 lam: float = 4.0,
                 lowercase: bool = True,
                 within_words_only: bool = True,
                 strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ"),
                 valid_char_predicate=None):
        super().__init__(input_provider)
        self.lam = lam
        self.lowercase = lowercase
        self.within_words_only = within_words_only
        self.strip_prefixes = strip_prefixes
        self.valid_char_predicate = valid_char_predicate

    def compute(self, tokenized_data: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results: Dict[str, Any] = {
            'ustat': {
                'per_tokenizer': {},
                'metadata': {
                    'description': 'U-statistics summaries computed over tokenizer vocab using input texts',
                    'definition': 'U(token) = min_adjacent_char_PMI(token) + lam * min(H_left, H_right)',
                    'lam': self.lam,
                    'lowercase': self.lowercase,
                    'within_words_only': self.within_words_only,
                }
            }
        }

        get_tok = getattr(self.input_provider, 'get_tokenizer', None)
        if not callable(get_tok):
            logger.warning("Input provider does not expose get_tokenizer; UStat requires tokenizer vocab access.")
            return results

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            texts = TokenizedDataProcessor.extract_texts(tokenized_data[tok_name])
            if not texts:
                logger.info(f"No raw texts available for tokenizer '{tok_name}'; skipping UStat.")
                continue

            try:
                tok_wrapper = get_tok(tok_name)
            except Exception as e:
                logger.warning(f"Cannot fetch tokenizer '{tok_name}': {e}")
                continue

            # Access vocabulary
            vocab: Optional[Dict[str, int]] = None
            try:
                if hasattr(tok_wrapper, 'get_vocab'):
                    vocab = tok_wrapper.get_vocab()
            except Exception as e:
                logger.warning(f"Tokenizer '{tok_name}' get_vocab() failed: {e}")

            if not vocab:
                logger.info(f"Tokenizer '{tok_name}' has no accessible vocabulary; skipping UStat.")
                continue

            # Compute per-token U values
            per_token_u, global_min = self._token_pmi_plus_entropy_stat_map(
                vocab_keys=list(vocab.keys()),
                texts=texts,
                lam=self.lam,
                lowercase=self.lowercase,
                within_words_only=self.within_words_only,
                strip_prefixes=self.strip_prefixes,
                valid_char_predicate=self.valid_char_predicate,
            )

            token_id_counts: Counter = Counter()
            for td in tokenized_data[tok_name]:
                token_id_counts.update(td.tokens)

            token_freq: Dict[str, int] = {}
            for token_str in per_token_u:
                tid = vocab.get(token_str)
                if tid is not None:
                    token_freq[token_str] = token_id_counts.get(tid, 0)

            summary = self._u_stats_from_per_token(per_token_u, token_freq)
            results['ustat']['per_tokenizer'][tok_name] = {
                'summary': summary,
                'min_u': global_min,
                'vocab_size': len(vocab)
            }

        return results

    # ------------------------- Internal helpers -------------------------

    @staticmethod
    def _normalize_token_for_chars(token: str, strip_prefixes: Tuple[str, ...]) -> str:
        normalized = token
        for prefix in strip_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized.replace("Ċ", "").replace("</w>", "")

    @staticmethod
    def _compute_char_and_bigram_counts(texts: List[str],
                                        lowercase: bool,
                                        within_words_only: bool,
                                        valid_char_predicate=None) -> Tuple[Counter, Counter, int]:
        char_counts: Counter = Counter()
        bigram_counts: Counter = Counter()
        total_bigrams: int = 0

        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            if lowercase:
                text = text.lower()
            words = text.split() if within_words_only else [text]
            for word in words:
                if valid_char_predicate is not None:
                    word = "".join([c for c in word if valid_char_predicate(c)])
                if not word:
                    continue
                char_counts.update(word)
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    bigram_counts[pair] += 1
                    total_bigrams += 1

        return char_counts, bigram_counts, total_bigrams

    @staticmethod
    def _entropy_from_counts(counter: Counter, log_base: Optional[float] = math.e) -> float:
        total = sum(counter.values())
        if total <= 0:
            return float('nan')
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * (math.log(p, log_base) if log_base else math.log(p))
        return entropy

    def _token_left_right_entropy_map(self,
                                      vocab_tokens: List[str],
                                      texts: List[str],
                                      lowercase: bool,
                                      within_words_only: bool,
                                      valid_char_predicate=None,
                                      strip_prefixes: Tuple[str, ...] = ("##", "▁", "Ġ"),
                                      drop_if_len_lt1: bool = True) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
        norm_to_tokens: Dict[str, List[str]] = defaultdict(list)
        max_len = 1
        for token in vocab_tokens:
            norm = self._normalize_token_for_chars(token, strip_prefixes)
            if lowercase:
                norm = norm.lower()
            if drop_if_len_lt1 and len(norm) < 1:
                continue
            norm_to_tokens[norm].append(token)
            if len(norm) > max_len:
                max_len = len(norm)

        token_set = set(norm_to_tokens.keys())
        if not token_set:
            return {}, float('nan')

        left_counts: Dict[str, Counter] = {nt: Counter() for nt in token_set}
        right_counts: Dict[str, Counter] = {nt: Counter() for nt in token_set}

        def process_text(text: str):
            if lowercase:
                text = text.lower()
            words = text.split() if within_words_only else [text]
            for word in words:
                if valid_char_predicate is not None:
                    word = "".join(ch for ch in word if valid_char_predicate(ch))
                if not word:
                    continue
                n = len(word)
                for i in range(n):
                    upto = min(max_len, n - i)
                    for k in range(1, upto + 1):
                        s = word[i:i + k]
                        if s in token_set:
                            lch = word[i - 1] if i > 0 else None
                            rch = word[i + k] if i + k < n else None
                            if lch is not None:
                                left_counts[s][lch] += 1
                            if rch is not None:
                                right_counts[s][rch] += 1

        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            process_text(text)

        per_token_entropy: Dict[str, Tuple[float, float, float]] = {}
        global_min = float('inf')
        for nt, toks in norm_to_tokens.items():
            h_left = self._entropy_from_counts(left_counts[nt])
            h_right = self._entropy_from_counts(right_counts[nt])
            h_min = min(h_left, h_right) if not (math.isnan(h_left) or math.isnan(h_right)) else float('nan')
            for t in toks:
                per_token_entropy[t] = (h_left, h_right, h_min)
            if not math.isnan(h_min):
                global_min = min(global_min, h_min)

        if global_min == float('inf'):
            global_min = float('nan')
        return per_token_entropy, global_min

    def _token_pmi_plus_entropy_stat_map(self,
                                         vocab_keys: List[str],
                                         texts: List[str],
                                         *,
                                         lam: float,
                                         lowercase: bool,
                                         within_words_only: bool,
                                         strip_prefixes: Tuple[str, ...],
                                         valid_char_predicate=None) -> Tuple[Dict[str, float], float]:
        # 1) Dataset-wide character stats for PMI
        char_counts, bigram_counts, total_bigrams = self._compute_char_and_bigram_counts(
            texts, lowercase=lowercase, within_words_only=within_words_only, valid_char_predicate=valid_char_predicate
        )

        # 2) Token-level left/right branching entropies
        ent_map, _ = self._token_left_right_entropy_map(
            vocab_tokens=vocab_keys,
            texts=texts,
            lowercase=lowercase,
            within_words_only=within_words_only,
            valid_char_predicate=valid_char_predicate,
            strip_prefixes=strip_prefixes,
            drop_if_len_lt1=True,
        )

        # 3) Combine per token
        token_to_u: Dict[str, float] = {}
        global_min = float('inf')

        for token in vocab_keys:
            norm = self._normalize_token_for_chars(token, strip_prefixes)
            if lowercase:
                norm = norm.lower()
            pmi_min = self._min_adjacent_pmi(norm, char_counts, bigram_counts, total_bigrams)
            h_tuple = ent_map.get(token)
            if h_tuple is None:
                h_min = float('nan')
            else:
                _, _, h_min = h_tuple

            if math.isnan(pmi_min) or math.isnan(h_min):
                continue

            u_val = pmi_min + lam * h_min
            token_to_u[token] = u_val
            global_min = min(global_min, u_val)

        if global_min == float('inf'):
            global_min = float('nan')

        return token_to_u, global_min

    @staticmethod
    def _compute_char_pmi(c1: str,
                          c2: str,
                          char_counts: Counter,
                          bigram_counts: Counter,
                          total_bigrams: int,
                          log_base: Optional[float] = math.e) -> float:
        f12 = bigram_counts.get((c1, c2), 0)
        f1 = char_counts.get(c1, 0)
        f2 = char_counts.get(c2, 0)
        numerator = f12 * max(total_bigrams, 1)
        denominator = f1 * f2
        if denominator <= 0 or numerator <= 0:
            return float('-inf')
        return math.log(numerator / denominator, log_base) if log_base else math.log(numerator / denominator)

    def _min_adjacent_pmi(self,
                          word: str,
                          char_counts: Counter,
                          bigram_counts: Counter,
                          total_bigrams: int,
                          log_base: Optional[float] = math.e) -> float:
        if len(word) < 2:
            return float('nan')
        values: List[float] = []
        for i in range(len(word) - 1):
            c1, c2 = word[i], word[i + 1]
            pmi = self._compute_char_pmi(c1, c2, char_counts, bigram_counts, total_bigrams, log_base)
            values.append(pmi)
        return float('nan') if not values else min(values)

    def _token_pmi_only_map(self,
                            vocab_keys: List[str],
                            texts: List[str],
                            *,
                            lowercase: bool,
                            within_words_only: bool,
                            strip_prefixes: Tuple[str, ...],
                            valid_char_predicate=None) -> Tuple[Dict[str, float], float]:
        """Compute only the min-adjacent-char-PMI component per token."""
        char_counts, bigram_counts, total_bigrams = self._compute_char_and_bigram_counts(
            texts, lowercase=lowercase, within_words_only=within_words_only,
            valid_char_predicate=valid_char_predicate,
        )

        token_to_pmi: Dict[str, float] = {}
        global_min = float('inf')

        for token in vocab_keys:
            norm = self._normalize_token_for_chars(token, strip_prefixes)
            if lowercase:
                norm = norm.lower()
            pmi_min = self._min_adjacent_pmi(norm, char_counts, bigram_counts, total_bigrams)
            if math.isnan(pmi_min) or math.isinf(pmi_min):
                continue
            token_to_pmi[token] = pmi_min
            global_min = min(global_min, pmi_min)

        if global_min == float('inf'):
            global_min = float('nan')
        return token_to_pmi, global_min

    def _token_boundary_entropy_only_map(self,
                                          vocab_keys: List[str],
                                          texts: List[str],
                                          *,
                                          lowercase: bool,
                                          within_words_only: bool,
                                          strip_prefixes: Tuple[str, ...],
                                          valid_char_predicate=None) -> Tuple[Dict[str, float], float]:
        """Compute only the min(H_left, H_right) component per token (no lambda scaling)."""
        ent_map, _ = self._token_left_right_entropy_map(
            vocab_tokens=vocab_keys,
            texts=texts,
            lowercase=lowercase,
            within_words_only=within_words_only,
            valid_char_predicate=valid_char_predicate,
            strip_prefixes=strip_prefixes,
            drop_if_len_lt1=True,
        )

        token_to_ent: Dict[str, float] = {}
        global_min = float('inf')

        for token in vocab_keys:
            h_tuple = ent_map.get(token)
            if h_tuple is None:
                continue
            _, _, h_min = h_tuple
            if math.isnan(h_min):
                continue
            token_to_ent[token] = h_min
            global_min = min(global_min, h_min)

        if global_min == float('inf'):
            global_min = float('nan')
        return token_to_ent, global_min

    @staticmethod
    def _u_stats_from_per_token(per_token_u: Dict[str, float],
                                token_frequencies: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        values = list(per_token_u.values())
        n = len(values)
        if n == 0:
            return {
                'n_tokens': 0,
                'u_mean': None,
                'u_mean_weighted': None,
                'u_mean_log_weighted': None,
                'u_coverage': None,
                'u_median': None,
                'u_stdev': None,
                'u_min': None,
                'u_max': None,
                'u_p1': None,
                'u_p5': None,
                'u_p10': None,
                'u_p25': None,
                'u_p50': None,
                'u_p75': None,
                'u_p90': None,
                'u_p95': None,
                'u_p99': None,
            }
        values_sorted = sorted(values)

        def percentile(sorted_values: List[float], p: float) -> float:
            k = (len(sorted_values) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_values[int(k)]
            return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

        import statistics
        stats = {
            'n_tokens': n,
            'u_mean': statistics.fmean(values) if hasattr(statistics, 'fmean') else sum(values) / n,
            'u_median': statistics.median(values),
            'u_stdev': statistics.stdev(values) if n > 1 else 0.0,
            'u_min': values_sorted[0],
            'u_max': values_sorted[-1],
            'u_p1': percentile(values_sorted, 1),
            'u_p5': percentile(values_sorted, 5),
            'u_p10': percentile(values_sorted, 10),
            'u_p25': percentile(values_sorted, 25),
            'u_p50': percentile(values_sorted, 50),
            'u_p75': percentile(values_sorted, 75),
            'u_p90': percentile(values_sorted, 90),
            'u_p95': percentile(values_sorted, 95),
            'u_p99': percentile(values_sorted, 99),
        }

        stats['u_mean_weighted'] = None
        stats['u_mean_log_weighted'] = None
        stats['u_coverage'] = None

        if token_frequencies:
            total_freq = sum(token_frequencies.get(t, 0) for t in per_token_u)
            if total_freq > 0:
                stats['u_mean_weighted'] = sum(
                    token_frequencies.get(t, 0) * u for t, u in per_token_u.items()
                ) / total_freq

                total_log_freq = sum(
                    math.log1p(token_frequencies.get(t, 0)) for t in per_token_u
                )
                if total_log_freq > 0:
                    stats['u_mean_log_weighted'] = sum(
                        math.log1p(token_frequencies.get(t, 0)) * u
                        for t, u in per_token_u.items()
                    ) / total_log_freq

                total_occurrences = sum(token_frequencies.values())
                if total_occurrences > 0:
                    stats['u_coverage'] = total_freq / total_occurrences

        return stats


class _DecomposedUStatBase(UStatMetrics):
    """Shared logic for per-language decomposed U-stat variants."""

    _result_key: str = ''  # override in subclasses

    def _compute_per_token_map(self, vocab_keys, texts):
        """Override in subclasses to return (per_token_dict, global_min)."""
        raise NotImplementedError

    def compute(self, tokenized_data: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results: Dict[str, Any] = {
            self._result_key: {
                'per_tokenizer': {},
                'metadata': self._build_metadata(),
            }
        }

        get_tok = getattr(self.input_provider, 'get_tokenizer', None)
        if not callable(get_tok):
            return results

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            try:
                tok_wrapper = get_tok(tok_name)
            except Exception:
                continue
            vocab = None
            try:
                if hasattr(tok_wrapper, 'get_vocab'):
                    vocab = tok_wrapper.get_vocab()
            except Exception:
                pass
            if not vocab:
                continue

            vocab_keys = list(vocab.keys())

            # Group data by language
            lang_groups: Dict[str, List] = defaultdict(list)
            for td in tokenized_data[tok_name]:
                lang_groups[td.language].append(td)

            per_language: Dict[str, Any] = {}
            all_texts: List[str] = []

            for lang, lang_data in lang_groups.items():
                lang_texts = TokenizedDataProcessor.extract_texts(lang_data)
                if not lang_texts:
                    continue
                all_texts.extend(lang_texts)

                per_token, lang_min = self._compute_per_token_map(vocab_keys, lang_texts)

                # Token frequencies for this language
                token_id_counts: Counter = Counter()
                for td in lang_data:
                    token_id_counts.update(td.tokens)
                token_freq: Dict[str, int] = {}
                for token_str in per_token:
                    tid = vocab.get(token_str)
                    if tid is not None:
                        token_freq[token_str] = token_id_counts.get(tid, 0)

                summary = self._u_stats_from_per_token(per_token, token_freq)
                per_language[lang] = summary

            # Global (all languages pooled)
            if all_texts:
                per_token_global, global_min = self._compute_per_token_map(vocab_keys, all_texts)
                token_id_counts_global: Counter = Counter()
                for td in tokenized_data[tok_name]:
                    token_id_counts_global.update(td.tokens)
                token_freq_global: Dict[str, int] = {}
                for token_str in per_token_global:
                    tid = vocab.get(token_str)
                    if tid is not None:
                        token_freq_global[token_str] = token_id_counts_global.get(tid, 0)
                summary_global = self._u_stats_from_per_token(per_token_global, token_freq_global)
            else:
                summary_global = self._u_stats_from_per_token({}, {})
                global_min = float('nan')

            results[self._result_key]['per_tokenizer'][tok_name] = {
                'summary': summary_global,
                'per_language': per_language,
                'min_u': global_min,
                'vocab_size': len(vocab),
            }

        return results

    def _build_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class UStatPMIMetrics(_DecomposedUStatBase):
    """U-stat variant: only the min-adjacent-char-PMI component."""

    _result_key = 'ustat_pmi'

    def _build_metadata(self):
        return {
            'description': 'PMI-only component of U-statistics (min adjacent char PMI per token)',
            'definition': 'U_pmi(token) = min_adjacent_char_PMI(token)',
            'lowercase': self.lowercase,
            'within_words_only': self.within_words_only,
        }

    def _compute_per_token_map(self, vocab_keys, texts):
        return self._token_pmi_only_map(
            vocab_keys=vocab_keys, texts=texts,
            lowercase=self.lowercase, within_words_only=self.within_words_only,
            strip_prefixes=self.strip_prefixes, valid_char_predicate=self.valid_char_predicate,
        )


class UStatBoundaryEntropyMetrics(_DecomposedUStatBase):
    """U-stat variant: only the boundary entropy component (min(H_left, H_right), no lambda scaling)."""

    _result_key = 'ustat_boundary_entropy'

    def _build_metadata(self):
        return {
            'description': 'Boundary-entropy-only component of U-statistics (raw, no lambda scaling)',
            'definition': 'U_ent(token) = min(H_left(token), H_right(token))',
            'lowercase': self.lowercase,
            'within_words_only': self.within_words_only,
        }

    def _compute_per_token_map(self, vocab_keys, texts):
        return self._token_boundary_entropy_only_map(
            vocab_keys=vocab_keys, texts=texts,
            lowercase=self.lowercase, within_words_only=self.within_words_only,
            strip_prefixes=self.strip_prefixes, valid_char_predicate=self.valid_char_predicate,
        )
