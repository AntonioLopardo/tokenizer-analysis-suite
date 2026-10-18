"""
Tokenizer Analysis Module

A modular framework for comprehensive tokenizer comparison and analysis.
Supports both pairwise and multi-tokenizer comparisons with various metrics
including morphological alignment, information-theoretic measures, and
segmentation analysis.
"""

__version__ = "1.0.0"
__author__ = "Tokenizer Analysis Project"

from .metrics.base import BaseMetrics
from .metrics.basic import BasicTokenizationMetrics
from .metrics.information_theoretic import InformationTheoreticMetrics
from .metrics.morphological import MorphologicalMetrics
from .metrics.gini import TokenizerGiniMetrics
from .metrics.cognitive_plausibility import CognitivePlausibilityMetrics
from .metrics.ustat import UStatMetrics
from .loaders import MorphologicalDataLoader

# Register custom tokenizer classes (tiktoken, tokenmonster, tekken)
try:
    from .core.custom_tokenizers import register_custom_tokenizers
    register_custom_tokenizers()
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"Custom tokenizers not available: {e}")
from .visualization import TokenizerVisualizer
from .main import UnifiedTokenizerAnalyzer, create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data

__all__ = [
    "BaseMetrics",
    "BasicTokenizationMetrics",
    "InformationTheoreticMetrics", 
    "MorphologicalMetrics",
    "TokenizerGiniMetrics",
    "CognitivePlausibilityMetrics",
    "UStatMetrics",
    "MorphologicalDataLoader",
    "TokenizerVisualizer",
    "UnifiedTokenizerAnalyzer",
    "create_analyzer_from_raw_inputs",
    "create_analyzer_from_tokenized_data"
]