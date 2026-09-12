"""
Utility functions for tokenizer analysis.
Attempts to import from parent codebase first, falls back to standalone versions.
"""

import math
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE  
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import torch
import pathpiece

def load_tokenizer_from_config(config, name: str = "tokenizer"):
    """
    Load tokenizer wrapper from configuration (here for backwards compatibility)
    
    Args:
        config: Configuration dictionary
        name: Tokenizer name (for the wrapper)
        
    Returns:
        TokenizerWrapper instance
    """
    from ..core.tokenizer_wrapper import create_tokenizer_wrapper
    logger.warning("Deprecated function; Use `create_tokenizer_wrapper` in core.tokenizer_wrapper instead")
    return create_tokenizer_wrapper(name, config)


def _load_huggingface_tokenizer(config):
    """
    Internal function to load raw HuggingFace tokenizer from configuration.
    
    This function is used by the HuggingFaceTokenizer wrapper.
    Tries to use original implementation first, falls back to simplified version.
    """
 
    tokenizer_class = config.get('class', 'huggingface')
    
    if tokenizer_class == "custom_bpe":
        # Custom BPE tokenizer loading
        vocab_file = os.path.join(config['path'], "vocab.json")
        merges_file = os.path.join(config['path'], "merges.txt")
    
        # Load vocab and merges from files
        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
        with open(merges_file, "r", encoding="utf-8") as mf:
            merges = [tuple(line.strip().split()) for line in mf if not line.startswith("#")]
    
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
    
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
    
        # Set special tokens
        tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
        tokenizer.model.unk_token = "<unk>"
    
        # Set post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ]
        )
        
        return tokenizer
    
    else:
        path = config['path']
        
        # Strategy 1: If path points to a JSON file, use Tokenizer.from_file
        if path.endswith('.json') or os.path.isfile(path):
            try:
                logger.info(f"Loading tokenizer from file: {path}")
                tokenizer = Tokenizer.from_file(path)
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from file {path}: {e}")
        
        # Strategy 2: Try loading as HuggingFace tokenizer (directory or model name)
        try:
            logger.info(f"Loading tokenizer from HuggingFace: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace tokenizer from {path}: {e}")
    
        # Strategy 3: If path is a directory, look for tokenizer files
        if os.path.isdir(path):
            # Look for common tokenizer file names
            for filename in ['tokenizer.json', 'vocab.json', 'merges.txt']:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    try:
                        if filename == 'tokenizer.json':
                            logger.info(f"Loading tokenizer from {file_path}")
                            return Tokenizer.from_file(file_path)
                        elif filename == 'vocab.json' and os.path.exists(os.path.join(path, 'merges.txt')):
                            # Load as BPE tokenizer
                            logger.info(f"Loading BPE tokenizer from directory: {path}")
                            return _load_bpe_from_directory(path)
                    except Exception as e:
                        logger.warning(f"Failed to load tokenizer from {file_path}: {e}")
                        continue
        
        raise ValueError(f"Could not load tokenizer from {path}.")


class HFCompatTokenizer:
    """PathPiece tokenizer wrapped to mimic HuggingFace interface."""

    def __init__(self, vocab, greedy, random_tiebreaker, eos="<|endoftext|>", eos_id=0):
        self.tok = pathpiece.Tokenizer(vocab, random_tiebreaker=random_tiebreaker, greedy=greedy)
        self.eos_token_id = eos_id
        self.eos_token = eos
        # SOAR: MorphScore requires a special_tokens_map (the tiktoken/tokenmonster wrappers set an empty one)
        self.special_tokens_map = {"eos_token": eos}

    def encode(self, text, **kwargs):
        return self.tok.encode(text)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        """SOAR: HF-style callable for MorphScore (mirrors the tiktoken/tokenmonster compat wrappers)."""
        ids = self.tok.encode(text)["input_ids"]
        class TokenizerOutput:
            def __init__(self, ids):
                self.ids = ids
                self.input_ids = ids
        return TokenizerOutput(ids)

    def encode_batch(self, texts, **kwargs):
        ids_list = self.tok.encode_batch(texts)["input_ids"]
        return [{"ids": ids} for ids in ids_list]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            return self.tok.decode([token_ids])
        return self.tok.decode(token_ids)

    def get_vocab(self):
        raw = self.tok.get_vocab()
        result = {}
        for k, v in raw.items():
            if isinstance(k, bytes):
                try:
                    key = k.decode('utf-8')
                except UnicodeDecodeError:
                    key = '<0x' + k.hex() + '>'
            else:
                key = k
            while key in result:
                key = key + '\x00'
            result[key] = v
        return result

    def get_vocab_size(self):
        raw = self.tok.get_vocab()
        return max(raw.values()) + 1 if raw else 0


def _load_pathpiece_tokenizer(config):
    """Load a PathPiece tokenizer from config."""
    model_name = config.get('path')
    if not model_name:
        raise ValueError("PathPiece config missing required 'path' field")

    base_dir = config.get('base_dir', "/Volumes/T7/SOAR/timtc_vocabs_models/vocabularies")
    vocab_name = model_name.split("/")[-1] + ".vocab"
    model_path = os.path.join(base_dir, vocab_name)

    # SOAR: explicit config keys override the name rule. The paper's TMTC metrics run used
    # greedy=False, random_tiebreaker=True for every .vocab family (tmtc_all_54_25k_nohup.log), which the
    # name rule does not reproduce for sage_*/unigram_greedy_* (greedy) or pathpiecel_* (no random tiebreak).
    greedy = bool(config["greedy"]) if "greedy" in config else ("greedy" in model_name)
    random_tiebreaker = bool(config["random_tiebreaker"]) if "random_tiebreaker" in config else ("pathpiecer" in model_name)

    logger.info(f"Loading pathpiece tokenizer from {model_path} (greedy={greedy}, random_tiebreaker={random_tiebreaker})")
    return HFCompatTokenizer(vocab=model_path, greedy=greedy, random_tiebreaker=random_tiebreaker)


def _load_bpe_from_directory(directory_path):
    """Helper function to load BPE tokenizer from directory with vocab.json and merges.txt"""
    vocab_file = os.path.join(directory_path, "vocab.json")
    merges_file = os.path.join(directory_path, "merges.txt")
    
    # Load vocab and merges from files
    with open(vocab_file, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    with open(merges_file, "r", encoding="utf-8") as mf:
        merges = [tuple(line.strip().split()) for line in mf if not line.startswith("#")]

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])

    # Set special tokens
    tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
    tokenizer.model.unk_token = "<unk>"

    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )
    
    return tokenizer


def setup_environment():
    """Setup environment for tokenizer analysis."""
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
