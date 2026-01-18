"""
Custom tokenizer wrappers for non-HuggingFace tokenizers.

Provides adapters for:
- tiktoken (OpenAI's tokenizer, used by GPT-4o)
- tokenmonster (ungreedy tokenization)
"""

from typing import Dict, List, Optional, Any
import logging

from .tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


class TiktokenTokenizer(TokenizerWrapper):
    """Wrapper for tiktoken tokenizers (OpenAI)."""
    
    def __init__(self, name: str, encoding, config: Dict[str, Any]):
        """
        Initialize tiktoken tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            encoding: tiktoken Encoding instance
            config: Original configuration dict
        """
        self._name = name
        self._encoding = encoding
        self._config = config
        self._vocab = None  # Lazy load
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return self._encoding.n_vocab
    
    def get_vocab(self) -> Optional[Dict[str, int]]:
        """Get vocabulary mapping. Built lazily from encoding."""
        if self._vocab is None:
            # tiktoken doesn't expose vocab directly, but we can build it
            # from the decoder. Note: this is expensive and may not include all tokens.
            try:
                self._vocab = {}
                # Try to decode each token ID
                for i in range(min(self._encoding.n_vocab, 200000)):
                    try:
                        token_bytes = self._encoding.decode_single_token_bytes(i)
                        token_str = token_bytes.decode('utf-8', errors='replace')
                        self._vocab[token_str] = i
                    except Exception:
                        # Some token IDs may not be decodable
                        pass
            except Exception as e:
                logger.warning(f"Could not build vocab for tiktoken: {e}")
                self._vocab = {}
        return self._vocab
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text)
    
    def can_pretokenize(self) -> bool:
        return False
    
    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError("tiktoken does not support pretokenization")
    
    def get_underlying_tokenizer(self):
        """Return the underlying tiktoken encoding object with MorphScore compatibility."""
        # Create a wrapper that adds HF-compatible attributes for MorphScore
        class TiktokenHFCompat:
            def __init__(self, encoding, parent):
                self._encoding = encoding
                self._parent = parent
                # Empty special tokens map - tiktoken special tokens are handled differently
                # GPT-4o has special tokens like <|endoftext|> but they're not exposed the same way
                self.special_tokens_map = {}
            
            def __call__(self, text, add_special_tokens=True, **kwargs):
                """HuggingFace-style callable interface."""
                token_ids = self._encoding.encode(text)
                # Return a simple object with .ids attribute like HF tokenizers
                class TokenizerOutput:
                    def __init__(self, ids):
                        self.ids = ids
                        self.input_ids = ids
                return TokenizerOutput(token_ids)
            
            def decode(self, token_ids, skip_special_tokens=True):
                """Decode token IDs to string."""
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return self._encoding.decode(token_ids)
            
            def encode(self, text):
                """Encode text to token IDs."""
                return self._encoding.encode(text)
            
            def get_vocab(self):
                return self._parent.get_vocab()
            
            @property
            def n_vocab(self):
                return self._encoding.n_vocab
            
            def __getattr__(self, name):
                # Forward other attributes to the underlying encoding
                return getattr(self._encoding, name)
        
        return TiktokenHFCompat(self._encoding, self)
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TiktokenTokenizer':
        """Create tiktoken tokenizer wrapper from config."""
        import tiktoken
        
        encoding_name = config.get('encoding', 'o200k_base')  # Default to GPT-4o
        
        # Handle different ways to specify the encoding
        if 'path' in config:
            # If path is like 'tiktoken/gpt-4o', extract the encoding
            path = config['path']
            if 'gpt-4o' in path.lower() or 'o200k' in path.lower():
                encoding_name = 'o200k_base'
            elif 'gpt-4' in path.lower() or 'cl100k' in path.lower():
                encoding_name = 'cl100k_base'
            elif 'gpt-3' in path.lower() or 'p50k' in path.lower():
                encoding_name = 'p50k_base'
        
        encoding = tiktoken.get_encoding(encoding_name)
        logger.info(f"Loaded tiktoken encoding: {encoding_name} (vocab_size={encoding.n_vocab})")
        return cls(name, encoding, config)


class TokenMonsterTokenizer(TokenizerWrapper):
    """Wrapper for TokenMonster tokenizers."""
    
    def __init__(self, name: str, vocab, config: Dict[str, Any]):
        """
        Initialize TokenMonster tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            vocab: tokenmonster Vocab instance
            config: Original configuration dict
        """
        self._name = name
        self._vocab_obj = vocab
        self._config = config
        self._vocab_dict = None  # Lazy load
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        # TokenMonster vocab size
        return len(self._vocab_obj)
    
    def get_vocab(self) -> Optional[Dict[str, int]]:
        """Get vocabulary mapping."""
        if self._vocab_dict is None:
            try:
                # TokenMonster uses get_dictionary() to access vocabulary
                # Returns a dict keyed by token ID: {id: {'id': ..., 'token': ..., ...}}
                self._vocab_dict = {}
                dictionary = self._vocab_obj.get_dictionary()
                for token_id, entry in dictionary.items():
                    # Use decoded token for human-readable form
                    token_str = entry.get('token_decoded', entry.get('token', ''))
                    if isinstance(token_str, bytes):
                        token_str = token_str.decode('utf-8', errors='replace')
                    self._vocab_dict[str(token_str)] = token_id
                logger.info(f"Built vocab for tokenmonster: {len(self._vocab_dict)} tokens")
            except Exception as e:
                logger.warning(f"Could not build vocab for tokenmonster: {e}")
                self._vocab_dict = {}
        return self._vocab_dict
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        # TokenMonster returns numpy array or list
        result = self._vocab_obj.tokenize(text)
        if hasattr(result, 'tolist'):
            return result.tolist()
        return list(result)
    
    def can_pretokenize(self) -> bool:
        return False
    
    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError("TokenMonster does not support pretokenization")
    
    def get_underlying_tokenizer(self):
        """Return the underlying TokenMonster vocab object with MorphScore compatibility."""
        # Create a wrapper that adds HF-compatible attributes for MorphScore
        class TokenMonsterHFCompat:
            def __init__(self, vocab_obj, parent):
                self._vocab = vocab_obj
                self._parent = parent
                # Empty special tokens map - TokenMonster handles special tokens differently
                self.special_tokens_map = {}
            
            def __call__(self, text, add_special_tokens=True, **kwargs):
                """HuggingFace-style callable interface."""
                result = self._vocab.tokenize(text)
                if hasattr(result, 'tolist'):
                    token_ids = result.tolist()
                else:
                    token_ids = list(result)
                # Return a simple object with .ids attribute like HF tokenizers
                class TokenizerOutput:
                    def __init__(self, ids):
                        self.ids = ids
                        self.input_ids = ids
                return TokenizerOutput(token_ids)
            
            def decode(self, token_ids, skip_special_tokens=True):
                """Decode token IDs to string."""
                return self._vocab.decode(token_ids)
            
            def encode(self, text):
                """Encode text to token IDs."""
                result = self._vocab.tokenize(text)
                if hasattr(result, 'tolist'):
                    return result.tolist()
                return list(result)
            
            def get_vocab(self):
                return self._parent.get_vocab()
            
            def __getattr__(self, name):
                # Forward other attributes to the underlying vocab
                return getattr(self._vocab, name)
        
        return TokenMonsterHFCompat(self._vocab_obj, self)
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TokenMonsterTokenizer':
        """Create TokenMonster tokenizer wrapper from config."""
        import tokenmonster
        
        # Default vocab path
        vocab_path = config.get('path', 'englishcode-32000-consistent-v1')
        
        # Handle different path formats
        if '/' in vocab_path:
            # Format like 'tokenmonster/englishcode-32000-consistent-v1'
            vocab_path = vocab_path.split('/')[-1]
        
        # Load the vocabulary
        vocab = tokenmonster.load(vocab_path)
        logger.info(f"Loaded TokenMonster vocab: {vocab_path} (vocab_size={len(vocab)})")
        return cls(name, vocab, config)


class MistralTekkenTokenizer(TokenizerWrapper):
    """
    Wrapper for Mistral's Tekken tokenizer.
    
    Note: Tekken is available via the mistral-common package or through HuggingFace
    via Mistral-Nemo-Base-2407 which uses the Tekken tokenizer.
    """
    
    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
        Initialize Tekken tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            tokenizer: HuggingFace tokenizer (from Mistral-Nemo) or mistral_common tokenizer
            config: Original configuration dict
        """
        self._name = name
        self._tokenizer = tokenizer
        self._config = config
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        if hasattr(self._tokenizer, 'vocab_size'):
            return self._tokenizer.vocab_size
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Optional[Dict[str, int]]:
        if hasattr(self._tokenizer, 'get_vocab'):
            return self._tokenizer.get_vocab()
        return None
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        result = self._tokenizer.encode(text)
        if isinstance(result, list):
            return result
        elif hasattr(result, 'ids'):
            return result.ids
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids']
        return list(result)
    
    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None
    
    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    
    def get_underlying_tokenizer(self):
        """Return the underlying tokenizer object."""
        return self._tokenizer
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'MistralTekkenTokenizer':
        """Create Tekken tokenizer wrapper from config."""
        from transformers import AutoTokenizer
        
        # Tekken is the tokenizer used by Mistral-Nemo
        path = config.get('path', 'mistralai/Mistral-Nemo-Base-2407')
        
        # Handle different path formats
        if path == 'mistralai/tekken':
            # The actual HF model that uses Tekken
            path = 'mistralai/Mistral-Nemo-Base-2407'
        
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        logger.info(f"Loaded Tekken tokenizer from: {path} (vocab_size={tokenizer.vocab_size})")
        return cls(name, tokenizer, config)


def register_custom_tokenizers():
    """Register all custom tokenizer classes with the framework."""
    from .tokenizer_wrapper import register_tokenizer_class
    
    register_tokenizer_class('tiktoken', TiktokenTokenizer)
    register_tokenizer_class('tokenmonster', TokenMonsterTokenizer)
    register_tokenizer_class('tekken', MistralTekkenTokenizer)
    
    logger.info("Registered custom tokenizer classes: tiktoken, tokenmonster, tekken")
