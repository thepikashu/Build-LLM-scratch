"""
Tokenizer implementations built from scratch.

Covers:
  - SimpleTokenizerV1: basic regex tokenizer with vocabulary lookup
  - SimpleTokenizerV2: adds <|unk|> and <|endoftext|> special tokens
  - BPE via tiktoken (GPT-2 encoding)
"""

import re


class SimpleTokenizerV1:
    """
    A minimal tokenizer that encodes text to integer IDs and back.
    Vocabulary must be pre-built (e.g., from a training corpus).
    Raises KeyError on out-of-vocabulary tokens.
    """

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.;:_\'?!"()]|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.;:?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    Extended tokenizer with special tokens:
      - <|unk|>       : unknown / out-of-vocabulary tokens
      - <|endoftext|> : document separator
    """

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.;:_\'?!"()]|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            token if token in self.str_to_int else "<|unk|>"
            for token in preprocessed
        ]
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.;:?!"()\'])', r'\1', text)
        return text


def build_vocab(text: str) -> dict:
    """Build a character-level vocabulary from raw text."""
    preprocessed = re.split(r'([,.;:_\'?!"()]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_tokens)}
