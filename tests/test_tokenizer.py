"""Tests for tokenizer.py"""
import pytest
from src.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2, build_vocab


SAMPLE_TEXT = "Hello, world. Hello again!"


@pytest.fixture
def vocab():
    return build_vocab(SAMPLE_TEXT)


def test_build_vocab_contains_special_tokens(vocab):
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab


def test_build_vocab_all_unique(vocab):
    assert len(vocab) == len(set(vocab.keys()))


def test_v1_encode_decode_roundtrip(vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    text = "Hello, world."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_v1_raises_on_unknown_token(vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    with pytest.raises(KeyError):
        tokenizer.encode("xyzzy_unknown_word")


def test_v2_handles_unknown_token(vocab):
    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode("xyzzy_unknown_word")
    # should not raise; maps to <|unk|>
    decoded = tokenizer.decode(ids)
    assert "<|unk|>" in decoded


def test_v2_endoftext_separator(vocab):
    tokenizer = SimpleTokenizerV2(vocab)
    text = "<|endoftext|>".join(["Hello, world.", "Hello again!"])
    ids = tokenizer.encode(text)
    assert vocab["<|endoftext|>"] in ids
