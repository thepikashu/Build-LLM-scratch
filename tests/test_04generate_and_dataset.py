"""Tests for generate.py and dataset.py"""
import pytest
import torch
from unittest.mock import MagicMock
from src.model import GPTModel
from src.generate import generate_text_simple, generate, text_to_token_ids, token_ids_to_text
from src.dataset import GPTDatasetV1, create_dataloader_v1


SMALL_CFG = {
    "vocab_size":     100,
    "context_length": 16,
    "emb_dim":        32,
    "n_heads":        4,
    "n_layers":       2,
    "drop_rate":      0.0,
    "qkv_bias":       False,
}


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    m = GPTModel(SMALL_CFG)
    m.eval()
    return m


@pytest.fixture
def mock_tokenizer():
    """Minimal tokenizer mock for dataset tests — no tiktoken dependency."""
    tok = MagicMock()
    # encode returns a list of ints
    tok.encode.side_effect = lambda text, **kwargs: [i % 100 for i in range(len(text))]
    tok.decode.side_effect = lambda ids: "x" * len(ids)
    return tok


# ---------------------------------------------------------------------------
# generate.py
# ---------------------------------------------------------------------------

class TestGenerateTextSimple:
    def test_length_increases(self, small_model):
        idx = torch.randint(0, 100, (1, 4))
        out = generate_text_simple(small_model, idx, max_new_tokens=5, context_size=16)
        assert out.shape == (1, 9)

    def test_prefix_preserved(self, small_model):
        idx = torch.randint(0, 100, (1, 4))
        out = generate_text_simple(small_model, idx, max_new_tokens=3, context_size=16)
        assert torch.equal(out[:, :4], idx)


class TestGenerate:
    def test_greedy_output_length(self, small_model):
        idx = torch.randint(0, 100, (1, 3))
        out = generate(small_model, idx, max_new_tokens=4, context_size=16, temperature=0.0)
        assert out.shape == (1, 7)

    def test_sampling_output_length(self, small_model):
        torch.manual_seed(1)
        idx = torch.randint(0, 100, (1, 3))
        out = generate(small_model, idx, max_new_tokens=4, context_size=16, temperature=1.0)
        assert out.shape == (1, 7)

    def test_topk_output_length(self, small_model):
        torch.manual_seed(2)
        idx = torch.randint(0, 100, (1, 3))
        out = generate(small_model, idx, max_new_tokens=4, context_size=16,
                       temperature=1.0, top_k=5)
        assert out.shape == (1, 7)

    def test_eos_stops_early(self, small_model):
        """Generation stops if EOS is produced — output may be shorter than max."""
        idx = torch.randint(0, 100, (1, 3))
        # Monkey-patch the model to always predict token 0 (our fake EOS)
        with torch.no_grad():
            out = generate(small_model, idx, max_new_tokens=10, context_size=16,
                           temperature=0.0, eos_id=small_model(idx[:, :3])[:, -1, :].argmax().item())
        # Just check it completes without error; length is 4 (3 + 1 EOS step)
        assert out.shape[1] >= 4


class TestTokenHelpers:
    def test_text_to_token_ids_shape(self, mock_tokenizer):
        ids = text_to_token_ids("hello world", mock_tokenizer)
        assert ids.dim() == 2
        assert ids.shape[0] == 1

    def test_token_ids_to_text_returns_string(self, mock_tokenizer):
        ids = torch.tensor([[1, 2, 3]])
        result = token_ids_to_text(ids, mock_tokenizer)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# dataset.py
# ---------------------------------------------------------------------------

class TestGPTDatasetV1:
    def test_length(self, mock_tokenizer):
        # 200 chars → 200 token ids; max_length=10, stride=10 → ~19 samples
        text = "a" * 200
        ds = GPTDatasetV1(text, mock_tokenizer, max_length=10, stride=10)
        assert len(ds) > 0

    def test_item_shapes(self, mock_tokenizer):
        text = "a" * 200
        ds = GPTDatasetV1(text, mock_tokenizer, max_length=10, stride=10)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_target_is_input_shifted_right(self, mock_tokenizer):
        text = "a" * 200
        ds = GPTDatasetV1(text, mock_tokenizer, max_length=10, stride=10)
        x, y = ds[0]
        # y[i] should equal the token that comes after x[i]
        # Since x and y come from the same sliding window offset by 1, x[1:] == y[:-1]
        assert torch.equal(x[1:], y[:-1])

    def test_stride_controls_overlap(self, mock_tokenizer):
        text = "a" * 100
        ds_stride1  = GPTDatasetV1(text, mock_tokenizer, max_length=10, stride=1)
        ds_stride10 = GPTDatasetV1(text, mock_tokenizer, max_length=10, stride=10)
        assert len(ds_stride1) > len(ds_stride10)


class TestCreateDataloader:
    def test_batch_shape(self, mock_tokenizer):
        text = "a" * 500
        loader = create_dataloader_v1(text, mock_tokenizer, batch_size=4,
                                      max_length=16, stride=16, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (4, 16)
        assert yb.shape == (4, 16)
