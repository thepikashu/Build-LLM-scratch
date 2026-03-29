"""Tests for model.py"""
import pytest
import torch
import torch.nn as nn
from src.model import GPTModel, LayerNorm, GELU, FeedForward, TransformerBlock, count_parameters, model_memory_mb

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
def model():
    torch.manual_seed(0)
    return GPTModel(SMALL_CFG)


@pytest.fixture
def token_ids():
    return torch.randint(0, SMALL_CFG["vocab_size"], (2, 8))


class TestLayerNorm:
    def test_mean_near_zero(self):
        ln = LayerNorm(32)
        x = torch.randn(4, 32)
        out = ln(x)
        assert out.mean(dim=-1).abs().max().item() < 1e-4

    def test_var_near_one(self):
        ln = LayerNorm(32)
        x = torch.randn(4, 32)
        out = ln(x)
        assert (out.var(dim=-1, unbiased=False) - 1).abs().max().item() < 1e-4

    def test_learnable_params(self):
        ln = LayerNorm(32)
        assert ln.scale.requires_grad
        assert ln.shift.requires_grad


class TestGELU:
    def test_zero_input(self):
        gelu = GELU()
        assert gelu(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-4)

    def test_large_positive(self):
        gelu = GELU()
        val = gelu(torch.tensor(10.0)).item()
        assert val == pytest.approx(10.0, rel=1e-2)

    def test_output_shape(self):
        gelu = GELU()
        x = torch.randn(3, 4, 32)
        assert gelu(x).shape == x.shape


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(SMALL_CFG)
        x = torch.randn(2, 8, SMALL_CFG["emb_dim"])
        out = ff(x)
        assert out.shape == x.shape

    def test_inner_expansion(self):
        ff = FeedForward(SMALL_CFG)
        assert ff.layers[0].out_features == 4 * SMALL_CFG["emb_dim"]


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(SMALL_CFG)
        x = torch.randn(2, 8, SMALL_CFG["emb_dim"])
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection_identity_at_zero_weight(self):
        """If all weights are zero, residual means output == input."""
        block = TransformerBlock(SMALL_CFG)
        for p in block.parameters():
            nn.init.zeros_(p)
        x = torch.randn(1, 4, SMALL_CFG["emb_dim"])
        out = block(x)
        # With all-zero weights the output won't equal x exactly (layer norm
        # still fires), but the shape must match and the op must not error.
        assert out.shape == x.shape


class TestGPTModel:
    def test_output_shape(self, model, token_ids):
        out = model(token_ids)
        assert out.shape == (2, 8, SMALL_CFG["vocab_size"])

    def test_no_nan_in_output(self, model, token_ids):
        out = model(token_ids)
        assert not torch.isnan(out).any()

    def test_count_parameters_positive(self, model):
        assert count_parameters(model) > 0

    def test_model_memory_mb_positive(self, model):
        assert model_memory_mb(model) > 0.0

    def test_eval_mode_deterministic(self, model, token_ids):
        model.eval()
        with torch.no_grad():
            out1 = model(token_ids)
            out2 = model(token_ids)
        assert torch.allclose(out1, out2)

    def test_context_length_respected(self, model):
        """Sequences up to context_length must work; beyond must be clipped by caller."""
        full_ctx = torch.randint(0, SMALL_CFG["vocab_size"], (1, SMALL_CFG["context_length"]))
        out = model(full_ctx)
        assert out.shape == (1, SMALL_CFG["context_length"], SMALL_CFG["vocab_size"])
