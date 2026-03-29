"""Tests for attention.py"""
import pytest
import torch
from src.attention import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
)

D_IN, D_OUT, SEQ_LEN, BATCH = 4, 8, 6, 2
CONTEXT_LEN = 16


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return torch.randn(SEQ_LEN, D_IN)


@pytest.fixture
def batch_inputs():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ_LEN, D_IN)


class TestSelfAttentionV1:
    def test_output_shape(self, inputs):
        attn = SelfAttention_v1(D_IN, D_OUT)
        out = attn(inputs)
        assert out.shape == (SEQ_LEN, D_OUT)

    def test_output_is_float(self, inputs):
        attn = SelfAttention_v1(D_IN, D_OUT)
        out = attn(inputs)
        assert out.dtype == torch.float32


class TestSelfAttentionV2:
    def test_output_shape(self, inputs):
        attn = SelfAttention_v2(D_IN, D_OUT)
        out = attn(inputs)
        assert out.shape == (SEQ_LEN, D_OUT)

    def test_bias_off_by_default(self):
        attn = SelfAttention_v2(D_IN, D_OUT, qkv_bias=False)
        assert attn.W_query.bias is None

    def test_bias_on(self):
        attn = SelfAttention_v2(D_IN, D_OUT, qkv_bias=True)
        assert attn.W_query.bias is not None


class TestCausalAttention:
    def test_output_shape(self, batch_inputs):
        attn = CausalAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0)
        out = attn(batch_inputs)
        assert out.shape == (BATCH, SEQ_LEN, D_OUT)

    def test_causal_mask_no_future_leakage(self, batch_inputs):
        """Perturbing future tokens must not change earlier output positions."""
        torch.manual_seed(42)
        attn = CausalAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0)
        attn.eval()

        out_original = attn(batch_inputs).detach()

        perturbed = batch_inputs.clone()
        perturbed[:, -1, :] += 100.0          # large change to the last token
        out_perturbed = attn(perturbed).detach()

        # All positions EXCEPT the last should be identical
        assert torch.allclose(out_original[:, :-1, :], out_perturbed[:, :-1, :], atol=1e-5)

    def test_mask_registered_as_buffer(self):
        attn = CausalAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0)
        assert "mask" in dict(attn.named_buffers())


class TestMultiHeadAttentionWrapper:
    def test_output_shape(self, batch_inputs):
        mha = MultiHeadAttentionWrapper(D_IN, D_OUT // 2, CONTEXT_LEN, dropout=0.0, num_heads=2)
        out = mha(batch_inputs)
        assert out.shape == (BATCH, SEQ_LEN, D_OUT)


class TestMultiHeadAttention:
    def test_output_shape(self, batch_inputs):
        mha = MultiHeadAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0, num_heads=2)
        out = mha(batch_inputs)
        assert out.shape == (BATCH, SEQ_LEN, D_OUT)

    def test_d_out_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(D_IN, 5, CONTEXT_LEN, dropout=0.0, num_heads=2)

    def test_deterministic_with_seed(self, batch_inputs):
        torch.manual_seed(1)
        mha1 = MultiHeadAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0, num_heads=2)
        out1 = mha1(batch_inputs)

        torch.manual_seed(1)
        mha2 = MultiHeadAttention(D_IN, D_OUT, CONTEXT_LEN, dropout=0.0, num_heads=2)
        out2 = mha2(batch_inputs)

        assert torch.allclose(out1, out2)
