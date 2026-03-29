"""
Attention mechanisms — implemented step-by-step from scratch.

Progression:
  1. SelfAttention_v1  — trainable weights via nn.Parameter
  2. SelfAttention_v2  — uses nn.Linear (better initialization)
  3. CausalAttention   — adds causal mask + dropout, supports batches
  4. MultiHeadAttentionWrapper — stacks CausalAttention heads (simple)
  5. MultiHeadAttention        — efficient single-matrix split-head version
                                 (used in the final GPT model)
"""

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """
    Basic self-attention with trainable W_Q, W_K, W_V (nn.Parameter).
    No causal masking. Useful for understanding the raw mechanics.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class SelfAttention_v2(nn.Module):
    """
    Self-attention using nn.Linear for W_Q, W_K, W_V.
    nn.Linear uses Kaiming uniform initialization (better for deep networks).
    Optional QKV bias.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class CausalAttention(nn.Module):
    """
    Masked (causal) self-attention with dropout — supports batched input.

    Causal masking ensures token i can only attend to tokens 0..i,
    preventing information leakage from future positions.

    Args:
        d_in:           Input embedding dimension.
        d_out:          Output dimension (per head).
        context_length: Maximum sequence length (used to pre-build the mask).
        dropout:        Dropout probability applied to attention weights.
        qkv_bias:       Whether to include bias in QKV projections.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out   = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Register as buffer so it moves with the model (CPU ↔ GPU)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


class MultiHeadAttentionWrapper(nn.Module):
    """
    Simple multi-head attention: stacks h independent CausalAttention heads
    and concatenates their outputs along the last dimension.

    Conceptually correct but less efficient than the split-head approach.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention using a single fused QKV projection.

    Instead of h separate weight matrices, we project to d_out and then
    split along the head dimension. This is exactly how GPT-2 works.

    Args:
        d_in:           Input embedding dimension.
        d_out:          Total output dimension (must be divisible by num_heads).
        context_length: Maximum sequence length.
        dropout:        Attention dropout probability.
        num_heads:      Number of parallel attention heads.
        qkv_bias:       Include bias in QKV projections (True for GPT-2).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out    = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # output projection
        self.dropout  = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # Split into heads: (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        # Then transpose to (b, num_heads, num_tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, head_dim) → (b, num_tokens, d_out)
        context = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)
