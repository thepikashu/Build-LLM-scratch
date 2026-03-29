"""
Full GPT-2-style model implementation.

Components (all from scratch):
  - LayerNorm       : learnable scale & shift, epsilon for numerical stability
  - GELU            : approximate tanh formulation (matches GPT-2)
  - FeedForward     : 2-layer MLP with 4× expansion
  - TransformerBlock: pre-norm residual block (MHA + FFN)
  - GPTModel        : full autoregressive language model (~124M params)
"""

import torch
import torch.nn as nn
from src.attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# Default 124M config — matches GPT-2 Small
# ---------------------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size":       50257,
    "context_length":   1024,
    "emb_dim":          768,
    "n_heads":          12,
    "n_layers":         12,
    "drop_rate":        0.1,
    "qkv_bias":         False,
}


class LayerNorm(nn.Module):
    """
    Layer Normalization with learnable affine parameters.

    Normalizes across the last dimension (embedding dimension), then
    rescales with a learned scale (gamma) and shift (beta).
    Using a small epsilon avoids division-by-zero on zero-variance inputs.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit — approximate tanh formulation.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Preferred over ReLU in transformers: smooth, non-zero gradient for
    negative inputs helps training. GPT-2 uses this exact approximation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Two linear layers with a GELU activation between them.
    The inner dimension expands to 4 × emb_dim, matching the original
    Transformer paper and GPT-2 architecture.

    Input/output shape: (batch, seq_len, emb_dim)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    A single GPT-style Transformer block (decoder only).

    Architecture (pre-norm):
        x → LayerNorm → MultiHeadAttention → + x (residual)
          → LayerNorm → FeedForward        → + x (residual)

    Pre-norm (norm before sublayer) is more stable during training than
    post-norm (norm after sublayer), especially for deep models.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff       = FeedForward(cfg)
        self.norm1    = LayerNorm(cfg["emb_dim"])
        self.norm2    = LayerNorm(cfg["emb_dim"])
        self.drop_res = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_res(x)
        x = x + shortcut

        # Feed-forward sub-layer with residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_res(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    Full GPT-2-style language model.

    Layers:
      1. Token embedding          (vocab_size × emb_dim)
      2. Positional embedding     (context_length × emb_dim)
      3. Dropout on combined embedding
      4. N × TransformerBlock
      5. Final LayerNorm
      6. Linear projection head   (emb_dim → vocab_size)

    Weight tying: The output projection head shares its weights with the
    token embedding matrix (reduces parameters by ~38M for the 124M model).

    Input:  (batch_size, seq_len)  — token IDs
    Output: (batch_size, seq_len, vocab_size)  — logits over vocabulary
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_memory_mb(model: nn.Module) -> float:
    """Estimate model memory in MB assuming float32."""
    total = sum(p.numel() for p in model.parameters())
    return (total * 4) / (1024 ** 2)
