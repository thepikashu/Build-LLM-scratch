"""
Text generation utilities for autoregressive GPT-style models.

Decoding strategies implemented:
  - Greedy decoding (generate_text_simple)
  - Temperature scaling
  - Top-k sampling
  - Combined temperature + top-k (generate)
"""

import torch
import tiktoken


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """Encode text to a (1, seq_len) tensor of token IDs."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """Decode a (1, seq_len) or (seq_len,) tensor back to a string."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Greedy autoregressive text generation.

    At each step, the model receives the current token sequence,
    predicts logits for the next token, and appends the argmax token.

    Args:
        model:          Trained GPTModel (in eval mode).
        idx:            Starting token IDs, shape (batch, n_tokens).
        max_new_tokens: Number of tokens to generate.
        context_size:   Model's maximum context length (trims input if needed).

    Returns:
        Token ID tensor of shape (batch, n_tokens + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]           # trim to context window
        with torch.no_grad():
            logits = model(idx_cond)                 # (batch, seq_len, vocab_size)
        logits = logits[:, -1, :]                    # last token's logits
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int = None,
    eos_id: int = None,
) -> torch.Tensor:
    """
    Autoregressive generation with temperature scaling and top-k sampling.

    Decoding modes:
      temperature == 0.0           → greedy (argmax)
      temperature > 0, top_k=None  → temperature-scaled multinomial sampling
      temperature > 0, top_k=N     → top-k sampling with temperature

    Args:
        model:          Trained GPTModel.
        idx:            Starting tokens, shape (batch, n_tokens).
        max_new_tokens: Tokens to generate.
        context_size:   Model's context window (clips input if needed).
        temperature:    Softmax temperature. 0 = greedy, >1 = more random.
        top_k:          If set, restrict sampling to top-k most likely tokens.
        eos_id:         If set, stop generation when this token is produced.

    Returns:
        Token IDs including the original prompt.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # --- Top-k filtering ---
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # --- Temperature scaling + sampling ---
        if temperature > 0.0:
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # --- Early stopping on EOS ---
        if eos_id is not None and (next_token == eos_id).all():
            break

        idx = torch.cat((idx, next_token), dim=1)

    return idx
