# LLM From Scratch

A complete, ground-up implementation of a GPT-style Large Language Model, built entirely in Python and PyTorch, without relying on HuggingFace or any high-level LLM libraries.

This project walks through every core concept required to understand and build modern LLMs: from raw text tokenization all the way to instruction fine-tuning and LLM-based evaluation.

## What This Project Covers

| Stage | Topics |
|---|---|
| **1. Tokenization** | Regex tokenizer, BPE via tiktoken, special tokens (`<\|endoftext\|>`, `<\|unk\|>`) |
| **2. Data Pipeline** | Sliding window dataset, DataLoader with stride & batching |
| **3. Embeddings** | Token embeddings + positional embeddings |
| **4. Attention** | Naive dot-product → scaled dot-product → causal masking → dropout |
| **5. Self-Attention** | Trainable W_Q, W_K, W_V weight matrices (v1 & v2) |
| **6. Multi-Head Attention** | Wrapper approach + efficient weight-split implementation |
| **7. Transformer Block** | Layer norm → MHA → Feed-forward (GELU) → residual connections |
| **8. Full GPT Model** | 124M parameter GPT-2 replica with weight tying |
| **9. Text Generation** | Greedy decoding, temperature scaling, top-k sampling |
| **10. Training Loop** | Cross-entropy loss, AdamW, train/val tracking, perplexity |
| **11. Pretrained Weights** | Loading OpenAI GPT-2 weights into the custom architecture |
| **12. Classification Fine-Tuning** | Spam detection with frozen backbone + classification head |
| **13. Instruction Fine-Tuning** | Alpaca-format SFT on instruction-following dataset |
| **14. LLM Evaluation** | Scoring model responses using Llama3 via Ollama |

## Architecture Overview

Input Text
    │
    ▼
┌─────────────────────┐
│  Tokenizer (BPE)    │  tiktoken / GPT-2 vocab (50,257 tokens)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │  vocab_size × emb_dim  (50257 × 768)
│  + Pos Embedding    │  context_length × emb_dim (1024 × 768)
└─────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│         Transformer Block × 12       │
│  ┌────────────────────────────────┐  │
│  │  LayerNorm                     │  │
│  │  Multi-Head Attention (12 heads)│  │
│  │  Residual Connection           │  │
│  │  LayerNorm                     │  │
│  │  FeedForward (GELU, 4× expand) │  │
│  │  Residual Connection           │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Final LayerNorm    │
│  Linear Head        │  emb_dim → vocab_size
└─────────────────────┘
    │
    ▼
 Logits / Generated Text

**Model Size:** ~124M parameters (matching GPT-2 Small)


## Key Concepts Implemented

### Custom Tokenizer
Two versions built from scratch:
- **V1** — simple regex-based word tokenizer with vocab lookup
- **V2** — adds `<|unk|>` and `<|endoftext|>` special tokens
- **BPE** — GPT-2's Byte Pair Encoding via `tiktoken`

### Attention Mechanism

1. Naive dot-product attention (no learned weights)
2. Scaled dot-product with softmax normalization
3. Self-attention with learned W_Q, W_K, W_V (v1: nn.Parameter, v2: nn.Linear)
4. Causal (masked) attention — prevents attending to future tokens
5. Dropout on attention weights
6. Multi-head attention — weight wrapper and efficient split-head implementation

### The Full GPT Stack
Every component hand-implemented:
- `LayerNorm` — with learnable scale & shift parameters
- `GELU` activation — approximate tanh formulation
- `FeedForward` — 2-layer MLP with 4× expansion
- `TransformerBlock` — pre-norm architecture with residual connections
- `GPTModel` — full 124M parameter model with weight tying

### Decoding Strategies
- **Greedy** — `argmax` over logits
- **Temperature scaling** — sharpen or flatten the distribution
- **Top-k sampling** — restrict sampling to top-k tokens, combine with temperature

### Fine-Tuning
- **Classification** — freeze GPT backbone, replace output head, fine-tune last transformer block on SMS spam dataset (~95%+ accuracy)
- **Instruction (SFT)** — Alpaca-format instruction fine-tuning, custom collate with ignore-index padding, trained on 1000-entry instruction dataset


## Model Configuration

python
GPT_CONFIG_124M = {
    "vocab_size":       50257,   # GPT-2 BPE vocabulary
    "context_length":   1024,    # Max sequence length
    "emb_dim":          768,     # Embedding dimension
    "n_heads":          12,      # Attention heads
    "n_layers":         12,      # Transformer blocks
    "drop_rate":        0.1,     # Dropout probability
    "qkv_bias":         False,   # QKV projection bias
}
# Total parameters: ~124M
# Memory footprint: ~473 MB (float32)

## References & Inspiration

- **Course** [Building LLMs from scratch](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=JPcvX6pdrTQouPBc) — Dr. Raj Dandekar
- **Book:** [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) — Sebastian Raschka
- **Original GPT-2 Paper:** [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — Radford et al., OpenAI
- **Attention Is All You Need:** Vaswani et al., 2017
- **GPT-2 Weights:** [OpenAI on HuggingFace](https://huggingface.co/openai-community/gpt2)

## Author

Yashasvee Taiwade 
