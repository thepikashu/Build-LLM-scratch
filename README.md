# Building an LLM from scratch

Implementation of a GPT-style decoder-only transformer in PyTorch, built from scratch including attention, embeddings, and autoregressive text generation.

## Architecture

Input → Tokenization → Embeddings (Token + Positional)  → Transformer Blocks (Masked Multi-Head Attention + FeedForward)  → Linear Layer → Softmax → Next Token Prediction

## Components

- Tokenization (BPE / subword)
- Token + Positional Embeddings
- Scaled Dot-Product Attention
- Masked (Causal) Attention
- Multi-Head Attention
- FeedForward Network
- Layer Normalization

## Implementation and Training

- Built in PyTorch
- Modular transformer components
- Batch training with DataLoader
- Autoregressive next-token prediction using shifted input-output pairs

## Key Learnings

- How attention mechanisms capture relationships between tokens
- Importance of masking for autoregressive models
- Role of embeddings in encoding semantic meaning and position
- Training challenges like stability and gradient issues
