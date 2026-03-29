"""
Dataset and DataLoader utilities for GPT-style language model training.

GPTDatasetV1 uses a sliding window over tokenized text to create
(input, target) pairs for next-token prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    Sliding-window dataset for autoregressive language modeling.

    For each window of `max_length` tokens, the target is the same
    window shifted one position to the right (next-token prediction).

    Args:
        txt:        Raw text string.
        tokenizer:  A tokenizer with an `.encode()` method (e.g. tiktoken).
        max_length: Number of tokens per input sample (context window size).
        stride:     Step size between consecutive windows.
                    stride == max_length  → no overlap (efficient)
                    stride == 1           → max overlap (all possible windows)
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience function: tokenizes text and wraps it in a PyTorch DataLoader.

    Args:
        txt:         Raw text to train on.
        tokenizer:   Tokenizer compatible with tiktoken API.
        batch_size:  Number of samples per batch.
        max_length:  Context window size (tokens per sample).
        stride:      Sliding window step. Set equal to max_length for no overlap.
        shuffle:     Whether to shuffle samples each epoch.
        drop_last:   Drop the final incomplete batch (recommended for training).
        num_workers: Number of parallel data-loading workers.
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
