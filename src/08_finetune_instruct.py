"""
Instruction fine-tuning (Supervised Fine-Tuning / SFT) for GPT-style models.

Uses the Alpaca prompt format:
    Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    <instruction text>

    ### Input:       ← optional
    <input text>

    ### Response:
    <model response>

Key design choices:
  - Pad shorter sequences to batch max_length using token 50256 (EOS)
  - Set target labels for padding tokens to -100 (ignored by cross-entropy)
  - Use a custom collate function to handle variable-length instructions
  - Optionally truncate sequences to a max context length
"""

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


def format_input(entry: dict) -> str:
    """
    Format a dataset entry into Alpaca-style instruction prompt.

    Args:
        entry: dict with keys 'instruction', 'input' (optional), 'output'.

    Returns:
        Formatted prompt string (without the response).
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Each item is a tokenized sequence: [instruction + input + response],
    terminated with the EOS token.

    Args:
        data:      List of instruction dicts with keys 'instruction', 'input', 'output'.
        tokenizer: tiktoken-compatible tokenizer.
    """

    def __init__(self, data: list, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]


def custom_collate_fn(
    batch: list,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length instruction sequences.

    Pads sequences to the length of the longest sequence in the batch + 1.
    Sets padding targets to `ignore_index` (-100) so they don't contribute
    to the cross-entropy loss.

    Args:
        batch:               List of token ID lists (variable length).
        pad_token_id:        Token ID for padding inputs (GPT-2 EOS = 50256).
        ignore_index:        Target value to ignore in loss (-100).
        allowed_max_length:  Clip sequences longer than this.
        device:              Move tensors to this device.

    Returns:
        (inputs, targets) — both shape (batch_size, max_seq_len).
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # append EOS as final target

        # Pad to batch max length
        pad_len = batch_max_length - len(new_item)
        new_item.extend([pad_token_id] * pad_len)

        if allowed_max_length:
            new_item = new_item[:allowed_max_length + 1]

        inputs  = torch.tensor(new_item[:-1], dtype=torch.long)
        targets = torch.tensor(new_item[1:],  dtype=torch.long)

        # Mask padding tokens in targets
        mask = targets == pad_token_id
        targets[mask] = ignore_index

        if allowed_max_length:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def create_instruction_dataloader(
    data: list,
    tokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    allowed_max_length: int = 1024,
) -> DataLoader:
    """
    Build a DataLoader for instruction fine-tuning with the custom collate function.
    """
    dataset = InstructionDataset(data, tokenizer)
    collate = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
