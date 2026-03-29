"""
Fine-tuning a pretrained GPT model for text classification (spam detection).

Strategy:
  1. Load pretrained GPT-2 weights into GPTModel
  2. Freeze all parameters
  3. Replace the LM output head with a binary classification head
  4. Unfreeze the final TransformerBlock + LayerNorm for fine-tuning
  5. Train with cross-entropy on binary labels (ham=0, spam=1)

Dataset: UCI SMS Spam Collection
  https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SpamDataset(Dataset):
    """
    Dataset for SMS spam classification.

    Tokenizes each message, truncates to max_length, and pads shorter
    sequences with the EOS token (pad_token_id=50256).

    Args:
        csv_file:     Path to CSV with columns ['Label', 'Text'].
        tokenizer:    tiktoken-compatible tokenizer.
        max_length:   Maximum token sequence length. If None, uses the
                      longest sequence in this split.
        pad_token_id: Token ID used for padding (GPT-2 EOS = 50256).
    """

    def __init__(self, csv_file: str, tokenizer, max_length: int = None, pad_token_id: int = 50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = max(len(enc) for enc in self.encoded_texts)
        else:
            self.max_length = max_length

        # Truncate and pad
        self.encoded_texts = [
            enc[:self.max_length] + [pad_token_id] * max(0, self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label   = self.data.iloc[idx]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label,   dtype=torch.long),
        )


def calc_accuracy_loader(data_loader, model: nn.Module, device: torch.device, num_batches: int = None) -> float:
    """Compute classification accuracy over `num_batches` batches."""
    model.eval()
    correct, total = 0, 0

    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch  = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # last token's logits
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == target_batch).sum().item()
        total   += target_batch.size(0)

    return correct / total


def calc_loss_batch_cls(input_batch, target_batch, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Cross-entropy loss using only the last token's logits (classification)."""
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def train_classifier_simple(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> tuple:
    """
    Fine-tuning loop for binary classification.

    Returns train_losses, val_losses, train_accs, val_accs, examples_seen.
    """
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    examples_seen = 0
    global_step   = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.size(0)
            global_step   += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tr_loss = sum(
                        calc_loss_batch_cls(xb, yb, model, device).item()
                        for i, (xb, yb) in enumerate(train_loader) if i < eval_iter
                    ) / eval_iter
                    v_loss = sum(
                        calc_loss_batch_cls(xb, yb, model, device).item()
                        for i, (xb, yb) in enumerate(val_loader) if i < eval_iter
                    ) / eval_iter
                model.train()
                train_losses.append(tr_loss)
                val_losses.append(v_loss)
                print(f"Ep {epoch+1} Step {global_step:05d} | Train {tr_loss:.3f} | Val {v_loss:.3f}")

        tr_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        v_acc  = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)
        train_accs.append(tr_acc)
        val_accs.append(v_acc)
        print(f"  → Train acc: {tr_acc*100:.1f}%  |  Val acc: {v_acc*100:.1f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(
    text: str,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_length: int = None,
    pad_token_id: int = 50256,
) -> str:
    """
    Classify a single text string as 'spam' or 'not spam'.

    Args:
        text:       Raw text to classify.
        model:      Fine-tuned GPT classifier.
        tokenizer:  tiktoken tokenizer.
        device:     Target device.
        max_length: Truncation length (should match training max_length).
    """
    model.eval()
    input_ids = tokenizer.encode(text)
    context_length = model.pos_emb.weight.shape[0]

    if max_length and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    elif max_length:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_ids = input_ids[:context_length]
    tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[:, -1, :]

    label = torch.argmax(logits, dim=-1).item()
    return "spam" if label == 1 else "not spam"
