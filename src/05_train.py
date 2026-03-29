"""
Training loop and loss functions for GPT-style language model pretraining.

Includes:
  - calc_loss_batch   : cross-entropy loss on a single batch
  - calc_loss_loader  : average loss over N batches from a DataLoader
  - evaluate_model    : compute train + val loss without gradient tracking
  - train_model_simple: full training loop with periodic eval & text sampling
  - plot_losses       : matplotlib dual-axis loss curve
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.generate import generate_text_simple, text_to_token_ids, token_ids_to_text


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Cross-entropy loss for a single (input, target) batch.

    Flattens (batch, seq_len, vocab_size) logits and (batch, seq_len) targets
    to 2-D and 1-D respectively before passing to nn.functional.cross_entropy.
    """
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )
    return loss


def calc_loss_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int = None,
) -> float:
    """
    Average cross-entropy loss over `num_batches` batches from a DataLoader.

    If num_batches is None, iterate over the entire DataLoader.
    Useful for estimating loss without running the full epoch.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()

    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Compute train and validation loss without updating gradients.

    Switches model to eval mode (disables dropout), runs eval_iter batches
    from each loader, then restores training mode.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    start_context: str,
) -> None:
    """Generate and print a short text sample during training."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
) -> tuple[list, list, list]:
    """
    Simple pretraining loop for a GPT-style language model.

    Every `eval_freq` steps:
      - Computes train and validation loss
      - Prints a generated text sample

    Args:
        model:         GPTModel instance (already moved to device).
        train_loader:  DataLoader for training data.
        val_loader:    DataLoader for validation data.
        optimizer:     Optimizer (AdamW recommended).
        device:        Target device.
        num_epochs:    Total training epochs.
        eval_freq:     Steps between evaluations.
        eval_iter:     Batches used per evaluation.
        start_context: Prompt string for text samples.
        tokenizer:     Tokenizer for decoding samples.

    Returns:
        train_losses, val_losses, track_tokens_seen
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen  = 0
    global_step  = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Generate a sample at end of each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: list,
    train_losses: list,
    val_losses: list,
) -> None:
    """
    Plot training and validation loss curves with a secondary tokens-seen axis.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses,   label="Validation loss", linestyle="--")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible — just to set axis scale
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
