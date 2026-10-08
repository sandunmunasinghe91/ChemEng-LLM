"""
GPT training loop with validation, checkpointing,
learning rate scheduling, and logging.
"""

import logging
from dataclasses import asdict
import torch
from src.config import GPTConfig, small_config
from src.dataset import create_dataloader
from src.model import GPTModel

# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

# =====================================================
# Device
# =====================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

# =====================================================
# Validation
# =====================================================

@torch.no_grad()
def evaluate(
    model: GPTModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_batches: int = 20,
) -> float:
    """
    Estimate validation loss over a limited number
    of validation batches.
    """
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    if not losses:
        raise ValueError("No validation batches were evaluated.")

    return sum(losses) / len(losses)

# =====================================================
# Checkpoint
# =====================================================

def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: GPTConfig
) -> None:
    """
    Save model state, optimizer state,
    configuration, and training metadata.
    """

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = (config.checkpoint_dir / f"step_{step:06d}.pt")

    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
        },
        path
    )

    logger.info("Checkpoint saved → %s", path)

# =====================================================
# Training
# =====================================================

def train() -> None:

    # ------------ Configuration ------------
    config = small_config()
    torch.manual_seed(config.seed)

    # ------------ Device ------------
    device = get_device()
    logger.info("Using device: %s", device)

    # ------------ Data ------------
    train_loader = create_dataloader(
        bin_path=config.train_bin, context_length=config.context_length,
        batch_size=config.batch_size, shuffle=False,
    )
    val_loader = create_dataloader(
        bin_path=config.val_bin, context_length=config.context_length,
        batch_size=config.batch_size, shuffle=False,
    )

    # ------------ Model ------------
    model = GPTModel(config).to(device)
    logger.info("Parameters: %s {model.count_parameters():,}")

    # ------------ Optimizer parameter groups ------------
    decay_params = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for _, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate, betas=(0.9, 0.95))

    # ------------ Learning-rate scheduler ------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate / 10,
        )

    # ------------ Training loop ------------

    model.train()
    best_val_loss = float("inf")

    for step, (x, y) in enumerate(train_loader):

        if step >= config.max_steps:
            break

        x = x.to(device)
        y = y.to(device)

       
        # ------------ Forward pass ------------
        optimizer.zero_grad()
        _, loss = model(x, y)

        # ------------ Backward pass ------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        # ------------ Training logging ------------
        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step %4d | loss %.4f | lr %.2e",
                step, loss.item(), lr )

        # ------------ Validation ------------

        if step > 0 and step % config.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            logger.info(
                "EVAL step %4d | train %.4f | val %.4f",
                step, loss.item(), val_loss,
            )

            # ------------ Save best model ------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, config)
                logger.info("New best val loss: %.4f", best_val_loss)

        # ------------ Periodic checkpoint ------------

        elif step > 0 and step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, loss.item(), config)

    logger.info("Training complete | best val loss: %.4f", best_val_loss)


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":
    train()