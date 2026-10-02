"""
Central configuration for GPT model training.
All hyperparameters live here so model, training,data loading, and generation code can use the same configuration.

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.embed_dim)
"""

from dataclasses import dataclass, field
from pathlib import Path

# =====================================================
# GPT configuration
# =====================================================

@dataclass
class GPTConfig:
    """
    Configuration for model architecture, training, data loading, and file paths.
    Values are validated when the object is created so configuration mistakes are caught early.
    """
    # ── tokenizer / vocabulary ────────────────────────────────────
    vocab_size: int = 50_257

    # ── sequence length ───────────────────────────────────────────
    context_length: int = 256

    # ── model architecture ────────────────────────────────────────
    embed_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1

    # Derived automatically from embed_dim // n_heads
    head_dim: int = field(init=False)

    # ── training ──────────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5_000
    eval_every: int = 500
    save_every: int = 1_000
    grad_clip: float = 1.0

    # ── reproducibility ───────────────────────────────────────────
    seed: int = 42

    # ── paths ─────────────────────────────────────────────────────
    train_bin: Path = Path("data/tokenized/train.bin")
    val_bin: Path = Path("data/tokenized/val.bin")
    checkpoint_dir: Path = Path("checkpoints")

    # =================================================
    # Validation
    # =================================================

    def __post_init__(self) -> None:
        """
        Validate configuration values and calculate derived values.
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")
        if self.context_length < 8:
            raise ValueError(
                f"context_length={self.context_length} is too small. Minimum is 8.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be greater than 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be greater than 0.")
        if self.save_every <= 0:
            raise ValueError("save_every must be greater than 0.")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be greater than 0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by "
                f"n_heads ({self.n_heads}).")
        
        # derived value — computed after validation
        self.head_dim = (self.embed_dim // self.n_heads)


    # =================================================
    # Display
    # =================================================

    def display(self) -> None:
        """
        Print a formatted configuration summary.
        """
        total_params = (self._estimate_params())
        print()
        print("=" * 50)
        print("MODEL CONFIGURATION")
        print("=" * 50)
        print(f"  {'─' * 30}")
        print("  ARCHITECTURE")
        print(f"  {'─' * 30}")
        print(f"  vocab_size     : {self.vocab_size:,}")
        print(f"  context_length : {self.context_length}")
        print(f"  embed_dim      : {self.embed_dim}")
        print(f"  n_heads        : {self.n_heads}")
        print(f"  head_dim       : {self.head_dim}")
        print(f"  n_layers       : {self.n_layers}")
        print(f"  dropout        : {self.dropout}")
        print(f"  est. params    : ~{total_params}M")
        print(f"  {'─' * 30}")
        print("  TRAINING")
        print(f"  {'─' * 30}")
        print(f"  batch_size     : {self.batch_size}")
        print(f"  learning_rate  : {self.learning_rate}")
        print(f"  weight_decay   : {self.weight_decay}")
        print(f"  max_steps      : {self.max_steps:,}")
        print(f"  eval_every     : {self.eval_every}")
        print(f"  save_every     : {self.save_every}")
        print(f"  grad_clip      : {self.grad_clip}")
        print(f"  seed           : {self.seed}")
        print(f"  {'─' * 30}")
        print("  PATHS")
        print(f"  {'─' * 30}")
        print(f"  train_bin      : {self.train_bin}")
        print(f"  val_bin        : {self.val_bin}")
        print(f"  checkpoint_dir : {self.checkpoint_dir}")
        print()


    # =================================================
    # Parameter estimate
    # =================================================

    def _estimate_params(self) -> str:
        """
        Estimate model parameter count in millions.
        This is only a rough estimate. It does not include every bias, LayerNorm parameter,or implementation detail.
        """

        token_embedding = self.vocab_size* self.embed_dim
        position_embedding = self.context_length * self.embed_dim
        per_layer = 4 * self.embed_dim * self.embed_dim + 8 * self.embed_dim * self.embed_dim
        total = token_embedding + position_embedding + (per_layer * self.n_layers)
        return f"{total / 1e6:.1f}"

# =====================================================
# Configuration presets
# =====================================================

def get_config() -> GPTConfig:
    """
    Default configuration for initial full training experiments.
    Approximately 30M parameters depending on the final model implementation.
    """
    return GPTConfig()

def small_config() -> GPTConfig:
    """
    Small configuration for quick debugging and CPU/MPS testing.
    """
    return GPTConfig( 
        embed_dim=128, n_heads=4, n_layers=2, context_length=64,
        batch_size=4, max_steps=100, eval_every=20,save_every=50,
    )


def medium_config() -> GPTConfig:
    """
    GPT-2-small-style architecture.
    Reduce batch size or context length when training on memory-constrained hardware.
    """

    return GPTConfig(
        embed_dim=768, n_heads=12, n_layers=12,context_length=1024, 
        batch_size=8,max_steps=50_000, eval_every=1_000,save_every=5_000,
    )
