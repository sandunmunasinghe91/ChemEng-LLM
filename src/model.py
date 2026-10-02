"""
GPT model components:
    FeedForward
    TransformerBlock
    GPTModel
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GPTConfig
from src.attention import CausalSelfAttention

# =====================================================
# FeedForward
# =====================================================

class FeedForward(nn.Module):
    """
    Position-wise feedforward network.
    E xpands to 4x embed_dim, applies GELU,then projects back. Applied after attention in each transformer block.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =====================================================
# TransformerBlock
# =====================================================

class TransformerBlock(nn.Module):
    """
    One transformer block.
    Pre-LayerNorm style (GPT-2):
        x = x + attention(norm(x))
        x = x + feedforward(norm(x))
    Residual connections let gradients flow directly through the network.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1          = nn.LayerNorm(config.embed_dim)
        self.attention    = CausalSelfAttention(config)
        self.ln2          = nn.LayerNorm(config.embed_dim)
        self.feed_forward = FeedForward(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

# =====================================================
# GPTModel
# =====================================================

class GPTModel(nn.Module):
    """
    GPT-2 language model.
    Given token IDs, predicts the next token at every position using causal self-attention.
    During training: pass targets to get loss. During inference: pass only token_ids, sample
    from the returned logits.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── embeddings ────────────────────────────────
        self.token_embedding    = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout            = nn.Dropout(config.dropout)

        # ── transformer blocks ────────────────────────
        self.blocks     = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # ── output head ───────────────────────────────
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head    = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # weight tying — input embedding shares weights
        # with output projection (saves ~20M parameters)
        self.lm_head.weight = self.token_embedding.weight

        # ── weight initialization ─────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights following GPT-2 paper.
        Linear and Embedding: normal(0, 0.02)
        LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            token_ids: (B, T) integer token IDs
            targets:   (B, T) next-token IDs for loss (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None
        """
        B, T = token_ids.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.config.context_length}."
            )

        # ── embeddings ────────────────────────────────
        positions = torch.arange(T, device=token_ids.device)
        x = self.dropout(
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # ── transformer blocks ────────────────────────
        for block in self.blocks:
            x = block(x)

        # ── output ────────────────────────────────────
        logits = self.lm_head(self.final_norm(x))

        # ── loss (training only) ──────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

