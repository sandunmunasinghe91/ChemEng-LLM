"""
Causal multi-head self-attention for the GPT model.
Shapes used throughout:
    B = batch size
    T = sequence length
    C = embed_dim
    H = n_heads
    D = head_dim = C // H
"""

import torch
import torch.nn as nn
from src.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Input shape:
        (batch_size, seq_length, embed_dim)
    Output shape:
        (batch_size, seq_length, embed_dim)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # -------------------------------------------------
        # Q, K, V projections
        # -------------------------------------------------
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear( self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # -------------------------------------------------
        # Output projection
        # -------------------------------------------------
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # -------------------------------------------------
        # Dropout
        # -------------------------------------------------
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # -------------------------------------------------
        # Causal mask
        # -------------------------------------------------
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask",mask.view(1,1,config.context_length,config.context_length,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C = x.shape

        
        # --------------- Create Q, K, V ---------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --------------- Split embedding dimension into heads ---------------
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # --------------- Move heads before sequence dimension ---------------
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --------------- Compute attention scores ---------------
        attention_scores = (q @ k.transpose(-2, -1))
        attention_scores = (attention_scores / (self.head_dim ** 0.5))

        # --------------- Apply causal mask ---------------
        mask = self.causal_mask[:, :, :T, :T]
        attention_scores = (attention_scores.masked_fill(mask == 0, float("-inf")))

        # --------------- Convert scores to probabilities ---------------
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = (self.attn_dropout(attention_weights))

        # --------------- Weighted sum of values ---------------
        output = (attention_weights @ v)
        
        # --------------- Merge attention heads ---------------
        output = output.transpose( 1, 2)
        output = output.contiguous().view(B,T,C)
        
        # --------------- Final projection ---------------
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output