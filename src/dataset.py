import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =====================================================
# Logging
# =====================================================

logger = logging.getLogger(__name__)

# =====================================================
# 1. Dataset
# =====================================================

class GPTDataset(Dataset):
    """
    Dataset for causal language modeling.
    Reads a flat uint16 token binary file and creates fixed-length input/target pairs.
    Uses np.memmap so the full corpus is never loaded into RAM. Only requested token windows are accessed as needed.
    """

    def __init__(self, bin_path: Path, context_length: int):
        self.bin_path = bin_path
        self.context_length = context_length

        if context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        
        if not bin_path.is_file():
            raise FileNotFoundError( f"Token file not found: {bin_path}\n" f"Run tokenize_corpus.py first.")

        # Memory-map token file instead of loading
        # the entire corpus into RAM.
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")

        if len(self.tokens) <= context_length:
            raise ValueError(
                f"Token file too small: "
                f"{len(self.tokens):,} tokens, "
                f"context_length={context_length:,}"
            )

        logger.info(
            "Loaded %s | %s tokens | %s windows",
            bin_path.name,
            f"{len(self.tokens):,}",
            f"{len(self):,}",
        )

    def __len__(self) -> int:
        """
        Return the number of valid starting positions in the token stream.
        """

        return (len(self.tokens)- self.context_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one input-target pair.
        x:
            tokens[
                index :
                index + context_length
            ]

        y:
            tokens[
                index + 1 :
                index + context_length + 1
            ]

        The target sequence is shifted one token to the right so the model learns next-token prediction.
        """

        x = self.tokens[index: index + self.context_length]

        y = self.tokens[index + 1: index + self.context_length + 1]

        # np.memmap slices are read-only views.
        # copy() creates a normal writable NumPy array.
        
        # torch.long = int64, which is required
        # for token indices used by nn.Embedding.
        x = torch.tensor(x.copy(), dtype=torch.long)

        y = torch.tensor(y.copy(), dtype=torch.long)

        return x, y


# =====================================================
# 2. DataLoader factory
# =====================================================

def create_dataloader(
    bin_path: Path,
    context_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a tokenized corpus split.
    Args:
        bin_path: Path to the .bin token file.
        context_length: Number of tokens in each sequence.
        batch_size: Number of sequences per batch.
        shuffle: True for training data. False for validation data.
        num_workers: Number of worker processes used to load data.
            0 means loading happens in the main process.
    Returns:
        DataLoader yielding:
            x.shape = (batch_size, context_length)
            y.shape = (batch_size, context_length)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = GPTDataset(bin_path=bin_path, context_length=context_length)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )

    logger.info(
        "DataLoader | %s | "
        "%d batches | "
        "batch_size=%d | "
        "context_length=%d | "
        "shuffle=%s",
        bin_path.stem,
        len(loader),
        batch_size,
        context_length,
        shuffle,
    )

    return loader

