"""
Tokenize train and validation corpus splits using the GPT-2 tokenizer from tiktoken.

The same tokenizer is used for both trainingand validation data.
"""

import logging
from pathlib import Path
import numpy as np
import tiktoken

# =====================================================
# 1. Configuration
# =====================================================

SPLIT_DIR = Path("data/splits")
TOKENIZED_DIR = Path("data/tokenized")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"

TOKENIZER_NAME = "gpt2"

# =====================================================
# 2. Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# =====================================================
# 3. Collect text files
# =====================================================
def get_text_files(directory: Path) -> list[Path]:
    """
    Collect all .txt files recursively.

    """  

    # If there is no directory raise an error.  
    if not directory.is_dir():
        raise NotADirectoryError(f'Directory not found: {directory}')

    # Recursively search for .tex files and sort them. 
    files = sorted(directory.rglob("*.txt"))

    # if there is no files raise an error
    if not files:
        raise FileNotFoundError(f'No .txt files found: {directory}')

    
    logger.info("Found %d files in %s",len(files),directory)

    return files

# =====================================================
# 4. Load tokenizer
# =====================================================

def load_tokenizer():
    """
    Load the pretrained GPT-2 tokenizer from tiktoken.
    """

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

    logger.info("Loaded tokenizer: %s",TOKENIZER_NAME)

    logger.info("Vocabulary size: %d",tokenizer.n_vocab)

    return tokenizer
#%%
# =====================================================
# 5. Tokenize and save
# =====================================================

def tokenize_and_save(files: list[Path],tokenizer,output_path: Path,split_label: str) -> int:
    """
    Tokenize all documents and save token IDs as a flat uint16 binary file.

    Each document is separated by GPT-2's end-of-text token.

    Returns:
        Total number of token IDs written.
    """
    # Create TOKENIZED_DIR
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True) 

    # Initialize the-end-of token ID
    eot_id = tokenizer.eot_token
    logger.info("%s: EOS token ID = %d",split_label,eot_id)

    # Check whether GPT-2 Vocab fits inside unit16
    if tokenizer.n_vocab > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary is too large for uint16 storage")
    
    total_tokens = 0

    # Open output file once and stream tokenized
    # documents directly to disk.
    with open(output_path, "wb") as output_file:
        for i, file_path in enumerate(files,start=1):
            try:
                text = file_path.read_text(encoding="utf-8",errors="ignore")
            except OSError as e:
                logger.warning("%s: could not read %s: %s",split_label,file_path.name,e)
                continue

            if not text.strip():
                logger.warning("%s: skipping empty file %s",split_label,file_path.name)

                continue

            # Convert text → token IDs
            token_ids = tokenizer.encode(text, allowed_special=set(),disallowed_special=())

            # Add EOS token between documents
            token_ids.append(eot_id)

            # Convert token IDs to compact uint16 array
            token_array = np.asarray(token_ids, dtype=np.uint16,)

            # Write this document directly to the
            # binary file instead of keeping the
            # entire corpus in memory.
            token_array.tofile(output_file)
            total_tokens += len(token_ids)

            if i % 200 == 0:
                logger.info("%s: tokenized %d / %d files | ""%s tokens written", split_label,i,
                    len(files),
                    f"{total_tokens:,}",
                )

    file_size_mb = (output_path.stat().st_size / 1_000_000)

    logger.info( "%s: saved %s tokens to %s (%.1f MB)", split_label, f"{total_tokens:,}", output_path, file_size_mb)

    return total_tokens

# =====================================================
# 6. Main
# =====================================================

def main() -> None:

    logger.info("Starting corpus tokenization")

    # Load tokenizer
    tokenizer = load_tokenizer()
   
    # Find train and validation files
    train_files = get_text_files(TRAIN_DIR)
    val_files = get_text_files(VAL_DIR)

    
    # Tokenize training data
    train_output = (TOKENIZED_DIR/ "train.bin")
    train_tokens = tokenize_and_save(
        files=train_files,
        tokenizer=tokenizer,
        output_path=train_output,
        split_label="train",
    )

    # Tokenize validation data
    val_output = (TOKENIZED_DIR / "val.bin")
    val_tokens = tokenize_and_save(
        files=val_files,
        tokenizer=tokenizer,
        output_path=val_output,
        split_label="val",
    )

  
    # Summary
   
    total_tokens = (train_tokens + val_tokens)
    print()
    print("=" * 55)
    print("TOKENIZATION SUMMARY")
    print("=" * 55)
    print(f"  Tokenizer     : " f"{TOKENIZER_NAME}")
    print(f"  Vocabulary    : " f"{tokenizer.n_vocab:,}")
    print(f"  Train tokens  : " f"{train_tokens:,}")
    print(f"  Val tokens    : " f"{val_tokens:,}")
    print(f"  Total tokens  : " f"{total_tokens:,}")
    print(f"  Train binary  : " f"{train_output}")
    print(f"  Val binary    : " f"{val_output}")
    print()

if __name__ == "__main__":

    main()
