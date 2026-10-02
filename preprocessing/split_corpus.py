"""
split_corpus.py

Split filtered corpus into train and validation sets.

Features:
- Splits whole documents to avoid data leakage
- Splits each source separately and independently
- Uses approximate word count to target VAL_RATIO
- Derives a unique deterministic seed per source
- Recreates split folders on every run (no stale files)
- Saves a manifest for full reproducibility

"""

import json
import logging
import random
import shutil
from pathlib import Path


# =====================================================
# 1. Configuration
# =====================================================


SOURCE_DIRS = {
    "arxiv": Path("data/filtered/arxiv_domain_candidates"),
    "wikipedia": Path("data/processed/wikipedia"),
}

SPLIT_DIR = Path("data/splits")
MANIFEST  = Path("reports/split_manifest.json")

VAL_RATIO = 0.05   # ~5% of each source's words go to validation
SEED      = 42     # base seed — unique seed derived per source


# =====================================================
# 2. Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# =====================================================
# 3. Helper functions
# =====================================================

def count_words(file_path: Path) -> int:
    """
    Count words in a text file.

    Word count is used as an approximate measure of
    document size before tokenization.
    Returns 0 if the file cannot be read.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return len(text.split())
    except OSError as e:
        logger.warning("Could not read %s: %s", file_path.name, e)
        return 0


def prepare_split_dirs() -> None:
    """
    Remove old train/val directories and recreate them.

    Wiping on every run prevents stale files from a previous
    split appearing in both train and validation sets.
    """
    for split_name in ("train", "val"):
        split_dir = SPLIT_DIR / split_name

        if split_dir.exists():
            shutil.rmtree(split_dir)
            logger.info("Removed old directory: %s", split_dir)

        split_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", split_dir)


def copy_files(files: list[Path], destination: Path, label: str) -> None:
    """
    Copy a list of files into a destination directory.

    Logs progress every 200 files and confirms completion.
    """
    destination.mkdir(parents=True, exist_ok=True)

    for i, file_path in enumerate(files, start=1):
        shutil.copy2(file_path, destination / file_path.name)

        if i % 200 == 0:
            logger.info("%s: copied %d / %d files", label, i, len(files))

    # always log completion — fires even when len(files) < 200
    logger.info("%s: done — %d files copied to %s", label, len(files), destination)


# =====================================================
# 4. Split one source
# =====================================================

def split_source(source_name: str, source_dir: Path) -> dict:
    """
    Split one corpus source into train and validation sets.

    The split is done by whole document to avoid data leakage.
    Validation receives approximately VAL_RATIO of total words.

    Each source gets its own deterministic seed derived from
    SEED + source_name so that each source shuffles independently
    while the result is still fully reproducible.

    Args:
        source_name: label used for logging and output dirs.
        source_dir:  directory containing cleaned .txt files.

    Returns:
        dict with total_files, total_words, train, val statistics.
    """

    # ── validate source ───────────────────────────────────────────
    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"[{source_name}] Source directory not found: {source_dir}"
        )

    files = sorted(source_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"[{source_name}] No .txt files found in: {source_dir}"
        )

    logger.info("%s: found %d files in %s", source_name, len(files), source_dir)

    # ── count words per file ──────────────────────────────────────
    file_word_counts: dict[Path, int] = {}

    for i, file_path in enumerate(files, start=1):
        word_count = count_words(file_path)

        if word_count > 0:
            file_word_counts[file_path] = word_count

        if i % 200 == 0:
            logger.info("%s: counted %d / %d files", source_name, i, len(files))

    usable_files  = list(file_word_counts.keys())
    skipped_files = len(files) - len(usable_files)

    if skipped_files > 0:
        logger.warning(
            "%s: skipped %d empty or unreadable files",
            source_name,
            skipped_files,
        )

    total_words = sum(file_word_counts.values())

    if total_words == 0:
        raise ValueError(
            f"[{source_name}] No usable text found in: {source_dir}"
        )

    logger.info(
        "%s: %d usable files | %s words",
        source_name,
        len(usable_files),
        f"{total_words:,}",
    )

    # ── reproducible shuffle ──────────────────────────────────────
    # Derive a unique seed per source so each source shuffles
    # independently. Both are fully deterministic from SEED alone.
    source_seed    = f"{SEED}-{source_name}"
    rng            = random.Random(source_seed)
    shuffled_files = usable_files.copy()
    rng.shuffle(shuffled_files)

    logger.info("%s: shuffled with seed '%s'", source_name, source_seed)

    # ── split by word budget ──────────────────────────────────────
    # Walk through shuffled files, filling the val bucket until
    # it reaches VAL_RATIO of total words.
    # Simple threshold — predictable and easy to reason about.
    val_target  = total_words * VAL_RATIO   # float, no premature rounding
    val_words   = 0
    train_files = []
    val_files   = []

    for file_path in shuffled_files:
        words = file_word_counts[file_path]

        if val_words < val_target:
            val_files.append(file_path)
            val_words += words
        else:
            train_files.append(file_path)

    train_words = total_words - val_words

    # ── output directories ────────────────────────────────────────
    train_dir = SPLIT_DIR / "train" / source_name
    val_dir   = SPLIT_DIR / "val"   / source_name

    # ── copy files ────────────────────────────────────────────────
    logger.info("%s: copying training files...", source_name)
    copy_files(train_files, train_dir, f"{source_name}-train")

    logger.info("%s: copying validation files...", source_name)
    copy_files(val_files, val_dir, f"{source_name}-val")

    # ── build result ──────────────────────────────────────────────
    result = {
        "source_dir":    str(source_dir),
        "source_seed":   source_seed,
        "total_files":   len(usable_files),
        "total_words":   total_words,
        "skipped_files": skipped_files,
        "train": {
            "directory":  str(train_dir),
            "n_files":    len(train_files),
            "n_words":    train_words,
            "percentage": round(100 * train_words / total_words, 2),
            "files":      [f.name for f in train_files],
        },
        "val": {
            "directory":  str(val_dir),
            "n_files":    len(val_files),
            "n_words":    val_words,
            "percentage": round(100 * val_words / total_words, 2),
            "files":      [f.name for f in val_files],
        },
    }

    logger.info(
        "%s split complete | Train: %d files (%s words, %.2f%%) | "
        "Val: %d files (%s words, %.2f%%)",
        source_name,
        result["train"]["n_files"], f"{train_words:,}", result["train"]["percentage"],
        result["val"]["n_files"],   f"{val_words:,}",   result["val"]["percentage"],
    )

    return result


# =====================================================
# 5. Main
# =====================================================

def main() -> None:

    logger.info(
        "Starting corpus split | sources: %s | val_ratio: %.2f | seed: %d",
        list(SOURCE_DIRS.keys()),
        VAL_RATIO,
        SEED,
    )

    # wipe old split directories before starting
    prepare_split_dirs()

    # ── split each source independently ──────────────────────────
    results: dict[str, dict] = {}

    for source_name, source_dir in SOURCE_DIRS.items():
        logger.info("=" * 50)
        logger.info("Processing source: %s", source_name)
        results[source_name] = split_source(source_name, source_dir)

    # ── combined totals across all sources ───────────────────────
    combined_train_words = sum(r["train"]["n_words"] for r in results.values())
    combined_val_words   = sum(r["val"]["n_words"]   for r in results.values())
    combined_total_words = combined_train_words + combined_val_words
    combined_train_files = sum(r["train"]["n_files"] for r in results.values())
    combined_val_files   = sum(r["val"]["n_files"]   for r in results.values())

    # ── save manifest ─────────────────────────────────────────────
    manifest = {
        "seed":      SEED,
        "val_ratio": VAL_RATIO,
        "sources":   results,
        "combined": {
            "total_words": combined_total_words,
            "train_words": combined_train_words,
            "val_words":   combined_val_words,
            "train_files": combined_train_files,
            "val_files":   combined_val_files,
            "train_pct":   round(100 * combined_train_words / combined_total_words, 2),
            "val_pct":     round(100 * combined_val_words   / combined_total_words, 2),
        },
    }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest saved to %s", MANIFEST)

    # ── print final summary ───────────────────────────────────────
    print()
    print("=" * 60)
    print("CORPUS SPLIT SUMMARY")
    print("=" * 60)

    for source_name, result in results.items():
        print(f"\n  {source_name.upper()}")
        print(f"  {'─' * 30}")
        print(f"  Total files  : {result['total_files']}")

        if result["skipped_files"] > 0:
            print(f"  Skipped      : {result['skipped_files']} (empty or unreadable)")

        print(f"  Total words  : {result['total_words']:,}")
        print(f"  Train files  : {result['train']['n_files']}")
        print(f"  Train words  : {result['train']['n_words']:,}  ({result['train']['percentage']:.2f}%)")
        print(f"  Val files    : {result['val']['n_files']}")
        print(f"  Val words    : {result['val']['n_words']:,}  ({result['val']['percentage']:.2f}%)")

    print(f"\n  COMBINED TOTALS")
    print(f"  {'─' * 30}")
    print(f"  Total words  : {combined_total_words:,}")
    print(f"  Train words  : {combined_train_words:,}  ({manifest['combined']['train_pct']:.2f}%)")
    print(f"  Val words    : {combined_val_words:,}  ({manifest['combined']['val_pct']:.2f}%)")
    print(f"  Train files  : {combined_train_files}")
    print(f"  Val files    : {combined_val_files}")
    print()
    print(f"  Manifest     : {MANIFEST}")
    print(f"  Train dir    : {SPLIT_DIR / 'train'}")
    print(f"  Val dir      : {SPLIT_DIR / 'val'}")
    print()


if __name__ == "__main__":
    main()