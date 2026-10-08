# ChemLLM

A GPT language model for chemical engineering, built from scratch — from data collection to training.

Pre-trained on a curated corpus of chemical engineering literature (arXiv papers + Wikipedia articles) spanning reaction engineering, process control, thermodynamics, separation processes, and related topics.

## Project Structure

```
ChemLLM/
├── data_collection/           # Scraping and raw text extraction
│   ├── scrape_arxiv.py            # Download arXiv PDFs
│   ├── scrape_arxiv_latex.py      # Download arXiv LaTeX sources
│   └── scrape_wikipedia.py        # Fetch Wikipedia articles
│
├── preprocessing/             # Cleaning, filtering, splitting, tokenization
│   ├── clean_arxiv.py             # Extract text from PDFs (Docling)
│   ├── clean_arxive_latex.py      # Parse and clean LaTeX sources
│   ├── clean_wikipedia.py         # Clean Wikipedia markup
│   ├── analyze.py                 # Quality analysis with CSV reporting
│   ├── filter_corpus.py           # Domain relevance filtering
│   ├── split_corpus.py            # Train/val split by document
│   └── tokenize_corpus.py         # Tokenize to uint16 binaries
│
├── src/                       # Model implementation
│   ├── config.py                  # GPT configuration (dataclass + presets)
│   ├── attention.py               # Causal multi-head self-attention
│   ├── model.py                   # FeedForward, TransformerBlock, GPTModel
│   ├── dataset.py                 # Memory-mapped dataset and DataLoader
│   └── train.py                   # Training loop with validation and checkpointing
│
├── data/
│   ├── raw/                       # Downloaded papers and articles
│   ├── processed/                 # Cleaned plain text
│   ├── filtered/                  # Domain-filtered corpus
│   ├── splits/                    # Train/val document splits
│   └── tokenized/                 # Binary token files (train.bin, val.bin)
│
├── reports/                   # Quality and filtering CSV reports
└── notebooks/                 # Exploration notebooks
```

## Data Pipeline

Five stages, each runnable as a standalone script:

### 1. Collection

Scientific literature scraped using ~50 queries across 10 chemical engineering sub-domains:

| Category | Example Topics |
|---|---|
| Fundamentals | Thermodynamics, transport phenomena, fluid mechanics |
| Reaction Engineering | Reactor design, kinetics, CSTR, plug flow |
| Separation Processes | Distillation, VLE, membrane separation |
| Process Engineering | Process design, simulation, optimization |
| Process Control | PID control, MPC, process dynamics |
| Plant Monitoring | Fault detection, anomaly detection, sensor diagnostics |
| Equipment | Heat exchangers, distillation columns, pressure vessels |
| Safety | Hazard analysis, risk assessment, accident prevention |
| Predictive Maintenance | Fault diagnosis, RUL estimation |
| AI in ChemE | ML for processes, digital twins, data-driven optimization |

**Sources:** arXiv (PDFs and LaTeX source archives, up to 100 papers per query) and Wikipedia (MediaWiki API, up to 50 articles per query).

### 2. Cleaning

- **LaTeX** — Resolves multi-file archives, strips metadata/bibliography/figures while preserving equations and scientific prose
- **PDF** — Text extraction via [Docling](https://github.com/DS4SD/docling) with formula enrichment and table extraction
- **Wikipedia** — Removes navigation/references, resolves equation fragments, converts wiki markup

### 3. Quality Assessment

Documents are evaluated on minimum length, LaTeX residue, broken references, prose ratio, garbled text, and repetition. Each is classified as **accepted**, **review**, or **rejected**.

### 4. Domain Filtering

Accepted papers are scored against chemical engineering keyword lists and sorted into domain candidates or review buckets.

### 5. Splitting and Tokenization

- Train (~95%) / validation (~5%) split by whole documents to prevent data leakage
- Tokenized with GPT-2 tokenizer (`tiktoken`, vocab size 50,257)
- Written to flat `uint16` binary files for memory-mapped loading

| Split | Files | Words | Tokens | Binary Size |
|---|---|---|---|---|
| Train | 4,147 | 12.5M | ~19M | 37 MB |
| Val | 231 | 664K | ~1M | 2.1 MB |

## Model

Decoder-only GPT with pre-LayerNorm (GPT-2 style), weight-tied embeddings, and cosine LR scheduling.

| Parameter | Default | Small | Medium (GPT-2) |
|---|---|---|---|
| Embedding dim | 384 | 128 | 768 |
| Attention heads | 6 | 4 | 12 |
| Layers | 6 | 2 | 12 |
| Context length | 256 | 64 | 1,024 |
| Parameters (est.) | ~30M | ~3M | ~124M |

**Components:**

- `GPTConfig` — Dataclass with validation for all hyperparameters, training settings, and paths. Includes `small`, `medium`, and default presets.
- `CausalSelfAttention` — Multi-head causal self-attention with separate Q/K/V projections and causal masking
- `FeedForward` — Position-wise FFN (4x expansion with GELU)
- `TransformerBlock` — Pre-norm attention + FFN with residual connections
- `GPTModel` — Full model with token/position embeddings, stacked transformer blocks, weight-tied output head, and GPT-2 weight initialization
- `GPTDataset` — Memory-mapped dataset for `uint16` token binaries
- Training loop — AdamW with weight decay separation, cosine annealing, gradient clipping, periodic validation, and best-model checkpointing

## Setup

```bash
git clone https://github.com/yourusername/ChemLLM.git
cd ChemLLM

conda create -n chemllm python=3.11
conda activate chemllm

pip install -r requirements.txt
```

Requires Python 3.11+ and Conda.

## Usage

```bash
# Data pipeline
python -m data_collection.scrape_arxiv_latex    # 1. Scrape arXiv LaTeX sources
python -m preprocessing.clean_arxive_latex      # 2. Clean LaTeX to plain text
python -m data_collection.scrape_wikipedia      # 3. Scrape Wikipedia articles
python -m preprocessing.clean_wikipedia         # 4. Clean Wikipedia articles
python -m preprocessing.analyze                 # 5. Quality analysis
python -m preprocessing.filter_corpus           # 6. Domain filtering
python -m preprocessing.split_corpus            # 7. Train/val split
python -m preprocessing.tokenize_corpus         # 8. Tokenize

# Training
python -m src.train
```

## Roadmap

- [x] Data collection (arXiv + Wikipedia)
- [x] Text extraction and cleaning pipelines
- [x] Quality assessment and domain filtering
- [x] Train/validation splitting and tokenization
- [x] GPT model architecture (attention, FFN, transformer blocks)
- [x] Full model assembly with weight tying
- [x] Training loop with checkpointing and LR scheduling
- [ ] Text generation / sampling
- [ ] Evaluation and benchmarking

## License

This project is for research and educational purposes.
