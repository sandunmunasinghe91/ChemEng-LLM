# ChemLLM

A domain-specific GPT language model for chemical engineering, built from scratch — from data collection to model training.

The goal is to pre-train a transformer on a curated corpus of chemical engineering literature so it can understand and generate text about reaction engineering, process control, thermodynamics, separation processes, and related topics.

## Project Status

The **data pipeline** is complete: scraping, cleaning, quality filtering, train/val splitting, and tokenization are all functional. The **model architecture** is in progress — configuration and causal self-attention are implemented. Next up: full transformer blocks, training loop, and evaluation.

## Architecture

```
ChemLLM/
├── data_collection/              # Scraping and raw text extraction
│   ├── scrape_arxiv.py               # Download arXiv PDFs by domain queries
│   ├── scrape_arxiv_latex.py         # Download arXiv LaTeX source archives
│   ├── scrape_wikipedia.py           # Fetch Wikipedia articles via MediaWiki API
│   ├── clean_arxiv.py                # Extract text from PDFs using Docling
│   ├── clean_arxive_latex.py         # Parse, combine, and clean LaTeX sources
│   └── clean_wikipedia.py            # Clean Wikipedia markup and equations
│
├── preprocessing/                # Corpus quality, splitting, and tokenization
│   ├── analyze.py                    # Quality analysis with CSV reporting
│   ├── filter_corpus.py              # Domain relevance filtering
│   ├── split_corpus.py               # Train/val split by whole documents
│   └── tokenize_corpus.py            # Tokenize splits to flat uint16 binaries
│
├── src/                          # Model implementation
│   ├── config.py                     # GPT model and training configuration
│   ├── dataset.py                    # Memory-mapped dataset and DataLoader
│   ├── attention.py                  # Causal multi-head self-attention
│   └── utils/                        # Shared utilities
│       ├── quality.py                    # Quality checks and domain keywords
│       ├── text.py                       # Text normalization helpers
│       └── retry.py                      # HTTP requests with exponential backoff
│
├── data/
│   ├── raw/                          # Downloaded papers and articles
│   ├── processed/                    # Cleaned plain text
│   ├── filtered/                     # Domain-filtered corpus
│   ├── splits/                       # Train/val document splits
│   └── tokenized/                    # Binary token files (train.bin, val.bin)
│
├── reports/                      # Quality and filtering CSV reports
├── notebooks/                    # Exploration notebooks
└── config.py                     # Legacy project config
```

## Data Pipeline

The pipeline follows five stages, each runnable as a standalone script:

### 1. Collection

Scientific literature is scraped using ~50 targeted queries spanning 10 chemical engineering sub-domains:

| Category | Example Topics |
|---|---|
| Fundamentals | Thermodynamics, transport phenomena, fluid mechanics |
| Reaction Engineering | Reactor design, kinetics, CSTR, plug flow |
| Separation Processes | Distillation, VLE, membrane separation, extraction |
| Process Engineering | Process design, simulation, optimization, intensification |
| Process Control | PID control, MPC, process dynamics |
| Plant Monitoring | Fault detection, anomaly detection, sensor diagnostics |
| Equipment | Heat exchangers, distillation columns, pressure vessels |
| Safety | Hazard analysis, risk assessment, accident prevention |
| Predictive Maintenance | Fault diagnosis, RUL estimation, equipment failure |
| AI in ChemE | ML for processes, digital twins, data-driven optimization |

**Sources:**
- **arXiv** — PDFs (via `arxiv` API) and LaTeX source archives (up to 100 papers per query)
- **Wikipedia** — Articles fetched through the MediaWiki API (up to 50 per query)

### 2. Cleaning

Raw sources go through format-specific cleaning:

- **LaTeX pipeline** — Identifies the main `.tex` document across multi-file archives, resolves `\input`/`\include` references, strips metadata/bibliography/figures/citations while preserving equations, chemical formulas, and scientific prose
- **PDF pipeline** — Uses [Docling](https://github.com/DS4SD/docling) with formula enrichment and table extraction
- **Wikipedia pipeline** — Removes navigation/reference sections, resolves fragmented equation duplicates, converts wiki markup to Markdown headings

### 3. Quality Assessment

Each cleaned document is evaluated against multiple checks:

| Check | Criteria |
|---|---|
| Minimum length | At least 300 words (reject below 100) |
| LaTeX residue | Formatting artifacts below threshold |
| Broken references | Unresolved `\ref`, empty cross-references |
| Prose ratio | Alphabetic character ratio above 30% |
| Garbled text | Control/replacement characters below 2% |
| Repetition | Duplicate lines below 30% |

Documents are classified as **accepted**, **review**, or **rejected**. Results are saved to `reports/arxiv_quality_report.csv`.

### 4. Domain Filtering

Accepted papers are checked against chemical engineering keyword lists (strong and weak matches) and sorted into:

- `arxiv_domain_candidates/` — Strong domain relevance confirmed
- `arxiv_relevance_review/` — Weak or uncertain relevance, needs manual review

### 5. Splitting and Tokenization

- **Splitting** — The corpus is split into train (~95%) and validation (~5%) sets by whole documents to prevent data leakage. Each source (arXiv, Wikipedia) is split independently with its own deterministic seed. A JSON manifest records exact file assignments for reproducibility.
- **Tokenization** — Both splits are tokenized using the GPT-2 tokenizer (`tiktoken`, vocab size 50,257). Documents are separated by end-of-text tokens and written to flat `uint16` binary files for efficient memory-mapped loading during training.

**Corpus statistics:**

| Split | Files | Words | Tokens | Binary Size |
|---|---|---|---|---|
| Train | 4,147 | 12.5M | ~19M | 37 MB |
| Val | 231 | 664K | ~1M | 2.1 MB |

## Model

The model is a decoder-only GPT with configurable architecture:

| Parameter | Default | Small | Medium (GPT-2 small) |
|---|---|---|---|
| Embedding dim | 384 | 128 | 768 |
| Attention heads | 6 | 4 | 12 |
| Layers | 6 | 2 | 12 |
| Context length | 256 | 64 | 1,024 |
| Parameters (est.) | ~30M | ~3M | ~124M |

**Implemented components:**
- **GPTConfig** — Centralized dataclass with validation for all hyperparameters, training settings, and file paths
- **CausalSelfAttention** — Multi-head causal self-attention with separate Q/K/V projections, causal masking, and dropout
- **GPTDataset** — Memory-mapped dataset that reads `uint16` token binaries without loading the full corpus into RAM
- **DataLoader factory** — Configurable loader with batch size, shuffling, and multi-worker support

## Setup

### Prerequisites

- Python 3.11+
- Conda (for environment management)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemLLM.git
cd ChemLLM

# Create and activate conda environment
conda create -n chemllm python=3.11
conda activate chemllm

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core:** PyTorch, NumPy, pandas, tiktoken, tqdm

**Data collection:** arxiv, wikipedia, Wikipedia-API, requests, BeautifulSoup4, PyMuPDF, Docling, datasets

## Usage

```bash
# 1. Scrape arXiv LaTeX sources
python -m data_collection.scrape_arxiv_latex

# 2. Clean LaTeX to plain text
python -m data_collection.clean_arxive_latex

# 3. Scrape Wikipedia articles
python -m data_collection.scrape_wikipedia

# 4. Clean Wikipedia articles
python -m data_collection.clean_wikipedia

# 5. Run quality analysis
python -m preprocessing.analyze

# 6. Filter by domain relevance
python -m preprocessing.filter_corpus

# 7. Split into train/val sets
python -m preprocessing.split_corpus

# 8. Tokenize corpus
python -m preprocessing.tokenize_corpus
```

## Roadmap

- [x] Data collection (arXiv + Wikipedia)
- [x] Text extraction and cleaning pipelines
- [x] Quality assessment framework
- [x] Domain relevance filtering
- [x] Train/validation corpus splitting
- [x] Tokenization with GPT-2 tokenizer (tiktoken)
- [x] Model configuration with presets
- [x] Causal multi-head self-attention
- [x] Memory-mapped dataset and DataLoader
- [ ] Transformer block (attention + FFN + LayerNorm)
- [ ] Full GPT model assembly
- [ ] Training loop with checkpointing
- [ ] Text generation / sampling
- [ ] Evaluation and benchmarking

## License

This project is for research and educational purposes.
