# ChemLLM

A domain-specific GPT-style language model for chemical engineering, built from the ground up — from data collection to model training.

The goal is to pre-train a transformer on a curated corpus of chemical engineering literature so it can understand and generate text about reaction engineering, process control, thermodynamics, separation processes, and related topics.

## Project Status

Currently in the **data collection and preprocessing** phase. The training corpus is being assembled from arXiv papers and Wikipedia articles, with automated pipelines for scraping, cleaning, quality assessment, and domain filtering.

## Architecture

```
ChemLLM/
├── data_collection/          # Scraping and raw text extraction
│   ├── scrape_arxiv.py           # Download arXiv PDFs by domain queries
│   ├── scrape_arxiv_latex.py     # Download arXiv LaTeX source archives
│   ├── scrape_wikipedia.py       # Fetch Wikipedia articles via MediaWiki API
│   ├── clean_arxiv.py            # Extract text from PDFs using Docling
│   ├── clean_arxive_latex.py     # Parse, combine, and clean LaTeX sources
│   └── clean_wikipedia.py        # Clean Wikipedia markup and equations
│
├── preprocessing/            # Corpus quality and filtering
│   ├── analyze.py                # Quality analysis with CSV reporting
│   └── filter_corpus.py          # Domain relevance filtering
│
├── src/utils/                # Shared utilities
│   ├── quality.py                # Quality checks and domain keyword matching
│   ├── text.py                   # Text normalization helpers
│   └── retry.py                  # HTTP requests with exponential backoff
│
├── data/
│   ├── raw/                      # Downloaded papers and articles
│   │   ├── arxiv_papers/             # arXiv PDFs
│   │   ├── arxiv_latex/              # arXiv LaTeX archives (.tar.gz)
│   │   └── wikipedia/                # Wikipedia articles (JSON)
│   ├── processed/                # Cleaned plain text
│   │   ├── arxiv_latex/
│   │   └── wikipedia/
│   └── filtered/                 # Domain-filtered corpus
│       ├── arxiv_domain_candidates/
│       └── arxiv_relevance_review/
│
├── Tokenized/                # Tokenized training data (WIP)
├── reports/                  # Quality and filtering CSV reports
├── notebooks/                # Exploration notebooks
└── config.py                 # Project configuration
```

## Data Pipeline

The pipeline follows four stages:

### 1. Collection

Scrape scientific literature using ~50 targeted queries spanning 10 chemical engineering sub-domains:

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
```

## Roadmap

- [x] Data collection (arXiv + Wikipedia)
- [x] Text extraction and cleaning pipelines
- [x] Quality assessment framework
- [x] Domain relevance filtering
- [ ] Tokenizer training (tiktoken / custom BPE)
- [ ] GPT architecture implementation
- [ ] Pre-training on chemical engineering corpus
- [ ] Evaluation and benchmarking

## License

This project is for research and educational purposes.
