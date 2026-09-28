import arxiv
import time
import logging
import tarfile
from pathlib import Path
from src.utils.retry import make_request


PDF_DIR = Path("data/raw/arxiv_papers")
LATEX_DIR = Path("data/raw/arxiv_latex")
LATEX_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

client = arxiv.Client()
logger.info("arXiv client initialized")

queries = [

    # 1. Chemical engineering fundamentals
    "chemical engineering thermodynamics",
    "transport phenomena chemical engineering",
    "fluid mechanics chemical engineering",
    "heat transfer chemical engineering",
    "mass transfer chemical engineering",

    # 2. Reaction engineering
    "chemical reaction engineering",
    "chemical reaction kinetics",
    "chemical reactor modeling",
    "chemical reactor design",
    "CSTR reactor",
    "plug flow reactor",
    "catalytic reactor",

    # 3. Separation processes
    "chemical separation processes",
    "distillation process",
    "vapor liquid equilibrium",
    "absorption process chemical engineering",
    "adsorption process chemical engineering",
    "membrane separation",
    "liquid liquid extraction",

    # 4. Process engineering
    "process systems engineering",
    "chemical process design",
    "chemical process simulation",
    "chemical process optimization",
    "process integration chemical engineering",
    "process intensification",

    # 5. Process dynamics and control
    "chemical process control",
    "process dynamics chemical engineering",
    "model predictive control chemical process",
    "PID control chemical process",
    "advanced process control",
    "process control optimization",

    # 6. Plant monitoring
    "chemical process monitoring",
    "industrial process monitoring",
    "process fault detection",
    "process fault diagnosis",
    "chemical process anomaly detection",
    "sensor fault detection industrial process",

    # 7. Equipment
    "heat exchanger chemical engineering",
    "distillation column modeling",
    "chemical reactor operation",
    "industrial pumps process engineering",
    "compressor process engineering",
    "industrial boiler process",
    "pressure vessel chemical engineering",

    # 8. Safety
    "chemical process safety",
    "process hazard analysis",
    "chemical plant risk assessment",
    "industrial process safety",
    "chemical process accident prevention",

    # 9. Troubleshooting / predictive maintenance
    "chemical process troubleshooting",
    "industrial process fault diagnosis",
    "predictive maintenance chemical plant",
    "predictive maintenance process industry",
    "equipment failure detection industrial process",
    "remaining useful life industrial equipment",

    # 10. AI / data-driven process engineering
    "machine learning chemical engineering",
    "machine learning chemical process",
    "deep learning process engineering",
    "artificial intelligence chemical process",
    "digital twin chemical process",
    "digital twin chemical plant",
    "industrial process optimization machine learning",
]


def download_latex(paper):
    paper_id = paper.get_short_id()
    safe_id = paper_id.replace("/", "_")

    # Possible output paths
    tar_gz_path = LATEX_DIR / f"{safe_id}.tar.gz"
    tar_path = LATEX_DIR / f"{safe_id}.tar"
    tex_path = LATEX_DIR / f"{safe_id}.tex"

    # Temporary download path
    temp_path = LATEX_DIR / f"{safe_id}.part"

    # Skip previously downloaded papers
    if any(path.exists() for path in [tar_gz_path,tar_path,tex_path,]):
        logger.info(f"Already downloaded: {safe_id}")
        return "skipped"

    # Construct source URL
    source_url = (f"https://arxiv.org/src/{paper_id}")
    try:

        logger.info(f"Downloading LaTeX source: {safe_id}")
        response = make_request( url=source_url,return_json=False)
        temp_path.write_bytes(response.content)

        # Read file header
        with open(temp_path, "rb") as f:
            header = f.read(4096)

        # Case 1: PDF returned
        if header.startswith(b"%PDF-"):
            logger.warning(f"PDF returned instead of LaTeX: {safe_id}")
            return "no_source"

        # Case 2: TAR archive
        if tarfile.is_tarfile(temp_path):
            with tarfile.open(temp_path, "r:*" ) as tar:
                tex_files = [member for member in tar.getmembers()if member.isfile() 
                             and member.name.lower().endswith((".tex", ".ltx"))]

            if not tex_files:
                logger.warning( f"No LaTeX files in archive: {safe_id}")
                return "no_source"

            # Identify compressed vs
            # uncompressed TAR archive
            if header.startswith(b"\x1f\x8b"):
                final_path = tar_gz_path
            else:
                final_path = tar_path

            # Save final archive
            temp_path.replace(final_path)
            logger.info(f"Downloaded: {safe_id} | "f"LaTeX files: {len(tex_files)}")
            return "downloaded"

        # Case 3: Single LaTeX file
   
        latex_markers = [

            b"\\documentclass",
            b"\\documentstyle",
            b"\\begin{document}",
            b"\\input",
            b"\\def",

        ]
        if any(marker in header for marker in latex_markers):
            temp_path.replace(tex_path)
            logger.info(f"Downloaded single LaTeX file: {safe_id}")
            return "downloaded"

        # Case 4: Unsupported format
        logger.warning(f"Unsupported source format: {safe_id}")
        return "unsupported"

    except Exception as e:

        if (hasattr(e, "response") and e.response is not None and e.response.status_code == 404):
            logger.warning(f"Source not found: {safe_id}")
            return "no_source"
        logger.error(f"Failed to download {safe_id}: {e}")
        return "failed"

    finally:
        # Remove incomplete temporary files
        temp_path.unlink(missing_ok=True)

# 7. Initialize counters

downloaded = skipped = failed = no_source = unsupported = processed = 0
seen_ids = set()

# 8. Search and download papers
for query in queries:
    logger.info(f"Searching arXiv: {query}")

    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        for paper in client.results(search):
            paper_id = paper.get_short_id()
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            processed += 1

            # Download source
            status = download_latex(paper)
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1
            elif status == "no_source":
                no_source += 1
            elif status == "unsupported":
                unsupported += 1

            logger.info(
                f"Processed: {processed} | "
                f"Downloaded: {downloaded} | "
                f"Skipped: {skipped} | "
                f"Failed: {failed}"
            )

            # Polite delay between downloads
            time.sleep(3)

    except arxiv.ArxivError as e:
        logger.error(
            f"Search failed for '{query}': {e}"
        )


# 9. Final summary

logger.info("=" * 50)
logger.info(
    f"Unique papers processed: {processed}"
)
logger.info(
    f"Downloaded: {downloaded}"
)
logger.info(
    f"Skipped: {skipped}"
)
logger.info(
    f"Failed: {failed}"
)
logger.info(
    f"No LaTeX source: {no_source}"
)
logger.info(
    f"Unsupported format: {unsupported}"
)
logger.info("=" * 50)
            