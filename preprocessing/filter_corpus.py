import csv
import logging
import shutil
from pathlib import Path
from src.utils.quality import check_domain_relevance


# =====================================================
# 1. Project directories
# =====================================================

PROCESSED_DIR = Path("data/processed/arxiv_latex")
QUALITY_REPORT = Path("reports/arxiv_quality_report.csv")
OUTPUT_DIR = Path("data/filtered")
DOMAIN_DIR = OUTPUT_DIR / "arxiv_domain_candidates"
REVIEW_DIR = OUTPUT_DIR / "arxiv_relevance_review"
FILTER_REPORT = Path("reports/arxiv_filter_report.csv")

# =====================================================
# 2. Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# 3. Filter one paper
# =====================================================

def filter_paper(file_path: Path) -> tuple[str, str]:
    """
    Check whether an accepted-quality paper
    contains strong chemical engineering keywords.

    Returns
    -------
    category:
        "domain_candidate" or "relevance_review"

    reason:
        Explanation of the domain check.
    """

    text = file_path.read_text(encoding="utf-8",errors="replace")
    relevant, reason = check_domain_relevance(text)

    if relevant:
        return "domain_candidate", reason

    return "relevance_review", reason


# =====================================================
# 4. Filter the accepted corpus
# =====================================================

def filter_corpus() -> dict:
    """
    Read the quality report and select papers
    whose quality status is accepted.

    Separate those papers based on domain keywords.
    """

    if not PROCESSED_DIR.is_dir():
        raise NotADirectoryError(f"Processed directory not found: {PROCESSED_DIR}")

    if not QUALITY_REPORT.is_file():
        raise FileNotFoundError(f"Quality report not found: {QUALITY_REPORT}")

    DOMAIN_DIR.mkdir(parents=True, exist_ok=True,)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True,)

    results = {
        "quality_accepted": 0,
        "domain_candidates": 0,
        "relevance_review": 0,
        "missing_files": 0,
    }

    report_rows = []

    # -------------------------------------------------
    # Read the existing quality report
    # -------------------------------------------------

    with QUALITY_REPORT.open("r",encoding="utf-8",newline="") as file:
        reader = csv.DictReader(file)
        accepted_rows = [row for row in reader if row["status"] == "accepted"]

    results["quality_accepted"] = len(accepted_rows)

    logger.info("Found %d quality-accepted papers", len(accepted_rows))

    # -------------------------------------------------
    # Check domain relevance
    # -------------------------------------------------

    for row in accepted_rows:
        filename = row["filename"]
        file_path = PROCESSED_DIR / filename
        if not file_path.is_file():
            logger.warning("Missing document: %s", filename)
            results["missing_files"] += 1
            continue
        category, reason = filter_paper(file_path)
        if category == "domain_candidate":
            destination = DOMAIN_DIR / filename
            results["domain_candidates"] += 1
        else:
            destination = REVIEW_DIR / filename
            results["relevance_review"] += 1

        # Copy the document without modifying the original.
        shutil.copy2(file_path,destination)

        report_rows.append(
            {
                "filename": filename,
                "quality_status": "accepted",
                "category": category,
                "reason": reason,
            }
        )

    # -------------------------------------------------
    # Save the filtering report
    # -------------------------------------------------

    FILTER_REPORT.parent.mkdir(parents=True,exist_ok=True)

    with FILTER_REPORT.open("w",encoding="utf-8",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["filename","quality_status","category","reason"])
        writer.writeheader()
        writer.writerows(report_rows)

    # -------------------------------------------------
    # Display the results
    # -------------------------------------------------

    logger.info("Domain candidates: %d", results["domain_candidates"])
    logger.info("Relevance review: %d", results["relevance_review"])
    logger.info("Missing files: %d",results["missing_files"])
    logger.info("Filtering report saved to %s", FILTER_REPORT)
    return results

# =====================================================
# 5. Main execution function
# =====================================================

def main():
    results = filter_corpus()
    print("\n" + "=" * 50)
    print("CORPUS FILTERING SUMMARY")
    print("=" * 50)
    print(f"Quality accepted: {results['quality_accepted']}")
    print(f"Domain candidates: {results['domain_candidates']}")
    print(f"Relevance review: {results['relevance_review']}")
    print(f"Missing files: {results['missing_files']}")
    print(f"\nReport: {FILTER_REPORT}")


if __name__ == "__main__":
    main()