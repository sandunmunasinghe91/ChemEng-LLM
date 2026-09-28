
"""
Analyze the quality of the cleaned arXiv corpus.

Run from the project root:

    python -m preprocessing.analyze
"""

import csv
import logging

from collections import Counter
from pathlib import Path

from src.utils.quality import check_paper_quality


logger = logging.getLogger(__name__)


# =====================================================
# 1. Project paths
# =====================================================


PROCESSED_DIR = Path("data/processed/arxiv_latex")

REPORT_PATH = Path("reports/arxiv_quality_report.csv")


# =====================================================
# 2. Analyze the complete corpus
# =====================================================

def analyze_corpus_quality(
    processed_dir: Path,
    report_path: Path | None = None,
    recursive: bool = False,
) -> dict:
    """
    Run quality checks on all cleaned .txt papers.

    Parameters
    ----------
    processed_dir:
        Directory containing cleaned text files.

    report_path:
        Optional path for the CSV report.

    recursive:
        Search subdirectories when True.

    Returns
    -------
    dict:
        Corpus statistics and quality-check results.

    This function does not modify or delete
    the original documents.
    """

    processed_dir = Path(processed_dir)

    if not processed_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {processed_dir}"
        )

    if recursive:
        files = sorted(processed_dir.rglob("*.txt"))
    else:
        files = sorted(processed_dir.glob("*.txt"))

    logger.info(
        "Analyzing %d papers...",
        len(files),
    )

    results = {
        "total": len(files),
        "accepted": 0,
        "review": 0,
        "rejected": 0,
        "read_errors": 0,
        "fail_reasons": {},
        "avg_word_count": 0,
        "total_words": 0,
    }

    fail_reasons = Counter()

    csv_rows = []

    # ---------------------------------------------
    # Process each paper
    # ---------------------------------------------

    for file_path in files:

        try:

            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError:

            logger.exception(
                "Cannot read %s",
                file_path.name,
            )

            results["read_errors"] += 1

            continue

        status, report = check_paper_quality(
            text=text,
            paper_id=file_path.stem,
        )

        results[status] += 1

        results["total_words"] += report["word_count"]

        # Find checks requiring attention.

        failed_checks = [
            name
            for name, data in report.items()
            if isinstance(data, dict)
            and not data["passed"]
        ]

        for name in failed_checks:
            fail_reasons[name] += 1

        # Save one row for the CSV report.

        csv_rows.append(
            {
                "filename": file_path.name,
                "status": status,
                "word_count": report["word_count"],
                "failed_checks": "; ".join(failed_checks),
                "reasons": "; ".join(
                    report[name]["reason"]
                    for name in failed_checks
                ),
            }
        )

        logger.info(
            "%s | %s",
            status.upper(),
            file_path.name,
        )

    # ---------------------------------------------
    # Calculate corpus statistics
    # ---------------------------------------------

    analyzed_count = (
        results["accepted"]
        + results["review"]
        + results["rejected"]
    )

    if analyzed_count:
        results["avg_word_count"] = (
            results["total_words"] // analyzed_count
        )

    results["fail_reasons"] = dict(fail_reasons)

    # ---------------------------------------------
    # Save the CSV report
    # ---------------------------------------------

    if report_path is not None:

        report_path = Path(report_path)

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "filename",
                    "status",
                    "word_count",
                    "failed_checks",
                    "reasons",
                ],
            )

            writer.writeheader()

            writer.writerows(csv_rows)

        logger.info(
            "Report saved to %s",
            report_path,
        )

    # ---------------------------------------------
    # Final summary
    # ---------------------------------------------

    logger.info(
        "Accepted: %d",
        results["accepted"],
    )

    logger.info(
        "Review: %d",
        results["review"],
    )

    logger.info(
        "Rejected: %d",
        results["rejected"],
    )

    logger.info(
        "Read errors: %d",
        results["read_errors"],
    )

    return results


# =====================================================
# 3. Main execution function
# =====================================================

def main():

    results = analyze_corpus_quality(
        processed_dir=PROCESSED_DIR,
        report_path=REPORT_PATH,
    )

    print("\n" + "=" * 50)
    print("CORPUS QUALITY SUMMARY")
    print("=" * 50)

    print(f"Total files:   {results['total']}")
    print(f"Accepted:      {results['accepted']}")
    print(f"Review:        {results['review']}")
    print(f"Rejected:      {results['rejected']}")
    print(f"Read errors:   {results['read_errors']}")

    print(
        f"Average words: {results['avg_word_count']}"
    )

    print("\nIssues detected:")

    for issue, count in results["fail_reasons"].items():

        print(f"  {issue}: {count}")

    print(f"\nReport saved to: {REPORT_PATH}")


# =====================================================
# 4. Script entry point
# =====================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    main()