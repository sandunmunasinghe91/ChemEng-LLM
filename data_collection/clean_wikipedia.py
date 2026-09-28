import json
import re
import logging
from pathlib import Path
from src.utils.text import clean_text,save_text

RAW_DIR = Path("raw/wikipedia")
PROCESSED_DIR = Path("processed/wikipedia")
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

#=====================================================
# Cleaning Wikipedia Equations
#=====================================================

def clean_wikipedia_equations(text: str) -> str:
    """
    Remove fragmented Wikipedia math duplicates
    when followed by a complete LaTeX expression.

    Preserve the LaTeX equation.
    """

    # Match a line containing a single mathematical symbol
    math_symbol = r"[A-Za-z0-9ΔδμλπθΩαβγ−+\-*/=≤≥()]+"

    # Match a sequence of fragmented mathematical lines
    fragmented_pattern = (
        rf"(?m)(?:^[ \t]*{math_symbol}[ \t]*\n){{3,}}"
    )

    # Match the LaTeX representation following the fragment
    latex_pattern = r"[ \t]*\{\\displaystyle\b"

    # Find fragmented equations followed by LaTeX
    pattern = fragmented_pattern + latex_pattern

    def remove_fragment(match):
        # Preserve the LaTeX expression
        return r"{\displaystyle"

    text = re.sub(pattern, remove_fragment, text)
    return text

#=====================================================
# Cleaning Wikipedia scientific text
#=====================================================

def clean_wikipedia_scientific_text(text: str) -> str:
    """
    Clean extracted Wikipedia scientific text.

    - Preserve mathematical and chemical expressions.
    - Remove trailing reference/navigation sections.
    - Convert Wikipedia headings to Markdown headings.
    """

    # Remove unwanted trailing sections
    unwanted_sections = {
        "references",
        "notes",
        "bibliography",
        "further reading",
        "external links",
        "see also",
        "sources",
        "cited sources"
    }

    #Clean fragmented equations
    text = clean_wikipedia_equations(text)

    #Identify top-level Wikipedia section headings
    heading_pattern = re.compile(
        r"(?m)^==[ \t]*([^=\n]+?)[ \t]*==[ \t]*$"
    )

    matches = list(heading_pattern.finditer(text))

    # Work backward through trailing sections
    for match in reversed(matches):
        heading = match.group(1).strip().lower()
        if heading in unwanted_sections:
            text = text[:match.start()]
        else:
            break

    # Convert Wikipedia headings to Markdown

    def normalize_heading(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        return "#" * level + " " + title
    text = re.sub(
        r"(?m)^(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$",
        normalize_heading,
        text
    )
    return text



#=====================================================
# Clean Wikipedia Article
#=====================================================

def clean_wikipedia_article(file_path: Path) -> str:

    # Read Wikipedia JSON
    with open(file_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    # Extract article title and text
    title = article.get("title", "").strip()
    text = article.get("text", "")

    # Apply Wikipedia-specific cleaning
    text = clean_wikipedia_scientific_text(text)

    # Apply common cleaning
    cleaned_text = clean_text(text)

    # Add article title
    if title and cleaned_text:
        cleaned_text = f"# {title}\n\n{cleaned_text}"

    return cleaned_text

#=====================================================
# Execution
#=====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    processed = skipped = failed = 0

    for file_path in RAW_DIR.glob("*.json"):
        try:
            # Extract and clean article text
            cleaned_text = clean_wikipedia_article(file_path)

            # Skip empty articles
            if not cleaned_text:
                skipped += 1
                continue

            # Create output path
            output_path = (PROCESSED_DIR / f"{file_path.stem}.txt")

            # Save cleaned text
            save_text(cleaned_text, output_path)

            processed += 1

            logger.info(f"Processed: {file_path.name}")

        except Exception as e:
            failed += 1
            logger.error(f"Failed: {file_path.name}: {e}")

    logger.info(f"Finished | Processed: {processed} | "
        f"Skipped: {skipped} | Failed: {failed}"
    )


if __name__ == "__main__":
    main()