import re
from pathlib import Path


def clean_text(text: str) -> str:

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "",text)
    # Normalize spaces and tabs
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

def save_text(text: str, output_path: str | Path) -> None:
    # Convert the path to a Path object
    output_path = Path(output_path)
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the cleaned text
    output_path.write_text(text, encoding="utf-8")