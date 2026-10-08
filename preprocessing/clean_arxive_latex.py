import logging
import re
import tarfile

from pathlib import Path, PurePosixPath


# =====================================================
# 1. Project directories
# =====================================================

RAW_DIR = Path("data/raw/arxiv_latex")
PROCESSED_DIR = Path("data/processed/arxiv_latex")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# 3. Read LaTeX files from a tar.gz archive
# =====================================================

def read_latex_archive(archive_path: Path) -> dict[str, str]:
    """
    Read all .tex files from one compressed arXiv archive.
    Returns
    -------
    dict[str, str]
        Mapping:

            archive path -> LaTeX source
    """

    tex_files = {}

    with tarfile.open(archive_path,mode="r:gz") as archive:
        for member in archive.getmembers():
            # Ignore directories and symbolic links.
            if not member.isfile():
                continue

            member_path = PurePosixPath(member.name)

            # Prevent unsafe paths.
            if member_path.is_absolute() or ".." in member_path.parts:
                continue

            if member_path.suffix.lower() != ".tex":
                continue

            # Skip unusually large source files.
            if member.size > 10_000_000:
                logger.warning("Skipping oversized file: %s", member.name)
                continue

            file_obj = archive.extractfile(member)

            if file_obj is None:
                continue

            content = file_obj.read().decode("utf-8",errors="replace")

            tex_files[str(member_path)] = content

    return tex_files


# =====================================================
# 4. Identify the main LaTeX document
# =====================================================

def find_main_tex(tex_files: dict[str, str]) -> str | None:
    """
    Identify the main LaTeX document.
    Prefer files containing:
        \\documentclass
        \\begin{document}
    Also give a small preference to filenames such as:
        main.tex
        paper.tex
        manuscript.tex
    """

    candidates = []

    for filename, content in tex_files.items():
        score = 0
        if re.search(r"\\documentclass\b",content):
            score += 10
        if re.search(r"\\begin\s*\{document\}",content):
            score += 10
        if PurePosixPath(filename).stem.lower()in {"main","paper","manuscript"}:
            score += 3

        candidates.append((score,len(content),filename))

    if not candidates:
        return None

    candidates.sort(reverse=True)

    # Score 0 means there was no clear document root.
    if candidates[0][0] == 0:
        return None

    return candidates[0][2]


# =====================================================
# 5. Resolve included LaTeX files
# =====================================================

INPUT_PATTERN = re.compile(
    r"\\(?:input|include)\s*"
    r"\{([^{}]+)\}"
)


def combine_latex(filename: str, tex_files: dict[str, str], visited: set[str] | None = None) -> str:
    """
    Replace \\input{} and \\include{} commands with
    the contents of the corresponding .tex files.

    A visited set prevents recursive inclusion.
    """

    if visited is None:
        visited = set()

    if filename in visited:
        return ""

    visited.add(filename)

    content = tex_files[filename]

    current_dir = PurePosixPath(filename).parent

    def replace_input(match):
        included_name = match.group(1).strip()

        if not included_name.endswith(".tex"):
            included_name += ".tex"

        target = current_dir / included_name
    
        # Normalize relative paths without
        # accessing the real filesystem.
        parts = []
        for part in target.parts:
            if part == "..":
                if not parts:
                    return ""
                parts.pop()
            elif part not in {".", ""}:
                parts.append(part)
        target_name = "/".join(parts)

        if target_name not in tex_files:
            logger.warning("Included file not found: %s", target_name)
            return ""

        return combine_latex(target_name, tex_files, visited)

    return INPUT_PATTERN.sub(replace_input, content)


# =====================================================
# 6. General LaTeX helper functions
# =====================================================

def extract_braced_argument(text: str,start: int):
    """
    Read a LaTeX argument beginning with '{'.
    Handles nested braces.
    Example
    -------
    {
        University,
        organization={Department of Engineering}
    }
    Returns
    -------
    tuple[str, int] | None
        argument_content,
        position_after_closing_brace
    """

    if start >= len(text) or text[start] != "{":
        return None

    depth = 0

    for i in range(start, len(text)):
        char = text[i]
        if char == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1

        elif char == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1,

    return None


def remove_braced_command(text: str,command: str) -> str:
    """
    Remove a LaTeX command together with its
    braced argument.

    Supports nested braces and one optional
    [...] argument.

    Example
    -------
    \\author[1]{John Smith}

        -> ""

    \\affiliation{
        organization={University},
        city={Boston}
    }

        -> ""
    """

    pattern = re.compile(
        rf"\\{re.escape(command)}"
        rf"(?:\[[^\]]*\])?"
        rf"\s*\{{"
    )

    position = 0

    while True:
        match = pattern.search(text, position)
        if match is None:
            break
        opening_brace =  match.end() - 1
        result = extract_braced_argument(text, opening_brace)

        if result is None:

            # Malformed braces:
            # do not risk deleting the rest
            # of the document.
            position = match.end()

            continue

        _, end_position = result

        text = text[:match.start()] + " " + text[end_position:]

        position = match.start()

    return text


def unwrap_braced_command(text: str, command: str) -> str:
    """
    Remove a command but preserve the text inside it.

    Example
    -------
    \\textbf{important}

        -> important
    """

    pattern = re.compile(
        rf"\\{re.escape(command)}"
        rf"(?:\[[^\]]*\])?"
        rf"\s*\{{"
    )

    position = 0

    while True:
        match = pattern.search(text, position)
        if match is None:
            break
        opening_brace = match.end() - 1
        result = extract_braced_argument(text, opening_brace)

        if result is None:
            position = match.end()
            continue

        argument, end_position = result

        text = text[:match.start()] + argument + text[end_position:]
    
        position = match.start() + len(argument)

    return text


# =====================================================
# 7. Clean section headings
# =====================================================

def clean_section_headings(text: str) -> str:
    """
    Convert LaTeX section commands into plain headings.

    Example
    -------
    \\section{Heat Transfer}

        ->

    Heat Transfer
    """

    pattern = re.compile(
        r"\\(?:part|chapter|section|subsection|"
        r"subsubsection|paragraph|subparagraph)"
        r"\*?\s*"
        r"(?:\[[^\]]*\]\s*)?"
        r"(?=\{)"
    )

    output = []

    position = 0

    while True:
        match = pattern.search(text, position)
        if match is None:
            output.append(text[position:])
            break

        output.append(text[position: match.start()]
)
        result = extract_braced_argument(text, match.end())

        if result is None:
            output.append(match.group())
            position = match.end()
            continue

        heading, end_position = result

        output.append( "\n\n" + heading.strip() + "\n\n")

        position = end_position

    return "".join(output)


# =====================================================
# 8. Remove document metadata
# =====================================================

def remove_latex_metadata(text: str) -> str:
    """
    Remove author information, affiliations,
    page formatting, and other document metadata.

    Scientific prose and mathematical expressions
    are preserved.
    """

    # -------------------------------------------------
    # Commands whose CONTENT should disappear
    # -------------------------------------------------

    metadata_commands = [
        "author", "address", "affiliation", "affil", "email", "ead",
        "orcid","cortext", "thanks", "thanksref", "dedicatory", "date", 
        "pacs",  "submitto", "shorttitle", "shortauthors", "institution", 
        "streetaddress", "city", "country", "postcode", "state"
    ]

    for command in metadata_commands:
        text = remove_braced_command(text, command)

    # -------------------------------------------------
    # Page/layout commands with braced arguments
    # -------------------------------------------------

    layout_commands = ["pagestyle","thispagestyle","pagenumbering","setcounter","addcontentsline"]

    for command in layout_commands:
        text = remove_braced_command(text, command)

    # -------------------------------------------------
    # Standalone commands
    # -------------------------------------------------

    standalone_commands = [
        "frontmatter", "mainmatter", "backmatter", "sloppy", "flushbottom",
        "frenchspacing","allowdisplaybreaks", "listofalgorithms", "listoffigures",
        "listoftables", "tableofcontents", "maketitle", "newpage", "clearpage",
        "pagebreak", "bigskip", "medskip", "smallskip", "noindent", "indent",
        "centering", "raggedright", "raggedleft",
    ]

    for command in standalone_commands:
        text = re.sub(rf"\\{re.escape(command)}\b", " ", text)

    return text


# =====================================================
# 9. Main scientific text cleaner
# =====================================================

def clean_latex_text(text: str) -> str:
    """
    Clean arXiv LaTeX source for language-model training.
    Preserve
    --------
    - Scientific paragraphs
    - Equations
    - Mathematical notation
    - Chemical formulas
    - Section headings
    - Numerical values
    Remove
    ------
    - Comments
    - Bibliography
    - Figures/tables
    - Author metadata
    - Page layout commands
    - Citation commands
    - Broken standalone references
    - Excessive whitespace
    """

    # =================================================
    # 9.1 Remove comments
    # =================================================

    def remove_comment(line):
        for i, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            # Even number of preceding
            # backslashes means '%' starts
            # a comment.
            if backslashes % 2 == 0:
                return line[:i]
        return line

    text = "\n".join(remove_comment(line) for line in text.splitlines())

    # =================================================
    # 9.2 Keep only document body
    # =================================================

    document_start = re.search(r"\\begin\s*\{document\}", text)
    if document_start:
        text = text[document_start.end():]
    document_end = re.search(r"\\end\s*\{document\}", text)
    if document_end:
        text = text[:document_end.start()]

    # =================================================
    # 9.3 Remove bibliography
    # =================================================

    text = re.split(
        r"\\begin\s*\{thebibliography\}|"
        r"\\bibliography\s*"
        r"(?:\[[^\]]*\]\s*)?\{|"
        r"\\printbibliography\b",
        text,
        maxsplit=1,
    )[0]

    text = re.sub(
        r"\\bibliographystyle"
        r"\s*\{[^{}]*\}",
        "",
        text,
    )

    # =================================================
    # 9.4 Remove unwanted environments
    # =================================================

    unwanted_environments = [
        "figure", "figure*", "table", "table*", "tikzpicture",
        "algorithm","algorithm*", "algorithmic", "lstlisting",
        "verbatim", "comment", "textblock", "textblock*","TAKEOUT",
    ]

    for env in unwanted_environments:
        escaped_env = re.escape(env)
        pattern = (
            r"\\begin\s*\{"
            + escaped_env
            + r"\}"
            r"(?:\[[^\]]*\])?"
            r".*?"
            r"\\end\s*\{"
            + escaped_env
            + r"\}"
        )

        text = re.sub(pattern, "\n\n", text,
            flags=(
                re.DOTALL
                | re.IGNORECASE)
        )

    # =================================================
    # 9.5 Remove document metadata
    # =================================================

    text = remove_latex_metadata(text)

    # =================================================
    # 9.6 Remove citation commands
    # =================================================

    citation_pattern = (
        r"\\(?:"
        r"cite|citep|citet|citealt|"
        r"citeauthor|citeyear|citeyearpar|"
        r"parencite|textcite|nocite"
        r")"
        r"\*?"
        r"(?:\s*\[[^\]]*\])*"
        r"\s*\{[^{}]*\}"
    )

    text = re.sub(citation_pattern, "", text)

    # Citation spacing artifacts.
    text = re.sub(r"~\s*([,.;:])", r"\1", text)

    # =================================================
    # 9.7 Remove figure/table references
    # =================================================

    text = re.sub(
        r"\b(?:Fig(?:ure)?s?\.?|Tables?)"
        r"\s*~?\s*"
        r"\\(?:ref|autoref|cref)"
        r"\s*\{[^{}]*\}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # =================================================
    # 9.8 Remove remaining reference commands
    # =================================================

    text = re.sub(
        r"\\(?:"
        r"ref|eqref|autoref|pageref|"
        r"cref|Cref|eref"
        r")"
        r"\s*\{[^{}]*\}",
        "",
        text,
    )

    text = re.sub(
        r"\\label\s*\{[^{}]*\}",
        "",
        text,
    )

    # =================================================
    # 9.9 Clean section headings
    # =================================================

    text = clean_section_headings(text)

    # =================================================
    # 9.10 Preserve title text
    # =================================================

    text = unwrap_braced_command(text,"title")

    # =================================================
    # 9.11 Preserve abstract/keywords but remove markers
    # =================================================

    text = re.sub(
        r"\\(?:begin|end)\s*"
        r"\{(?:"
        r"frontmatter|abstract|keyword|keywords"
        r")\}",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    # =================================================
    # 9.12 Clean list environments
    # =================================================

    text = re.sub(
        r"\\(?:begin|end)\s*"
        r"\{(?:itemize|enumerate|description)\}",
        "\n",
        text,
    )

    text = re.sub(
        r"\\item(?:\[[^\]]*\])?",
        "\n- ",
        text,
    )

    # =================================================
    # 9.13 Unwrap text-formatting commands
    # =================================================

    text_formatting_commands = [
        "textbf", "textit", "emph", "texttt",
        "textrm", "textsf", "textsc", "underline", "mbox",
    ]

    for command in text_formatting_commands:
        text = unwrap_braced_command(text,command)

    # -------------------------------------------------
    # Math formatting commands
    #
    # Preserve the expression inside them.
    # -------------------------------------------------

    math_formatting_commands = ["mathbf","mathrm","mathit","mathsf","mathcal"]

    for command in math_formatting_commands:
        text = unwrap_braced_command(text,command)

    # =================================================
    # 9.14 Remove selected standalone formatting
    # =================================================

    text = re.sub(
        r"\\(?:"
        r"newline|linebreak|nonumber|notag|"
        r"displaystyle"
        r")\b",
        " ",
        text,
    )

    # =================================================
    # 9.15 Normalize escaped characters
    # =================================================

    replacements = {
        r"\%": "%",
        r"\&": "&",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }

    for old, new in replacements.items():
        text = text.replace(old,new)

    # Convert LaTeX non-breaking spaces
    # outside meaningful commands into spaces.
    text = text.replace("~"," ")

    # =================================================
    # 9.16 Remove trailing References heading
    # =================================================

    text = re.sub(
        r"\n\s*References\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # =================================================
    # 9.17 Normalize whitespace
    # =================================================

    text = re.sub(r"[ \t]+", " ",text)
    text = re.sub(r" *\n *","\n",text)
    text = re.sub(r"\n{3,}","\n\n",text,)

    return text.strip()


# =====================================================
# 10. Process one archive
# =====================================================

def process_archive(archive_path: Path) -> bool:
    tex_files = read_latex_archive(archive_path)
    if not tex_files:
        logger.warning("No LaTeX files found: %s", archive_path.name)
        return False

    main_file = find_main_tex(tex_files)

    if main_file is None:
        logger.warning("Main document not found: %s",archive_path.name)
        return False

    combined_text = combine_latex(main_file, tex_files)
    cleaned_text = clean_latex_text(combined_text)

    if len(cleaned_text) < 500:
        logger.warning("Cleaned text too short: %s",archive_path.name)
        return False

    paper_id = archive_path.name.removesuffix(".tar.gz")

    output_path = PROCESSED_DIR / f"{paper_id}.txt"

    output_path.write_text(cleaned_text, encoding="utf-8")

    logger.info("Processed %s -> %s", archive_path.name, output_path.name)

    return True


# =====================================================
# 11. Process all archives
# =====================================================

def main():
    archives = sorted(RAW_DIR.glob("*.tar.gz"))

    logger.info("Found %d archives", len(archives))

    processed = 0
    skipped = 0
    failed = 0

    for archive_path in archives:
        output_path = (PROCESSED_DIR / (archive_path.name.removesuffix(".tar.gz") + ".txt"))

        # Resume safely after interruption.
        if output_path.exists():
            skipped += 1
            continue
        try:
            success = process_archive(archive_path)
            if success:
                processed += 1
            else:
                failed += 1
        except (tarfile.TarError, OSError, UnicodeError, ValueError):
            logger.exception("Failed to process %s", archive_path.name)

            failed += 1

    logger.info("Finished | Processed: %d | Skipped: %d | Failed: %d", processed, skipped, failed)


if __name__ == "__main__":
    main()