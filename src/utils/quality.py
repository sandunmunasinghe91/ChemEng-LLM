
"""
Quality validation for cleaned scientific papers.

This module evaluates the quality of one document.

Possible statuses:
    accepted : No issues detected by the configured checks.
    review   : Potentially useful document requiring inspection.
    rejected : Empty, extremely short, or severely corrupted document.

This module does not read, move, or delete files.
"""

import logging
import re


logger = logging.getLogger(__name__)


# =====================================================
# 1. Configuration
# =====================================================

MIN_REJECT_WORDS = 100
MIN_ACCEPT_WORDS = 300

MAX_GARBLED_RATIO = 0.02
MIN_ALPHA_RATIO = 0.30
MAX_REPETITION_RATIO = 0.30

MIN_LINES_FOR_REPETITION = 10


# =====================================================
# 2. LaTeX artifact patterns
# =====================================================

# Detect unwanted formatting commands.
# Do not count mathematical commands such as
# \frac, \theta, \alpha, \sum, or \int.

FORMATTING_PATTERN = re.compile(
    r"\\(?:"
    r"author|address|ead|thanks|dedicatory|"
    r"pagestyle|pagenumbering|fancyhead|"
    r"fancyfoot|fancyhf|setcounter|"
    r"renewcommand|glsadd|vskip|"
    r"newline|maketitle"
    r")\b"
)


# Detect unwanted LaTeX environments.

UNWANTED_ENVIRONMENT_PATTERN = re.compile(
    r"\\(?:begin|end)\s*"
    r"\{(?:"
    r"frontmatter|thebibliography|TAKEOUT|"
    r"figure\*?|table\*?|tikzpicture"
    r")\}",
    flags=re.IGNORECASE,
)


# Detect unresolved LaTeX reference commands.

UNRESOLVED_REFERENCE_PATTERN = re.compile(
    r"\\(?:"
    r"ref|eqref|eref|autoref|pageref|cref|Cref"
    r")\s*\{[^{}]*\}"
)


# Detect obvious broken references.

BROKEN_REFERENCE_PATTERN = re.compile(
    r"\b(?:"
    r"Eq(?:uation)?s?\.?|"
    r"Sec(?:tion)?s?\.?|"
    r"Fig(?:ure)?s?\.?|"
    r"Refs?\.?|"
    r"Tables?"
    r")"
    r"\s*(?:\(\s*\)|(?=[,;]|\b(?:we|has|is|are)\b))",
    flags=re.IGNORECASE,
)


# Detect garbled or corrupted characters.

GARBLED_PATTERN = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|"
    r"\\x[0-9a-fA-F]{2}|"
    r"\uFFFD"
)


# =====================================================
# 3. Chemical engineering keywords
# =====================================================

STRONG_DOMAIN_KEYWORDS = [
    "chemical engineering", "process engineering", "chemical reactor", "reaction engineering",
    "chemical reaction kinetics", "process control", "mass transfer", "heat transfer",
    "heat exchanger", "distillation", "vapor liquid equilibrium", "vapour liquid equilibrium",
    "catalytic reactor", "process simulation", "process optimization", "transport phenomena",
    "fluidized bed", "membrane separation", "separation process", "film boiling",
    "phase equilibrium", "process intensification", "process safety", "chemical process",
    "reactor design", "reactor modeling", "reaction rate",
]


WEAK_DOMAIN_KEYWORDS = [
    "reaction", "reactor", "temperature", "pressure", "concentration", "equilibrium",
    "thermodynamics", "kinetics", "fluid", "diffusion", "simulation", "optimization",
    "control", "prediction", "model", "catalyst", "separation",
]


# =====================================================
# 4. Individual quality checks
# =====================================================

def check_minimum_length(text: str,min_words: int = MIN_ACCEPT_WORDS) -> tuple[bool, str]:
    """
    Check whether the document contains enough words.
    """
    word_count = len(text.split())

    if word_count < min_words:
        return False, f"Short document: {word_count} words, minimum {min_words})"
    return True, f"Length OK: {word_count} words"


def check_latex_residue(text: str,max_artifacts_per_1000: float = 2.0) -> tuple[bool, str]:
    """
    Detect unwanted LaTeX formatting commands.
    Mathematical notation is intentionally preserved.
    """

    formatting_matches = FORMATTING_PATTERN.findall(text)

    environment_matches = UNWANTED_ENVIRONMENT_PATTERN.findall(text)

    artifact_count = len(formatting_matches) + len(environment_matches)
    
    word_count = max(len(text.split()), 1)

    artifact_rate = (artifact_count / word_count) * 1000

    # Flag unwanted environments even if their
    # overall frequency is low.

    serious_environment_found = bool(environment_matches)

    passed = artifact_rate <= max_artifacts_per_1000 and not serious_environment_found

    message = (
        f"{artifact_count} formatting artifacts; "
        f"{artifact_rate:.2f} per 1000 words"
    )

    if serious_environment_found:
        message += "; unwanted environment detected"

    return passed, message


def check_broken_references(text: str, max_per_1000_words: float = 2.0) -> tuple[bool, str]:

    broken = BROKEN_REFERENCE_PATTERN.findall(text)
    unresolved = UNRESOLVED_REFERENCE_PATTERN.findall(text)

    total = len(broken) + len(unresolved)
    word_count = max(len(text.split()), 1)

    rate = (total / word_count) * 1000

    if rate > max_per_1000_words:
        return False, f"{total} reference problems, ({rate:.2f} per 1000 words)"

    return True, f"Reference problems OK: {total} ({rate:.2f} per 1000 words)"


    
def check_prose_ratio(text: str, min_ratio: float = MIN_ALPHA_RATIO) -> tuple[bool, str]:
    """
    Estimate the proportion of alphabetic characters.

    This is an approximate text-composition check,
    not a true measure of prose quality.
    """

    if not text:
        return False, "Empty text"

    alpha_chars = sum(char.isalpha() for char in text)
    ratio = alpha_chars / len(text)

    if ratio < min_ratio:
        return False, f"Low alphabetic-character ratio: {ratio:.1%}"
        
    return True, f"Alphabetic-character ratio OK: {ratio:.1%}"



def check_domain_relevance(text: str) -> tuple[bool, str]:
    """
    Estimate chemical engineering relevance.

    A failed result requires review rather than
    automatic rejection.
    """
    text_lower = text.lower()

    strong_hits = [ keyword for keyword in STRONG_DOMAIN_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b",text_lower)]

    weak_hits = [keyword for keyword in WEAK_DOMAIN_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b",text_lower)]

    if strong_hits:
        return (True,"Potential domain match: "+ ", ".join(strong_hits[:5]))

    return False, "Domain relevance uncertain; weak terms: " + (", ".join(weak_hits[:5]) or "none")



def check_garbled_text(text: str, max_ratio: float = MAX_GARBLED_RATIO) -> tuple[bool, str]:
    """
    Detect control characters, replacement characters,
    and escaped hexadecimal artifacts.
    """

    if not text:
        return False, "Empty text"

    matches = GARBLED_PATTERN.findall(text)

    ratio = len(matches) / len(text)

    if ratio > max_ratio:
        return False, f"Excessive garbled text: {len(matches)} artifacts ({ratio:.2%})"

    return True, f"Garbled text check OK: {len(matches)} artifacts ({ratio:.2%})"


def check_repetition(text: str,max_ratio: float = MAX_REPETITION_RATIO) -> tuple[bool, str]:
    """
    Detect excessive repetition in substantial text lines.

    Ignore short lines and lines beginning with
    LaTeX commands to reduce false positives.
    """

    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 80 and not line.strip().startswith("\\")]

    if len(lines) < MIN_LINES_FOR_REPETITION:
        return True, "Insufficient substantial lines for repetition check"

    unique_lines = set(lines)

    repeated_ratio = 1 - len(unique_lines) / len(lines)

    if repeated_ratio > max_ratio:
        return False, f"Excessive repeated lines: {repeated_ratio:.1%}"

    return True, f"Repetition check OK: {repeated_ratio:.1%}"


# =====================================================
# 5. Evaluate one paper
# =====================================================

def check_paper_quality(text: str, paper_id: str = "", verbose: bool = False) -> tuple[str, dict]:
    """
    Run all quality checks on one cleaned paper.

    Returns
    -------
    status:
        "accepted", "review", or "rejected"

    report:
        Dictionary containing individual check results,
        explanations, word count, paper ID, and status.
    """

    checks = {
        "minimum_length": check_minimum_length(text),
        "latex_residue": check_latex_residue(text),
        "broken_references": check_broken_references(text),
        "prose_ratio": check_prose_ratio(text),
        "domain_relevance": check_domain_relevance(text),
        "garbled_text": check_garbled_text(text),
        "repetition": check_repetition(text),
    }

    word_count = len(text.split())

    # ---------------------------------------------
    # Determine the document status
    # ---------------------------------------------

    quality_checks = [
        checks["minimum_length"][0],
        checks["latex_residue"][0],
        checks["broken_references"][0],
        checks["prose_ratio"][0],
        checks["garbled_text"][0],
        checks["repetition"][0],
    ]

    if not text.strip() or word_count < MIN_REJECT_WORDS or not checks["garbled_text"][0]:
        status = "rejected"
        
    elif not all(quality_checks):
        status = "review"

    else:
        status = "accepted"

    # ---------------------------------------------
    # Build the report
    # ---------------------------------------------

    report = {
        name: {
            "passed": passed,
            "reason": reason,
        }
        for name, (passed, reason) in checks.items()
    }

    report["paper_id"] = paper_id
    report["status"] = status
    report["word_count"] = word_count

    # ---------------------------------------------
    # Optional detailed logging
    # ---------------------------------------------

    if verbose:
        logger.info("%s | %s", status.upper(), paper_id)
        for name, (passed, reason) in checks.items():
            label = "PASS" if passed else "CHECK"
            logger.info("  %s | %s | %s", label, name, reason)

    return status, report