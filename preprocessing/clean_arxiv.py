#%%
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)


converter = DocumentConverter()


pipeline_options = PdfPipelineOptions()

# Disable OCR for PDFs containing selectable text.
pipeline_options.do_ocr = False

# Enable mathematical formula recognition.
pipeline_options.do_formula_enrichment = True

# Enable table structure extraction.
pipeline_options.do_table_structure = True


# Initialize the converter only once.
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract structured text from a PDF using Docling.

    Returns:
        Extracted text in Markdown format.
        Returns an empty string if extraction fails.
    """

    pdf_path = Path(pdf_path)

    try:
        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        if not text.strip():
            logger.warning( "No text extracted from %s",pdf_path.name)

        return text

    except Exception:
        logger.exception("Failed to extract PDF: %s",pdf_path)

        return ""

# %%
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )


    # Select one PDF for testing
    pdf_path =  '../raw/arxiv_papers/0707.1265v1.pdf'

    # Extract text
    text = extract_text_from_pdf(pdf_path)

    # Display the result
    print(f"Characters extracted: {len(text)}")
    print("\nExtracted text:\n")
    print(text)



# %%
