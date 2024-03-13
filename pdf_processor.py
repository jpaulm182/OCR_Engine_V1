import fitz
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text and metadata from a PDF file using PyMuPDF.

    :param pdf_path: Path to the PDF file
    :return: Extracted text and metadata
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist")

    text_content = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_content.append(page.get_text("text"))
    except Exception as e:
        raise Exception(f"Failed to extract text from {pdf_path}: {str(e)}")

    return text_content

# This function can be expanded to extract more metadata or handle different content types within the PDF.
