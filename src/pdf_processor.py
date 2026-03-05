"""
PDF Processor — Converts PDF pages to PIL Images using PyMuPDF (fitz).
"""

from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """Convert each page of a PDF to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. 200 recommended by dots.ocr docs.
             Higher DPI = more detail but slower inference.

    Returns:
        List of PIL Images, one per page.
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page at specified DPI
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images
