"""PDF text extraction with fallbacks for non-extractable (e.g. image-only) PDFs."""
import os
import tempfile

# Minimum characters to consider "has text" before trying next method
_MIN_TEXT_LEN = 80


def _extract_pypdf2(pdf_path: str) -> str | None:
    """Extract text using PyPDF2 (fast, good for normal PDFs)."""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n".join(parts).strip() if parts else ""
    except Exception:
        return None


def _extract_pymupdf(pdf_path: str) -> str | None:
    """Extract text using PyMuPDF (often works when PyPDF2 returns nothing)."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            parts = []
            for page in doc:
                t = page.get_text("text")
                if t:
                    parts.append(t.strip())
            return "\n".join(parts).strip() if parts else ""
        finally:
            doc.close()
    except Exception:
        return None


def _extract_ocr(pdf_path: str) -> str | None:
    """Fallback: render each page to image and run Tesseract OCR. Requires pytesseract and Pillow."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        parts = []
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=150, alpha=False)
                mode = "RGBA" if pix.n == 4 else "RGB" if pix.n == 3 else "L"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img).strip()
                if text:
                    parts.append(text)
            return "\n".join(parts).strip() if parts else ""
        finally:
            doc.close()
    except Exception:
        return None


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extract text from a PDF using a bulletproof fallback chain:

    1. PyPDF2 – standard extraction (fast).
    2. PyMuPDF (fitz) – different parser, often gets text when PyPDF2 returns nothing.
    3. OCR (Tesseract) – if both above yield little/no text, render pages to images and OCR.
       Optional: requires pytesseract, Pillow, and system Tesseract. If not installed, skipped.

    Returns:
        Extracted text as a string, or None if all methods fail or yield no text.
    """
    path = os.path.abspath(pdf_path)
    if not os.path.isfile(path):
        return None

    # 1. Try PyPDF2 first
    text1 = (_extract_pypdf2(path) or "").strip()
    if len(text1) >= _MIN_TEXT_LEN:
        return text1

    # 2. Try PyMuPDF (often works when PyPDF2 returns nothing)
    text2 = (_extract_pymupdf(path) or "").strip()
    if len(text2) >= _MIN_TEXT_LEN:
        return text2

    # 3. Return whichever had more text (better than nothing)
    if text1 or text2:
        return text1 if len(text1) >= len(text2) else text2

    # 4. OCR fallback for image-only / scanned PDFs (optional: needs pytesseract + Pillow + Tesseract)
    text3 = (_extract_ocr(path) or "").strip()
    return text3 or None
