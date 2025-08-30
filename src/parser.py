from __future__ import annotations
import io
from typing import Optional

def parse_text_bytes(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding, errors="ignore")

def _try_pypdf(data: bytes) -> str:
    try:
        import pypdf
        text = []
        reader = pypdf.PdfReader(io.BytesIO(data))
        for page in reader.pages:
            t = page.extract_text() or ""
            text.append(t)
        return "\n".join(text).strip()
    except Exception:
        return ""

def _try_pdfminer(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            return (extract_text(tmp.name) or "").strip()
    except Exception:
        return ""

def _try_ocr(data: bytes, max_pages: int = 3) -> str:
    """Very light OCR fallback (first few pages only)."""
    try:
        import tempfile, fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import numpy as np
        # fitz is optional; install via: pip install pymupdf
        out = []
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            doc = fitz.open(tmp.name)
            for i, page in enumerate(doc):
                if i >= max_pages: break
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                txt = pytesseract.image_to_string(img)
                out.append(txt)
        return "\n".join(out).strip()
    except Exception:
        return ""

def parse_pdf_bytes(data: bytes, use_ocr: bool = False) -> str:
    """Parse PDF with multi-backend fallback."""
    # 1) pypdf (cepat)
    text = _try_pypdf(data)
    if text:
        return text
    # 2) pdfminer (lebih dalam)
    text = _try_pdfminer(data)
    if text:
        return text
    # 3) optional OCR (untuk scan)
    if use_ocr:
        text = _try_ocr(data)
        if text:
            return text
    return ""
