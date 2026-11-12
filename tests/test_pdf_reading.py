"""Which pages of a PDF are read, and how."""
import io

from reportlab.pdfgen import canvas

from backend_api.app import extraction


def _pdf(pages):
    """A PDF with one page per item: text for a digital page, None for a scanned one (no text layer)."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_a_scanned_page_after_a_digital_one_is_still_read(monkeypatch):
    """OCR used to run only when the whole file had almost no text, so these pages were lost."""
    asked = []
    monkeypatch.setattr(extraction, "_ocr_page", lambda content, n: asked.append(n) or f"Haemoglobin 12.{n} g/dL")
    text = extraction.extract_text_from_pdf(_pdf(["Lipid profile: LDL 140 mg/dL, HDL 45 mg/dL", None, None]))
    assert asked == [2, 3], "only pages without a text layer go to OCR"
    assert "LDL 140" in text and "Haemoglobin 12.2" in text and "Haemoglobin 12.3" in text
    assert text.index("Page 1") < text.index("Page 2") < text.index("Page 3")


def test_ocr_is_capped_per_file(monkeypatch):
    asked = []
    monkeypatch.setattr(extraction, "_ocr_page", lambda content, n: asked.append(n) or "x")
    monkeypatch.setattr(extraction, "PDF_OCR_MAX_PAGES", 2)
    extraction.extract_text_from_pdf(_pdf([None] * 5))
    assert asked == [1, 2]


def test_pages_beyond_the_cap_are_not_read(monkeypatch):
    monkeypatch.setattr(extraction, "PDF_MAX_PAGES", 3)
    text = extraction.extract_text_from_pdf(_pdf([f"Result page {n} with enough text to count" for n in range(1, 6)]))
    assert "page 3" in text and "page 4" not in text


def test_a_file_that_is_not_a_readable_pdf_returns_none():
    assert extraction.extract_text_from_pdf(b"%PDF-1.7 truncated garbage") is None
