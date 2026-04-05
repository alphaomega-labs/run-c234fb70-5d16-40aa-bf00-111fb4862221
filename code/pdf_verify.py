from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium


def verify_pdf_readability(pdf_path: Path) -> dict[str, object]:
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    bmp = page.render(scale=1.5)
    img = bmp.to_pil()
    width, height = img.size
    pixels = img.convert("L")
    histogram = pixels.histogram()
    non_white = sum(histogram[:-5]) / max(1, sum(histogram))
    ok = width >= 800 and height >= 500 and non_white > 0.01
    return {
        "path": str(pdf_path),
        "width": width,
        "height": height,
        "non_white_ratio": float(non_white),
        "readability_ok": bool(ok),
    }
