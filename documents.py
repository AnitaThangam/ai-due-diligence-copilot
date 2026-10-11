from io import BytesIO
import fitz

def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract selectable text page by page so citations keep their page numbers."""
    pages = []
    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": " ".join(text.split())
                })
    return pages

def chunk_pages(pages: list[dict], chunk_words: int = 180, overlap_words: int = 35) -> list[dict]:
    """Split page text into overlapping word chunks; retain source and page metadata."""
    chunks = []
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be larger than overlap_words")
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            piece = words[start:start + chunk_words]
            if piece:
                chunks.append({
                    "source": page["source"],
                    "page": int(page["page"]),
                    "text": " ".join(piece)
                })
            if start + chunk_words >= len(words):
                break
            start += chunk_words - overlap_words
    return chunks
