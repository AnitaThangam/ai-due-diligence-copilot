from src.documents import chunk_pages

def test_chunk_pages_preserves_source_and_page():
    pages = [{"source": "sample.pdf", "page": 3, "text": "one two three four five six"}]
    chunks = chunk_pages(pages, chunk_words=4, overlap_words=1)
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["page"] == 3
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six"

def test_empty_page_produces_no_chunks():
    assert chunk_pages([{"source": "empty.pdf", "page": 1, "text": ""}]) == []
