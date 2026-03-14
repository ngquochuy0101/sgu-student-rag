from rag_sgu.config import load_settings
from rag_sgu.schemas import ExtractedDocument
from rag_sgu.text_chunking import TextChunker


def test_chunk_documents_keeps_source_metadata(tmp_path):
    settings = load_settings(base_dir=tmp_path)
    settings.chunk_size = 80
    settings.chunk_overlap = 10

    chunker = TextChunker(settings)
    document = ExtractedDocument(
        source_path=tmp_path / "sample.pdf",
        text="Thong tin sinh vien " * 100,
        page_count=10,
        ocr_pages=3,
        extraction_method="hybrid",
        source_hash="hash-123",
    )

    chunks = chunker.chunk_documents([document])

    assert len(chunks) > 2
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["source_hash"] == "hash-123" for chunk in chunks)
    assert chunks[0].metadata["chunk_id"] == 0
    assert chunks[0].metadata["page_number"] == 1
    assert "source_relpath" in chunks[0].metadata
