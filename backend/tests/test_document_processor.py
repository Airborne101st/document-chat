"""Unit tests for LangChain-based document processor."""

from pathlib import Path

import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document

from app.core.document_processor import (
    DocumentProcessor,
    DocumentProcessingError,
    ProcessedDocument,
    process_document,
)


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    def test_initialization_with_defaults(self):
        """Test processor initialization with default settings."""
        processor = DocumentProcessor()
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert processor.text_splitter is not None

    def test_initialization_with_custom_settings(self):
        """Test processor initialization with custom chunk settings."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200

    def test_supported_extensions(self):
        """Test that supported extensions are correct."""
        extensions = DocumentProcessor.supported_extensions()
        assert "pdf" in extensions
        assert "docx" in extensions
        assert "txt" in extensions
        assert len(extensions) == 3


class TestProcessTXT:
    """Tests for processing TXT files."""

    def test_process_simple_txt(self, tmp_path):
        """Test processing a simple text file."""
        file_path = tmp_path / "test.txt"
        content = "This is a test file.\n" * 100  # Make it long enough to chunk
        file_path.write_text(content)

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_document(file_path)

        assert isinstance(result, ProcessedDocument)
        assert result.filename == "test.txt"
        assert result.file_type == "txt"
        assert result.total_chunks > 0
        assert len(result.chunks) == result.total_chunks
        assert all(isinstance(chunk, Document) for chunk in result.chunks)

    def test_process_empty_txt(self, tmp_path):
        """Test processing an empty text file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        processor = DocumentProcessor()
        result = processor.process_document(file_path)

        # Empty files should still be processed
        assert result.filename == "empty.txt"
        assert result.file_type == "txt"

    def test_process_txt_with_unicode(self, tmp_path):
        """Test processing text file with unicode characters."""
        file_path = tmp_path / "unicode.txt"
        content = "Hello 世界 🌍 Привет\n" * 50
        file_path.write_text(content, encoding="utf-8")

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_document(file_path)

        assert result.total_chunks > 0
        # Verify unicode is preserved
        assert any("世界" in chunk.page_content for chunk in result.chunks)


class TestProcessPDF:
    """Tests for processing PDF files."""

    def create_test_pdf(self, file_path: Path, content: str):
        """Helper to create a simple test PDF."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(file_path), pagesize=letter)

        # Split content into lines and write
        y_position = 750
        for line in content.split('\n'):
            c.drawString(100, y_position, line[:80])  # Limit line length
            y_position -= 20
            if y_position < 100:  # New page if needed
                c.showPage()
                y_position = 750

        c.save()

    def test_process_pdf(self, tmp_path):
        """Test processing a PDF file."""
        file_path = tmp_path / "test.pdf"
        content = "This is a test PDF.\n" * 100
        self.create_test_pdf(file_path, content)

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_document(file_path)

        assert result.filename == "test.pdf"
        assert result.file_type == "pdf"
        assert result.total_chunks > 0
        assert all(chunk.metadata["file_type"] == "pdf" for chunk in result.chunks)
        assert all(chunk.metadata["filename"] == "test.pdf" for chunk in result.chunks)


class TestProcessDOCX:
    """Tests for processing DOCX files."""

    def create_test_docx(self, file_path: Path, paragraphs: list[str]):
        """Helper to create a test DOCX file."""
        doc = DocxDocument()
        for para_text in paragraphs:
            doc.add_paragraph(para_text)
        doc.save(str(file_path))

    def test_process_docx(self, tmp_path):
        """Test processing a DOCX file."""
        file_path = tmp_path / "test.docx"
        paragraphs = ["This is paragraph {}.".format(i) for i in range(50)]
        self.create_test_docx(file_path, paragraphs)

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_document(file_path)

        assert result.filename == "test.docx"
        assert result.file_type == "docx"
        assert result.total_chunks > 0
        assert all(chunk.metadata["file_type"] == "docx" for chunk in result.chunks)

    def test_process_empty_docx(self, tmp_path):
        """Test processing an empty DOCX file."""
        file_path = tmp_path / "empty.docx"
        self.create_test_docx(file_path, [])

        processor = DocumentProcessor()
        result = processor.process_document(file_path)

        assert result.filename == "empty.docx"


class TestErrorHandling:
    """Tests for error handling."""

    def test_nonexistent_file(self):
        """Test processing a non-existent file."""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process_document("nonexistent.txt")

    def test_unsupported_extension(self, tmp_path):
        """Test processing a file with unsupported extension."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("test content")

        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            processor.process_document(file_path)

    def test_corrupted_pdf(self, tmp_path):
        """Test processing a corrupted PDF file."""
        file_path = tmp_path / "corrupted.pdf"
        file_path.write_text("This is not a valid PDF")

        processor = DocumentProcessor()
        with pytest.raises(DocumentProcessingError):
            processor.process_document(file_path)

    def test_corrupted_docx(self, tmp_path):
        """Test processing a corrupted DOCX file."""
        file_path = tmp_path / "corrupted.docx"
        file_path.write_text("This is not a valid DOCX")

        processor = DocumentProcessor()
        with pytest.raises(DocumentProcessingError):
            processor.process_document(file_path)


class TestChunkText:
    """Tests for chunk_text method."""

    def test_chunk_raw_text(self):
        """Test chunking raw text."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "This is a test sentence. " * 50

        chunks = processor.chunk_text(text, metadata={"source": "test"})

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)

    def test_chunk_text_without_metadata(self):
        """Test chunking text without metadata."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "Short text."

        chunks = processor.chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestConvenienceFunction:
    """Tests for process_document convenience function."""

    def test_process_document_function(self, tmp_path):
        """Test the convenience function."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content " * 50)

        result = process_document(file_path, chunk_size=100, chunk_overlap=20)

        assert isinstance(result, ProcessedDocument)
        assert result.filename == "test.txt"
        assert result.total_chunks > 0

    def test_process_document_with_string_path(self, tmp_path):
        """Test convenience function with string path."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content")

        result = process_document(str(file_path))

        assert result.filename == "test.txt"


class TestMetadataEnrichment:
    """Tests for metadata enrichment."""

    def test_chunks_have_correct_metadata(self, tmp_path):
        """Test that chunks have correct metadata."""
        file_path = tmp_path / "test.txt"
        content = "Test content " * 50
        file_path.write_text(content)

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        result = processor.process_document(file_path)

        for chunk in result.chunks:
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"] == "test.txt"
            assert "file_type" in chunk.metadata
            assert chunk.metadata["file_type"] == "txt"
            assert "source" in chunk.metadata