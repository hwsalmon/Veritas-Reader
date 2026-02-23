"""Tests for core.file_handler."""

import textwrap
from pathlib import Path

import pytest

from core.file_handler import FileHandlerError, read_file, write_docx, write_markdown


class TestReadFile:
    def test_read_txt(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        assert read_file(f) == "Hello, world!"

    def test_read_md(self, tmp_path):
        f = tmp_path / "sample.md"
        content = "# Title\n\nParagraph."
        f.write_text(content, encoding="utf-8")
        assert read_file(f) == content

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "sample.pdf"
        f.write_text("data")
        with pytest.raises(FileHandlerError, match="Unsupported file type"):
            read_file(f)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileHandlerError, match="File not found"):
            read_file(tmp_path / "nonexistent.txt")


class TestWriteMarkdown:
    def test_writes_utf8(self, tmp_path):
        out = tmp_path / "out.md"
        write_markdown("# Hello\n\nWorld.", out)
        assert out.read_text(encoding="utf-8") == "# Hello\n\nWorld."


class TestWriteDocx:
    def test_creates_docx(self, tmp_path):
        out = tmp_path / "out.docx"
        write_docx("# Title\n\nBody text.", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_headings_are_preserved(self, tmp_path):
        from docx import Document
        out = tmp_path / "out.docx"
        write_docx("# H1\n## H2\n### H3\nNormal.", out)
        doc = Document(str(out))
        styles = [p.style.name for p in doc.paragraphs if p.text.strip()]
        assert "Heading 1" in styles
        assert "Heading 2" in styles
        assert "Heading 3" in styles
