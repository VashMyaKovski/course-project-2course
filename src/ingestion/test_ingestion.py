from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.ingestion.docs_parser import DocumentParser, parse_document
from src.ingestion.pipeline import IngestionPipeline


def _write_file(path: Path, content: bytes) -> None:
    path.write_bytes(content)


class TestDocumentParser:
    def test_parse_text_file_returns_text_and_metadata(self, tmp_path: Path):
        file_path = tmp_path / "document.txt"
        _write_file(file_path, b"Hello ingestion")

        result = DocumentParser().parse(str(file_path))

        assert len(result) == 1
        text, metadata = next(iter(result.items()))
        assert text == "Hello ingestion"
        assert metadata["file_name"] == "document.txt"
        assert metadata["file_stem"] == "document"
        assert metadata["file_extension"] == ".txt"
        assert metadata["encoding"] == "utf-8"
        assert metadata["content_length"] == len("Hello ingestion")
        assert metadata["file_size_bytes"] == len(b"Hello ingestion")
        assert metadata["mime_type"] == "text/plain"
        assert metadata["file_path"] == str(file_path.resolve())
        UUID(metadata["uuid"])

    def test_parse_cp1251_text_file_uses_cp1251_encoding(self, tmp_path: Path):
        file_path = tmp_path / "ru.txt"
        content = "Привет, мир!"
        _write_file(file_path, content.encode("cp1251"))

        result = DocumentParser().parse(str(file_path))

        text, metadata = next(iter(result.items()))
        assert text == content
        assert metadata["encoding"] == "cp1251"

    def test_parse_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            DocumentParser().parse("missing-file.txt")

    def test_parse_raises_for_directory_path(self, tmp_path: Path):
        directory = tmp_path / "nested"
        directory.mkdir()

        with pytest.raises(ValueError):
            DocumentParser().parse(str(directory))


def test_parse_document_function_uses_document_parser(monkeypatch: pytest.MonkeyPatch):
    expected = {"text": {"uuid": "123"}}
    parse_mock = MagicMock(return_value=expected)
    monkeypatch.setattr("src.ingestion.docs_parser.DocumentParser.parse", parse_mock)

    result = parse_document("input.txt")

    assert result == expected
    parse_mock.assert_called_once_with("input.txt")


def test_ingestion_pipeline_calls_parser_and_returns_document():
    parser = MagicMock()
    expected = {"abc": {"uuid": "u-1", "content_length": 3}}
    parser.parse.return_value = expected
    pipeline = IngestionPipeline(parser=parser)

    result = pipeline.run("doc.txt")

    assert result == expected
    parser.parse.assert_called_once_with("doc.txt")
