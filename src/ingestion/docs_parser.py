from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

Metadata = dict[str, Any]
IngestedDocument = dict[str, Metadata]


class DocumentParser:
    """Extract document text and build normalized file metadata."""

    _TEXT_ENCODINGS: Final[tuple[str, ...]] = ("utf-8", "utf-8-sig", "cp1251")

    def parse(self, file_path: str) -> IngestedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        text, encoding = self._extract_text(path)
        metadata = self._build_metadata(path, encoding=encoding, content_length=len(text))
        return {text: metadata}

    def _extract_text(self, path: Path) -> tuple[str, str]:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf_text(path), "pdf-extractor"

        return self._extract_text_file(path)

    def _extract_text_file(self, path: Path) -> tuple[str, str]:
        file_bytes = path.read_bytes()

        for encoding in self._TEXT_ENCODINGS:
            try:
                return file_bytes.decode(encoding), encoding
            except UnicodeDecodeError:
                continue

        return file_bytes.decode("latin-1"), "latin-1"

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing requires dependency 'pypdf'. "
                "Install it with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(page for page in pages if page)

    def _build_metadata(
        self,
        path: Path,
        *,
        encoding: str,
        content_length: int,
    ) -> Metadata:
        stat = path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        mime_type, _ = mimetypes.guess_type(str(path))

        return {
            "uuid": str(uuid4()),
            "file_name": path.name,
            "file_stem": path.stem,
            "file_extension": path.suffix.lower(),
            "file_path": str(path.resolve()),
            "file_size_bytes": stat.st_size,
            "created_at": created_at,
            "modified_at": modified_at,
            "mime_type": mime_type or "application/octet-stream",
            "encoding": encoding,
            "content_length": content_length,
        }


def parse_document(file_path: str) -> IngestedDocument:
    """Functional API around :class:`DocumentParser`."""
    return DocumentParser().parse(file_path)
