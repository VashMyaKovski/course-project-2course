import re
from typing import Any, Dict, List, Sequence


class ParagraphChunker:
    """A smart chunker that splits text into paragraphs without breaking sentences or words."""

    DEFAULT_PARAGRAPH_SEPARATORS = ["\r\n\r\n", "\n\n", "\r\r"]

    def __init__(
        self,
        min_chunk_size: int = 1,
        max_chunk_size: int = 10000,
        paragraph_separators: Sequence[str] = None,
    ):
        """
        Initialize the chunker with optional size constraints.

        Args:
            min_chunk_size: Minimum size of a chunk in characters (default: 1)
            max_chunk_size: Maximum size of a chunk in characters (default: 1000)
            paragraph_separators: Optional custom paragraph separator strings.
        """
        self.min_chunk_size = max(1, min_chunk_size)
        self.max_chunk_size = max_chunk_size
        self.paragraph_separators = (
            list(paragraph_separators)
            if paragraph_separators is not None
            else self.DEFAULT_PARAGRAPH_SEPARATORS
        )

    def chunk(self, input_dict: Dict[str, Any]) -> Dict[str, object]:
        """
        Split input dictionary into text chunks and preserve metadata.

        Args:
            input_dict: Dictionary with 'data' and 'metadata'.

        Returns:
            Dictionary with 'chunks' and 'metadata'.
        """
        if not isinstance(input_dict, dict):
            raise TypeError("input_dict must be a dictionary")

        text = input_dict.get("data", "")
        metadata = input_dict.get("metadata", "")

        return {
            "chunks": self.chunk_text(text),
            "metadata": "" if metadata is None else str(metadata),
        }

    def chunk_text(self, text: str) -> List[str]:
        """Split raw text into paragraph chunks."""
        normalized = self._normalize_newlines(text)
        paragraphs = self._split_into_paragraphs(normalized)

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if len(paragraph.strip()) >= self.min_chunk_size
        ]

    def _normalize_newlines(self, text: str) -> str:
        """Normalize all newline types to a single `\n` representation."""
        if text is None:
            return ""
        return re.sub(r"\r\n|\r", "\n", str(text))

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split normalized text into paragraphs by blank lines or custom separators."""
        if not text or not text.strip():
            return []

        separator_pattern = self._build_separator_pattern()
        paragraphs = re.split(separator_pattern, text)
        return [paragraph for paragraph in paragraphs if paragraph.strip()]

    def _build_separator_pattern(self) -> str:
        """Build a regex pattern for paragraph separators."""
        separators = [sep for sep in self.paragraph_separators if sep]
        if not separators:
            return r"\n{2,}"

        escaped = [re.escape(self._normalize_newlines(sep)) for sep in separators]
        return "|".join(escaped)


if __name__ == "__main__":

    chunker = ParagraphChunker()

    input_data = {
        "data": "Это первый абзац, который показывает, как хранить многострочный контент в одной переменной Python.\n\nВо втором абзаце мы разделяем смысловые блоки явными символами новой строки, что позволяет гибко управлять форматированием вывода.\n\nТретий абзац завершает пример, демонстрируя готовую строку, которую можно сразу передать в функции печати, логирования или сохранения в файл.",
        "metadata": "date:21.01.2021, author: ME",
    }

    print(chunker.chunk(input_dict=input_data))
