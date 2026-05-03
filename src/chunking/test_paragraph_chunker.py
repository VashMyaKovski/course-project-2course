import pytest

from src.chunking.paragraph_chunker import ParagraphChunker


class TestParagraphChunker:
    """Test suite for the ParagraphChunker class."""

    def test_basic_paragraph_split(self):
        """Test splitting text into paragraphs."""
        input_dict = {
            "data": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "metadata": "some-metadata",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3
        assert result["chunks"] == [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph.",
        ]
        assert result["metadata"] == "some-metadata"

    def test_multiple_newlines(self):
        """Test handling of various newline combinations."""
        input_dict = {
            "data": "Para 1\r\n\r\nPara 2\r\n\r\n\r\nPara 3",
            "metadata": "meta",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3
        assert "Para 1" in result["chunks"][0]
        assert "Para 2" in result["chunks"][1]

    def test_trailing_leading_whitespace(self):
        """Test that whitespace is trimmed from chunks."""
        input_dict = {
            "data": "  First paragraph with spaces.  \n\n  Second paragraph.  ",
            "metadata": "",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert result["chunks"][0] == "First paragraph with spaces."
        assert result["chunks"][1] == "Second paragraph."

    def test_min_chunk_size_filter(self):
        """Test that chunks smaller than min_chunk_size are filtered out."""
        input_dict = {
            "data": "This is a valid paragraph.\n\nNo.\n\nAnother valid paragraph here.",
            "metadata": "test",
        }
        chunker = ParagraphChunker(min_chunk_size=20)
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 2
        assert "This is a valid paragraph." in result["chunks"]
        assert "No." not in result["chunks"]
        assert "Another valid paragraph here." in result["chunks"]

    def test_empty_input(self):
        """Test handling of empty data field."""
        input_dict = {"data": "", "metadata": "test"}
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert result["chunks"] == []
        assert result["metadata"] == "test"

    def test_only_whitespace(self):
        """Test handling of whitespace-only input."""
        input_dict = {"data": "   \n\n   \n\n   ", "metadata": "meta"}
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert result["chunks"] == []

    def test_single_paragraph(self):
        """Test text with no paragraph separators."""
        input_dict = {
            "data": "This is just one continuous paragraph without breaks.",
            "metadata": "",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 1
        assert (
            result["chunks"][0]
            == "This is just one continuous paragraph without breaks."
        )

    def test_multiple_consecutive_separators(self):
        """Test text with multiple consecutive newline groups."""
        input_dict = {"data": "Para 1\n\n\n\nPara 2\n\n\n\n\nPara 3", "metadata": "x"}
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3

    def test_preserves_sentence_integrity(self):
        """Test that sentences are not split across chunks."""
        input_dict = {
            "data": "This is sentence one. This is sentence two.\n\nAnother paragraph here.",
            "metadata": "test",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 2
        assert "This is sentence one. This is sentence two." == result["chunks"][0]
        assert "Another paragraph here." == result["chunks"][1]

    def test_missing_metadata_field(self):
        """Test handling when metadata field is missing."""
        input_dict = {"data": "Some text here."}
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert result["metadata"] == ""
        assert result["chunks"] == ["Some text here."]

    def test_mixed_separators(self):
        """Test text with mixed \n\n and \r\n\r\n separators."""
        input_dict = {"data": "First\n\nSecond\r\n\r\nThird", "metadata": "test"}
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3

    def test_unicode_content(self):
        """Test handling of unicode characters."""
        input_dict = {
            "data": "Привет мир.\n\nمرحبا بالعالم.\n\nHello world 🌍.",
            "metadata": "",
        }
        chunker = ParagraphChunker()
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3

    def test_long_paragraph_not_split(self):
        """Test that a long paragraph remains as one chunk (no auto-splitting)."""
        long_text = " ".join(["word"] * 200)  # Very long paragraph
        input_dict = {"data": long_text, "metadata": ""}
        chunker = ParagraphChunker(max_chunk_size=500)
        result = chunker.chunk(input_dict)

        # Should still be one chunk since we split only on paragraphs
        assert len(result["chunks"]) == 1
        assert long_text.strip() == result["chunks"][0]

    def test_custom_separators(self):
        """Test using custom paragraph separators."""
        input_dict = {"data": "First||Second||Third", "metadata": ""}
        chunker = ParagraphChunker(paragraph_separators=["||"])
        result = chunker.chunk(input_dict)

        assert len(result["chunks"]) == 3
        assert result["chunks"] == ["First", "Second", "Third"]
