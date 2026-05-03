import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from report_generator.generator import ReportGenerator as FileReportGenerator
from report_generator.llm_report_generator import OpenRouterLLMProvider, ReportGenerator
from report_generator.report_prompt import (
    ContradictionAnalysisPromptBuilder,
    ReportPromptBuilder,
)


class TestReportPromptBuilder:
    """Test suite for report prompt building."""

    def test_prompt_builder_is_abstract(self):
        """Test that ReportPromptBuilder cannot be instantiated."""
        with pytest.raises(TypeError):
            ReportPromptBuilder()

    def test_contradiction_prompt_builder_init(self):
        """Test prompt builder initialization."""
        builder = ContradictionAnalysisPromptBuilder()
        assert builder.model_name == "google/gemma-2-9b-it"
        assert builder.system_instruction is not None

    def test_build_prompt_with_empty_list(self):
        """Test prompt building with empty contradictions list."""
        builder = ContradictionAnalysisPromptBuilder()
        prompt = builder.build_prompt([])

        assert "internally consistent" in prompt.lower()
        assert "No contradictions" in prompt

    def test_build_prompt_with_contradictions(self):
        """Test prompt building with contradiction data."""
        builder = ContradictionAnalysisPromptBuilder()

        contradictions = [
            {
                "base_sentence": "The sun rises in the east.",
                "nli_results": {
                    "The sun rises in the west.": "contradiction",
                    "The sun shines during the day.": "entailment",
                },
            }
        ]

        prompt = builder.build_prompt(contradictions)

        assert "The sun rises in the east." in prompt
        assert "The sun rises in the west." in prompt
        assert "CONTRADICTION" in prompt
        assert "ENTAILMENT" in prompt

    def test_build_prompt_with_multiple_analyses(self):
        """Test prompt building with multiple contradiction sets."""
        builder = ContradictionAnalysisPromptBuilder()

        contradictions = [
            {
                "base_sentence": "Fact 1",
                "nli_results": {"Related fact 1": "neutral"},
            },
            {
                "base_sentence": "Fact 2",
                "nli_results": {"Related fact 2": "entailment"},
            },
        ]

        prompt = builder.build_prompt(contradictions)

        assert "Analysis 1:" in prompt
        assert "Analysis 2:" in prompt
        assert "Fact 1" in prompt
        assert "Fact 2" in prompt

    def test_relation_label_mapping(self):
        """Test that relation labels are correctly mapped."""
        builder = ContradictionAnalysisPromptBuilder()

        assert "CONTRADICTION" in builder._get_relation_label("contradiction")
        assert "ENTAILMENT" in builder._get_relation_label("entailment")
        assert "NEUTRAL" in builder._get_relation_label("neutral")


class TestOpenRouterLLMProvider:
    """Test suite for OpenRouter LLM provider."""

    def test_provider_init_with_explicit_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = OpenRouterLLMProvider(api_key="test-key-123")
        assert provider.api_key == "test-key-123"
        assert provider.model == OpenRouterLLMProvider.DEFAULT_MODEL
        assert provider.temperature == OpenRouterLLMProvider.DEFAULT_TEMPERATURE

    def test_provider_init_raises_without_api_key(self):
        """Test that provider raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError):
                OpenRouterLLMProvider(api_key=None)

    def test_provider_reads_api_key_from_env(self):
        """Test that provider reads API key from environment."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key-xyz"}):
            provider = OpenRouterLLMProvider(api_key=None)
            assert provider.api_key == "env-key-xyz"

    def test_custom_model_and_parameters(self):
        """Test provider with custom model and parameters."""
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            model="custom/model",
            temperature=0.5,
            max_tokens=1000,
        )
        assert provider.model == "custom/model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1000

    @patch("httpx.Client.post")
    def test_generate_report_success(self, mock_post):
        """Test successful report generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated report text"}}]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterLLMProvider(api_key="test-key")
        result = provider.generate_report("Test prompt")

        assert result == "Generated report text"
        mock_post.assert_called_once()

    @patch("httpx.Client.post")
    def test_generate_report_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_post.side_effect = Exception("API Error")

        provider = OpenRouterLLMProvider(api_key="test-key")

        with pytest.raises(RuntimeError):
            provider.generate_report("Test prompt")

    @patch("httpx.Client.post")
    def test_api_call_structure(self, mock_post):
        """Test that API call has correct structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test"}}]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterLLMProvider(api_key="test-key", model="test/model")
        provider.generate_report("test prompt")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]

        assert payload["model"] == "test/model"
        assert payload["messages"][0]["content"] == "test prompt"
        assert "temperature" in payload
        assert "max_tokens" in payload


class TestLLMReportGenerator:
    """Test suite for LLM-based report generator."""

    def test_report_generator_init(self):
        """Test report generator initialization."""
        mock_provider = Mock()
        mock_builder = Mock()

        generator = ReportGenerator(mock_provider, mock_builder)

        assert generator.llm_provider == mock_provider
        assert generator.prompt_builder == mock_builder

    def test_generate_report(self):
        """Test report generation."""
        mock_provider = Mock()
        mock_provider.generate_report.return_value = "Test report"

        mock_builder = Mock()
        mock_builder.build_prompt.return_value = "Test prompt"

        generator = ReportGenerator(mock_provider, mock_builder)

        contradictions = [
            {"base_sentence": "Fact", "nli_results": {"Related": "neutral"}}
        ]

        result = generator.generate(contradictions)

        assert result == "Test report"
        mock_builder.build_prompt.assert_called_once_with(contradictions)
        mock_provider.generate_report.assert_called_once()

    def test_generate_with_metadata(self):
        """Test report generation with metadata."""
        mock_provider = Mock()
        mock_provider.generate_report.return_value = "Report text"
        mock_provider.model = "test/model"

        mock_builder = Mock()
        mock_builder.build_prompt.return_value = "Prompt"

        generator = ReportGenerator(mock_provider, mock_builder)

        contradictions = [{"base_sentence": "Fact", "nli_results": {}}]

        result = generator.generate_with_metadata(contradictions)

        assert "timestamp" in result
        assert result["model"] == "test/model"
        assert result["num_contradictions"] == 1
        assert result["report"] == "Report text"


class TestFileReportGenerator:
    """Test suite for file-based report generator."""

    def test_init_creates_output_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        output_dir = tmp_path / "reports"

        generator = FileReportGenerator(
            output_dir=str(output_dir),
            llm_provider=Mock(),
        )

        assert output_dir.exists()

    def test_generate_saves_report(self, tmp_path):
        """Test that report is saved to file."""
        mock_provider = Mock()
        mock_provider.generate_report.return_value = "Test report content"
        mock_provider.model = "test/model"

        output_dir = tmp_path / "reports"
        generator = FileReportGenerator(
            output_dir=str(output_dir),
            llm_provider=mock_provider,
        )

        contradictions = [
            {"base_sentence": "Test fact", "nli_results": {"Related": "neutral"}}
        ]

        result = generator.generate(contradictions, report_name="test_report")

        assert result["status"] == "success"
        assert Path(result["file_path"]).exists()

        # Verify file contents
        with open(result["file_path"], "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["report"] == "Test report content"
        assert saved_data["num_contradictions"] == 1

    def test_generate_with_empty_contradictions_raises_error(self, tmp_path):
        """Test that empty contradictions list raises error."""
        generator = FileReportGenerator(
            output_dir=str(tmp_path),
            llm_provider=Mock(),
        )

        with pytest.raises(ValueError):
            generator.generate([])

    def test_unique_file_naming(self, tmp_path):
        """Test that files get unique names when duplicates exist."""
        mock_provider = Mock()
        mock_provider.generate_report.return_value = "Report"
        mock_provider.model = "test/model"

        generator = FileReportGenerator(
            output_dir=str(tmp_path),
            llm_provider=mock_provider,
        )

        contradictions = [{"base_sentence": "Fact", "nli_results": {}}]

        # Generate first report
        result1 = generator.generate(contradictions, report_name="report")
        assert Path(result1["file_path"]).exists()

        # Generate second report with same name
        result2 = generator.generate(contradictions, report_name="report")
        assert Path(result2["file_path"]).exists()

        # Files should have different names
        assert result1["file_path"] != result2["file_path"]

    def test_generate_batch(self, tmp_path):
        """Test batch report generation."""
        mock_provider = Mock()
        mock_provider.generate_report.return_value = "Report"
        mock_provider.model = "test/model"

        generator = FileReportGenerator(
            output_dir=str(tmp_path),
            llm_provider=mock_provider,
        )

        contradictions_list = [
            [{"base_sentence": "Fact 1", "nli_results": {}}],
            [{"base_sentence": "Fact 2", "nli_results": {}}],
            [{"base_sentence": "Fact 3", "nli_results": {}}],
        ]

        results = generator.generate_batch(contradictions_list, batch_name="batch")

        assert len(results) == 3
        for result in results:
            assert result["status"] == "success"
            assert Path(result["file_path"]).exists()
