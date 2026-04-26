import pytest
import json
import os
from src.verification.report_generator import ReportGenerator
from src.verification.result_aggregator import VerificationResult
from src.verification.vector_store import VectorRecord
import numpy as np


@pytest.fixture
def sample_verification_result():
    """Create a sample VerificationResult for testing."""
    supporting = [
        VectorRecord(
            id="1",
            text="Supporting evidence text",
            embedding=np.array([0.1] * 768),
            metadata={"date": "2023-01-01", "source": "SupportSource"}
        )
    ]
    contradicting = [
        VectorRecord(
            id="2",
            text="Contradicting evidence text",
            embedding=np.array([0.2] * 768),
            metadata={"date": "2023-01-02", "source": "ContraSource"}
        )
    ]
    neutral = [
        VectorRecord(
            id="3",
            text="Neutral evidence text",
            embedding=np.array([0.3] * 768),
            metadata={"date": "2023-01-03", "source": "NeutralSource"}
        )
    ]
    
    return VerificationResult(
        claim="Test claim for verification",
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        neutral_evidence=neutral,
        overall_label="ENTAILMENT",
        confidence_score=0.85,
        metadata={"test": "metadata"}
    )


def test_report_generator_init():
    """Test initialization of ReportGenerator."""
    generator = ReportGenerator()
    assert generator.logger is not None


def test_generate_report(sample_verification_result):
    """Test report generation."""
    generator = ReportGenerator()
    report = generator.generate_report(sample_verification_result)
    
    # Check basic structure
    assert "claim" in report
    assert "verification" in report
    assert "evidence" in report
    assert "metadata" in report
    
    # Check claim
    assert report["claim"] == sample_verification_result.claim
    
    # Check verification section
    assert report["verification"]["overall_label"] == sample_verification_result.overall_label
    assert report["verification"]["confidence_score"] == sample_verification_result.confidence_score
    
    # Check evidence sections
    assert len(report["evidence"]["supporting"]) == 1
    assert len(report["evidence"]["contradicting"]) == 1
    assert len(report["evidence"]["neutral"]) == 1
    
    # Check evidence content (embedding should not be included)
    supporting_evidence = report["evidence"]["supporting"][0]
    assert supporting_evidence["id"] == "1"
    assert supporting_evidence["text"] == "Supporting evidence text"
    assert "metadata" in supporting_evidence
    assert "embedding" not in supporting_evidence  # Should not include embedding
    
    # Check metadata
    assert report["metadata"] == {"test": "metadata"}


def test_generate_report_empty_evidence():
    """Test report generation with empty evidence."""
    empty_result = VerificationResult(
        claim="Test claim",
        supporting_evidence=[],
        contradicting_evidence=[],
        neutral_evidence=[],
        overall_label="UNCERTAIN",
        confidence_score=0.0,
        metadata={}
    )
    
    generator = ReportGenerator()
    report = generator.generate_report(empty_result)
    
    assert report["claim"] == "Test claim"
    assert report["verification"]["overall_label"] == "UNCERTAIN"
    assert report["verification"]["confidence_score"] == 0.0
    assert report["evidence"]["supporting"] == []
    assert report["evidence"]["contradicting"] == []
    assert report["evidence"]["neutral"] == []


def test_save_report(sample_verification_result, tmp_path):
    """Test saving report to file."""
    generator = ReportGenerator()
    report = generator.generate_report(sample_verification_result)
    
    # Create a temporary file path
    file_path = tmp_path / "test_report.json"
    
    # Save the report
    generator.save_report(report, str(file_path))
    
    # Check that file exists and contains valid JSON
    assert file_path.exists()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_report = json.load(f)
    
    # Check that loaded report matches original
    assert loaded_report["claim"] == report["claim"]
    assert loaded_report["verification"] == report["verification"]
    assert loaded_report["evidence"] == report["evidence"]
    assert loaded_report["metadata"] == report["metadata"]