import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from src.verification.verification_service import VerificationService
from src.verification.vector_store import VectorStoreInterface, VectorRecord
from src.verification.result_aggregator import VerificationResult


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = AsyncMock(spec=VectorStoreInterface)
    return store


@pytest.fixture
def verification_service(mock_vector_store):
    """Create a verification service with mocked dependencies."""
    # Patch the NLI classifier to avoid loading the actual model
    with patch('src.verification.verification_service.NLIClassifier') as mock_nli_class:
        mock_nli_instance = MagicMock()
        mock_nli_instance.batch_predict.return_value = [
            ("ENTAILMENT", 0.8),
            ("ENTAILMENT", 0.9)
        ]
        mock_nli_class.return_value = mock_nli_instance
        
        service = VerificationService(vector_store=mock_vector_store)
        # Replace the nli_classifier with our mock
        service.nli_classifier = mock_nli_instance
        return service


@pytest.fixture
def sample_records():
    """Create sample VectorRecord objects for testing."""
    return [
        VectorRecord(
            id="1",
            text="The sky is blue.",
            embedding=np.array([0.1] * 768),
            metadata={"date": "2023-01-01", "source": "WeatherReport"}
        ),
        VectorRecord(
            id="2",
            text="The grass is green.",
            embedding=np.array([0.2] * 768),
            metadata={"date": "2023-01-02", "source": "NatureBlog"}
        )
    ]


@pytest.mark.asyncio
async def test_verify_claim_no_evidence(verification_service, mock_vector_store):
    """Test verification when no evidence is found."""
    mock_vector_store.search.return_value = []
    
    result = await verification_service.verify_claim("Test claim")
    
    assert result.claim == "Test claim"
    assert result.overall_label == "UNCERTAIN"
    assert result.confidence_score == 0.0
    assert len(result.supporting_evidence) == 0
    assert len(result.contradicting_evidence) == 0
    assert len(result.neutral_evidence) == 0
    mock_vector_store.search.assert_called_once()


@pytest.mark.asyncio
async def test_verify_claim_with_evidence(verification_service, mock_vector_store, sample_records):
    """Test verification with evidence found."""
    # Mock the vector store search to return sample records
    mock_vector_store.search.return_value = sample_records
    
    # The NLI classifier is already mocked in the fixture
    # We just need to set the return value for this specific test
    verification_service.nli_classifier.batch_predict.return_value = [
        ("ENTAILMENT", 0.8),  # First record entails
        ("CONTRADICTION", 0.7) # Second record contradicts
    ]
    
    result = await verification_service.verify_claim("Test claim")
    
    assert result.claim == "Test claim"
    # With weights: ENTAILMENT=0.8, CONTRADICTION=0.7
    # ENTAILMENT has higher weight, so should be overall label
    assert result.overall_label == "ENTAILMENT"
    assert result.confidence_score > 0.0
    assert len(result.supporting_evidence) == 1
    assert len(result.contradicting_evidence) == 1
    assert len(result.neutral_evidence) == 0
    
    # Verify the vector store search was called
    mock_vector_store.search.assert_called_once()
    # Verify NLI classifier was called
    verification_service.nli_classifier.batch_predict.assert_called_once()


@pytest.mark.asyncio
async def test_verify_claim_with_metadata_filters(verification_service, mock_vector_store, sample_records):
    """Test verification with metadata filters applied."""
    # Only return the record that matches the filters (record 1 with WeatherReport)
    filtered_records = [sample_records[0]]  # Only the WeatherReport record
    mock_vector_store.search.return_value = filtered_records
    verification_service.nli_classifier.batch_predict.return_value = [
        ("ENTAILMENT", 0.8)
    ]
    
    metadata_filters = {
        "date": {"start_date": "2023-01-01"},
        "source": {"sources": ["WeatherReport"], "exact_match": True}
    }
    
    result = await verification_service.verify_claim(
        claim="Test claim",
        metadata_filters=metadata_filters
    )
    
    # The vector store search should be called with the filters
    mock_vector_store.search.assert_called_once()
    call_args = mock_vector_store.search.call_args
    assert call_args.kwargs.get('filter_conditions') == metadata_filters
    
    # Should have only one piece of evidence (the WeatherReport record)
    assert len(result.supporting_evidence) == 1
    assert result.supporting_evidence[0].id == "1"


@pytest.mark.asyncio
async def test_verify_claim_and_generate_report(verification_service, mock_vector_store, sample_records):
    """Test the combined verification and report generation."""
    mock_vector_store.search.return_value = sample_records[:1]  # Just one record for simplicity
    verification_service.nli_classifier.batch_predict.return_value = [("ENTAILMENT", 0.85)]
    
    result, report = await verification_service.verify_claim_and_generate_report(
        claim="Test claim",
        report_path=None  # Don't actually save to file
    )
    
    # Check the result
    assert result.claim == "Test claim"
    assert result.overall_label == "ENTAILMENT"
    
    # Check the report structure
    assert "claim" in report
    assert "verification" in report
    assert "evidence" in report
    assert "metadata" in report
    
    assert report["claim"] == "Test claim"
    assert report["verification"]["overall_label"] == "ENTAILMENT"
    assert isinstance(report["evidence"]["supporting"], list)
    assert len(report["evidence"]["supporting"]) == 1


def test_update_confidence_threshold(verification_service):
    """Test updating the confidence threshold."""
    verification_service.update_confidence_threshold(0.8)
    assert verification_service.result_aggregator.confidence_threshold == 0.8
    
    # Test invalid threshold
    with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
        verification_service.update_confidence_threshold(1.5)