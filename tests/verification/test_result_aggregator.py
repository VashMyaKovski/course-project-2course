import pytest
from src.verification.result_aggregator import ResultAggregator, VerificationResult
from src.verification.vector_store import VectorRecord
import numpy as np


@pytest.fixture
def sample_records():
    """Create sample VectorRecord objects for testing."""
    return [
        VectorRecord(
            id="1",
            text="Supporting evidence",
            embedding=np.array([0.1] * 768),
            metadata={}
        ),
        VectorRecord(
            id="2",
            text="Contradicting evidence",
            embedding=np.array([0.2] * 768),
            metadata={}
        ),
        VectorRecord(
            id="3",
            text="Neutral evidence",
            embedding=np.array([0.3] * 768),
            metadata={}
        )
    ]


def test_result_aggregator_init():
    """Test initialization of ResultAggregator."""
    aggregator = ResultAggregator(confidence_threshold=0.7)
    assert aggregator.confidence_threshold == 0.7


def test_result_aggregator_init_default():
    """Test initialization with default confidence threshold."""
    aggregator = ResultAggregator()
    assert aggregator.confidence_threshold == 0.7  # default value


def test_result_aggregator_update_confidence_threshold():
    """Test updating confidence threshold."""
    aggregator = ResultAggregator(confidence_threshold=0.5)
    aggregator.update_confidence_threshold(0.8)
    assert aggregator.confidence_threshold == 0.8


def test_result_aggregator_update_confidence_threshold_invalid():
    """Test updating confidence threshold with invalid values."""
    aggregator = ResultAggregator()
    
    with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
        aggregator.update_confidence_threshold(1.5)
    
    with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
        aggregator.update_confidence_threshold(-0.1)


def test_aggregate_results_all_entailment(sample_records):
    """Test aggregation when all evidence entails the claim."""
    aggregator = ResultAggregator(confidence_threshold=0.5)
    
    # All predictions are ENTAILMENT with high confidence
    predictions = [
        ("ENTAILMENT", 0.9),
        ("ENTAILMENT", 0.8),
        ("ENTAILMENT", 0.7)
    ]
    
    result = aggregator.aggregate_results(
        claim="Test claim",
        evidence_records=sample_records,
        nli_predictions=predictions
    )
    
    assert result.claim == "Test claim"
    assert result.overall_label == "ENTAILMENT"
    assert result.confidence_score > 0.0
    assert len(result.supporting_evidence) == 3
    assert len(result.contradicting_evidence) == 0
    assert len(result.neutral_evidence) == 0


def test_aggregate_results_mixed_predictions(sample_records):
    """Test aggregation with mixed predictions."""
    aggregator = ResultAggregator(confidence_threshold=0.5)
    
    # Mix of predictions
    predictions = [
        ("ENTAILMENT", 0.8),    # Supporting
        ("CONTRADICTION", 0.7), # Contradicting
        ("NEUTRAL", 0.9)        # Neutral
    ]
    
    result = aggregator.aggregate_results(
        claim="Test claim",
        evidence_records=sample_records,
        nli_predictions=predictions
    )
    
    assert result.claim == "Test claim"
    # With weights: ENTAILMENT=0.8, CONTRADICTION=0.7, NEUTRAL=0.9
    # NEUTRAL has highest weight, so should be overall label
    assert result.overall_label == "NEUTRAL"
    assert result.confidence_score > 0.0
    assert len(result.supporting_evidence) == 1
    assert len(result.contradicting_evidence) == 1
    assert len(result.neutral_evidence) == 1


def test_aggregate_results_below_threshold(sample_records):
    """Test aggregation when all predictions are below confidence threshold."""
    aggregator = ResultAggregator(confidence_threshold=0.9)  # High threshold
    
    # All predictions below threshold
    predictions = [
        ("ENTAILMENT", 0.5),
        ("CONTRADICTION", 0.4),
        ("NEUTRAL", 0.3)
    ]
    
    result = aggregator.aggregate_results(
        claim="Test claim",
        evidence_records=sample_records,
        nli_predictions=predictions
    )
    
    assert result.claim == "Test claim"
    assert result.overall_label == "UNCERTAIN"
    assert result.confidence_score == 0.0
    assert len(result.supporting_evidence) == 0
    assert len(result.contradicting_evidence) == 0
    assert len(result.neutral_evidence) == 0


def test_aggregate_results_empty_inputs():
    """Test aggregation with empty inputs."""
    aggregator = ResultAggregator()
    
    result = aggregator.aggregate_results(
        claim="Test claim",
        evidence_records=[],
        nli_predictions=[]
    )
    
    assert result.claim == "Test claim"
    assert result.overall_label == "UNCERTAIN"
    assert result.confidence_score == 0.0
    assert len(result.supporting_evidence) == 0
    assert len(result.contradicting_evidence) == 0
    assert len(result.neutral_evidence) == 0


def test_aggregate_results_mismatched_lengths(sample_records):
    """Test aggregation with mismatched lengths of records and predictions."""
    aggregator = ResultAggregator()
    
    # 3 records but only 2 predictions
    predictions = [("ENTAILMENT", 0.8), ("CONTRADICTION", 0.7)]
    
    with pytest.raises(ValueError, match="Number of evidence records must match number of predictions"):
        aggregator.aggregate_results(
            claim="Test claim",
            evidence_records=sample_records,
            nli_predictions=predictions
        )


def test_aggregate_results_high_confidence_weights(sample_records):
    """Test that higher confidence predictions have more influence."""
    aggregator = ResultAggregator(confidence_threshold=0.5)
    
    # High confidence NEUTRAL vs low confidence ENTAILMENT and CONTRADICTION
    predictions = [
        ("ENTAILMENT", 0.6),    # weight 0.6
        ("CONTRADICTION", 0.6), # weight 0.6
        ("NEUTRAL", 0.9)        # weight 0.9 -> should win
    ]
    
    result = aggregator.aggregate_results(
        claim="Test claim",
        evidence_records=sample_records,
        nli_predictions=predictions
    )
    
    assert result.overall_label == "NEUTRAL"
    # Confidence should be 0.9 / (0.6 + 0.6 + 0.9) = 0.9 / 2.1 ≈ 0.428
    assert abs(result.confidence_score - 0.428) < 0.01