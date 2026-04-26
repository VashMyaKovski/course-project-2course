import pytest
from unittest.mock import patch, MagicMock
import torch
from src.verification.nli_classifier import NLIClassifier


@patch('src.verification.nli_classifier.AutoTokenizer.from_pretrained')
@patch('src.verification.nli_classifier.AutoModelForSequenceClassification.from_pretrained')
def test_nli_classifier_init(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    """Test initialization of NLIClassifier."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model
    
    classifier = NLIClassifier(model_name_or_path="test-model", device="cpu")
    assert classifier.device == "cpu"
    assert classifier.label_map == {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    
    # Verify that the mocks were called
    mock_tokenizer_from_pretrained.assert_called_once_with("test-model")
    mock_model_from_pretrained.assert_called_once_with("test-model")


@patch('src.verification.nli_classifier.AutoTokenizer.from_pretrained')
@patch('src.verification.nli_classifier.AutoModelForSequenceClassification.from_pretrained')
def test_nli_classifier_predict(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    """Test predict method."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    # Mock the tokenizer to return an object that has a 'to' method
    mock_encoding = MagicMock()
    mock_encoding.to.return_value = mock_encoding  # to() returns self
    mock_encoding.__getitem__.side_effect = lambda key: {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 6251, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }[key]
    mock_tokenizer.return_value = mock_encoding
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    # Mock the model output
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.2, 0.1, 0.7]])  # HIGH probability for class 2 (CONTRADICTION)
    mock_model.return_value = mock_outputs
    mock_model_from_pretrained.return_value = mock_model
    
    classifier = NLIClassifier(model_name_or_path="test-model", device="cpu")
    label, confidence = classifier.predict("Premise text", "Hypothesis text")
    
    assert label == "CONTRADICTION"
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


@patch('src.verification.nli_classifier.AutoTokenizer.from_pretrained')
@patch('src.verification.nli_classifier.AutoModelForSequenceClassification.from_pretrained')
def test_nli_classifier_batch_predict(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    """Test batch_predict method."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    # Mock the tokenizer to return an object that has a 'to' method
    mock_encoding = MagicMock()
    mock_encoding.to.return_value = mock_encoding  # to() returns self
    mock_encoding.__getitem__.side_effect = lambda key: {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 6251, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }[key]
    mock_tokenizer.return_value = mock_encoding
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    # Mock the model output for batch processing
    # We need to mock the model to return different outputs for each call to predict
    # Since batch_predict calls predict multiple times, we need to set up side effects
    mock_outputs1 = MagicMock()
    mock_outputs1.logits = torch.tensor([[0.2, 0.1, 0.7]])  # First: contradiction
    mock_outputs2 = MagicMock()
    mock_outputs2.logits = torch.tensor([[0.6, 0.3, 0.1]])  # Second: entailment
    
    mock_model.side_effect = [mock_outputs1, mock_outputs2]
    mock_model_from_pretrained.return_value = mock_model
    
    classifier = NLIClassifier(model_name_or_path="test-model", device="cpu")
    premises = ["Premise 1", "Premise 2"]
    hypotheses = ["Hypothesis 1", "Hypothesis 2"]
    results = classifier.batch_predict(premises, hypotheses)
    
    assert len(results) == 2
    assert results[0][0] == "CONTRADICTION"
    assert results[1][0] == "ENTAILMENT"
    assert all(0.0 <= conf <= 1.0 for _, conf in results)


@patch('src.verification.nli_classifier.AutoTokenizer.from_pretrained')
@patch('src.verification.nli_classifier.AutoModelForSequenceClassification.from_pretrained')
def test_nli_classifier_batch_predict_length_mismatch(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    """Test batch_predict with mismatched lengths."""
    # Setup mocks to avoid actual model loading
    mock_tokenizer = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model.from_pretrained.return_value = mock_model
    mock_model_from_pretrained.return_value = mock_model
    
    classifier = NLIClassifier(model_name_or_path="test-model", device="cpu")
    with pytest.raises(ValueError, match="Premises and hypotheses lists must have the same length"):
        classifier.batch_predict(["Premise"], ["Hypothesis 1", "Hypothesis 2"])