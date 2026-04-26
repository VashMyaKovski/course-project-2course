import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class NLIClassifier:
    """NLI classifier using DeBERTa model for contradiction detection."""

    def __init__(self, model_name_or_path: str = "microsoft/deberta-v3-large", device: str = None):
        """
        Initialize the NLI classifier.
        
        Args:
            model_name_or_path: Path to model or model name from HuggingFace
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading NLI model from {model_name_or_path} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for DeBERTa NLI models (usually: 0: entailment, 1: neutral, 2: contradiction)
        self.label_map = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        
    def predict(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Predict relationship between premise and hypothesis.
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            Tuple of (label, confidence_score)
        """
        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            
        label = self.label_map[predicted_class.item()]
        confidence_score = confidence.item()
        
        logger.debug(f"NLI prediction: {label} (confidence: {confidence_score:.4f})")
        return label, confidence_score
    
    def batch_predict(self, premises: List[str], hypotheses: List[str]) -> List[Tuple[str, float]]:
        """
        Predict relationships for multiple premise-hypothesis pairs.
        
        Args:
            premises: List of premise texts
            hypotheses: List of hypothesis texts
            
        Returns:
            List of tuples (label, confidence_score)
        """
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses lists must have the same length")
            
        results = []
        for premise, hypothesis in zip(premises, hypotheses):
            label, confidence = self.predict(premise, hypothesis)
            results.append((label, confidence))
        return results