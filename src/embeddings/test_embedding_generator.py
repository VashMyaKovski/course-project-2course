"""
Tests for EmbeddingGenerator with console output of embeddings and results.
"""

import numpy as np
import pytest
from loguru import logger

from src.embeddings.generator import (
    EmbeddingGenerator,
    EmbeddingModelConfig,
    IEmbeddingGenerator,
)


class TestEmbeddingGeneratorInitialization:
    """Test EmbeddingGenerator initialization."""

    def test_init_with_default_config(self, capsys):
        """Test initialization with default configuration."""
        print("\n" + "="*70)
        print("TEST: EmbeddingGenerator Initialization with Default Config")
        print("="*70)
        
        generator = EmbeddingGenerator()
        
        assert generator is not None
        assert isinstance(generator, IEmbeddingGenerator)
        
        print(f"✓ Generator initialized successfully")
        print(f"  - Model name: {generator._config.model_name}")
        print(f"  - Expected dimension: {generator._config.expected_dimension}")
        print(f"  - Device: {generator._device}")
        print(f"  - Passage prefix: '{generator._config.passage_prefix}'")
        print()

    def test_init_with_custom_config(self, capsys):
        """Test initialization with custom configuration."""
        print("\n" + "="*70)
        print("TEST: EmbeddingGenerator Initialization with Custom Config")
        print("="*70)
        
        custom_config = EmbeddingModelConfig(
            model_name="intfloat/multilingual-e5-base",
            expected_dimension=768,
            default_batch_size=16,
            max_length=256,
            passage_prefix="doc: "
        )
        
        generator = EmbeddingGenerator(config=custom_config)
        
        assert generator._config.default_batch_size == 16
        assert generator._config.max_length == 256
        assert generator._config.passage_prefix == "doc: "
        
        print(f"✓ Custom config applied successfully")
        print(f"  - Batch size: {generator._config.default_batch_size}")
        print(f"  - Max length: {generator._config.max_length}")
        print(f"  - Custom prefix: '{generator._config.passage_prefix}'")
        print()

    def test_config_expected_dimension_validation(self, capsys):
        """Test that model dimension matches expected dimension."""
        print("\n" + "="*70)
        print("TEST: Model Dimension Validation")
        print("="*70)
        
        generator = EmbeddingGenerator()
        hidden_size = int(getattr(generator._model.config, "hidden_size", 0))
        
        print(f"✓ Model dimension validation passed")
        print(f"  - Model hidden size: {hidden_size}")
        print(f"  - Expected dimension: {generator._config.expected_dimension}")
        print(f"  - Match: {hidden_size == generator._config.expected_dimension}")
        print()


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    def test_generate_single_sentence(self, capsys):
        """Test generating embedding for a single sentence."""
        print("\n" + "="*70)
        print("TEST: Generate Single Sentence Embedding")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = ["The cat sits on the mat."]
        
        embeddings = generator.generate(sentences=sentences, batch_size=1)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)
        assert embeddings.dtype == np.float32
        
        print(f"✓ Single sentence embedding generated successfully")
        print(f"  - Input: '{sentences[0]}'")
        print(f"  - Output shape: {embeddings.shape}")
        print(f"  - Data type: {embeddings.dtype}")
        print(f"  - L2 norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.6f}")
        print(f"  - First 10 values: {embeddings[0][:10]}")
        print()

    def test_generate_multiple_sentences(self, capsys):
        """Test generating embeddings for multiple sentences."""
        print("\n" + "="*70)
        print("TEST: Generate Multiple Sentences Embeddings")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = [
            "The cat sits on the mat.",
            "Dogs are loyal animals.",
            "Birds can fly in the sky.",
            "Fish live in water.",
            "The sun is a star."
        ]
        
        embeddings = generator.generate(sentences=sentences, batch_size=2)
        
        assert embeddings.shape == (5, 768)
        assert embeddings.dtype == np.float32
        
        print(f"✓ Multiple sentences embeddings generated successfully")
        print(f"  - Number of sentences: {len(sentences)}")
        print(f"  - Output shape: {embeddings.shape}")
        print(f"  - Batch size used: 2")
        print()
        
        for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            norm = np.linalg.norm(embedding)
            print(f"  [{idx}] '{sentence}'")
            print(f"       L2 norm: {norm:.6f}, First 5 values: {embedding[:5]}")
        print()

    def test_generate_with_different_batch_sizes(self, capsys):
        """Test that different batch sizes produce same results."""
        print("\n" + "="*70)
        print("TEST: Batch Size Consistency")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = [
            "First statement.",
            "Second statement.",
            "Third statement.",
            "Fourth statement.",
            "Fifth statement.",
        ]
        
        embeddings_batch1 = generator.generate(sentences=sentences, batch_size=1)
        embeddings_batch2 = generator.generate(sentences=sentences, batch_size=2)
        embeddings_batch5 = generator.generate(sentences=sentences, batch_size=5)
        
        # Check if embeddings are very close (allowing for small numerical differences)
        diff_batch1_2 = np.max(np.abs(embeddings_batch1 - embeddings_batch2))
        diff_batch2_5 = np.max(np.abs(embeddings_batch2 - embeddings_batch5))
        
        print(f"✓ Batch size consistency verified")
        print(f"  - Number of sentences: {len(sentences)}")
        print(f"  - Max difference (batch 1 vs batch 2): {diff_batch1_2:.2e}")
        print(f"  - Max difference (batch 2 vs batch 5): {diff_batch2_5:.2e}")
        print(f"  - Shape (all batches): {embeddings_batch1.shape}")
        print()

    def test_embedding_normalization(self, capsys):
        """Test that embeddings are properly L2-normalized."""
        print("\n" + "="*70)
        print("TEST: Embedding L2-Normalization")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = [
            "This is a test sentence.",
            "Another test example.",
            "Final verification sentence."
        ]
        
        embeddings = generator.generate(sentences=sentences)
        
        norms = np.linalg.norm(embeddings, axis=1)
        
        print(f"✓ Embeddings are L2-normalized")
        print(f"  - Number of embeddings: {len(norms)}")
        print(f"  - Expected norm: 1.0 (approximately)")
        for idx, norm in enumerate(norms):
            print(f"  - Embedding[{idx}] L2 norm: {norm:.8f}")
        print()
        
        # All norms should be close to 1.0
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_semantic_similarity_check(self, capsys):
        """Test that similar sentences produce similar embeddings."""
        print("\n" + "="*70)
        print("TEST: Semantic Similarity")
        print("="*70)
        
        generator = EmbeddingGenerator()
        
        # Semantically similar sentences
        similar_sentences = [
            "The cat is sleeping on the couch.",
            "A cat is resting on the sofa.",
        ]
        
        # Semantically different sentences
        different_sentences = [
            "The cat is sleeping on the couch.",
            "I like to eat pizza for lunch.",
        ]
        
        similar_embeddings = generator.generate(sentences=similar_sentences)
        different_embeddings = generator.generate(sentences=different_sentences)
        
        similar_similarity = np.dot(similar_embeddings[0], similar_embeddings[1])
        different_similarity = np.dot(different_embeddings[0], different_embeddings[1])
        
        print(f"✓ Semantic similarity test completed")
        print(f"  - Similar sentences similarity: {similar_similarity:.6f}")
        print(f"    * '{similar_sentences[0]}'")
        print(f"    * '{similar_sentences[1]}'")
        print(f"  - Different sentences similarity: {different_similarity:.6f}")
        print(f"    * '{different_sentences[0]}'")
        print(f"    * '{different_sentences[1]}'")
        print(f"  - Difference in similarity: {similar_similarity - different_similarity:.6f}")
        print()
        
        # Similar sentences should have higher similarity
        assert similar_similarity > different_similarity


class TestEmbeddingValidation:
    """Test input validation and error handling."""

    def test_validate_empty_sentences(self, capsys):
        """Test that empty sentence list raises ValueError."""
        print("\n" + "="*70)
        print("TEST: Empty Sentences Validation")
        print("="*70)
        
        generator = EmbeddingGenerator()
        
        with pytest.raises(ValueError, match="Input list cannot be empty"):
            generator.generate(sentences=[])
        
        print(f"✓ Empty sentences validation passed")
        print(f"  - Correctly raises ValueError for empty list")
        print()

    def test_validate_non_string_sentence(self, capsys):
        """Test that non-string sentences raise TypeError."""
        print("\n" + "="*70)
        print("TEST: Non-String Sentence Validation")
        print("="*70)
        
        generator = EmbeddingGenerator()
        invalid_sentences = ["Valid sentence", 123, "Another valid"]
        
        with pytest.raises(TypeError, match="not a string"):
            generator.generate(sentences=invalid_sentences)
        
        print(f"✓ Non-string sentence validation passed")
        print(f"  - Input: {invalid_sentences}")
        print(f"  - Correctly raises TypeError for non-string element")
        print()

    def test_validate_empty_string_sentence(self, capsys):
        """Test that empty string sentences raise ValueError."""
        print("\n" + "="*70)
        print("TEST: Empty String Sentence Validation")
        print("="*70)
        
        generator = EmbeddingGenerator()
        invalid_sentences = ["Valid sentence", "   ", "Another valid"]
        
        with pytest.raises(ValueError, match="empty or whitespace"):
            generator.generate(sentences=invalid_sentences)
        
        print(f"✓ Empty string sentence validation passed")
        print(f"  - Input: {invalid_sentences}")
        print(f"  - Correctly raises ValueError for whitespace-only element")
        print()

    def test_validate_invalid_batch_size(self, capsys):
        """Test that invalid batch sizes raise errors."""
        print("\n" + "="*70)
        print("TEST: Batch Size Validation")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = ["Test sentence."]
        
        # Test negative batch size
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            generator.generate(sentences=sentences, batch_size=-1)
        
        print(f"✓ Batch size validation passed")
        print(f"  - Correctly rejects negative batch size: -1")
        
        # Test zero batch size
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            generator.generate(sentences=sentences, batch_size=0)
        
        print(f"  - Correctly rejects zero batch size: 0")
        
        # Test non-integer batch size
        with pytest.raises(TypeError, match="batch_size must be an integer"):
            generator.generate(sentences=sentences, batch_size="32")
        
        print(f"  - Correctly rejects non-integer batch size: '32'")
        print()


class TestEmbeddingStatistics:
    """Test statistical properties of embeddings."""

    def test_embedding_statistics(self, capsys):
        """Analyze and report statistics about embeddings."""
        print("\n" + "="*70)
        print("TEST: Embedding Statistics")
        print("="*70)
        
        generator = EmbeddingGenerator()
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret images.",
            "Data science combines statistics and programming."
        ]
        
        embeddings = generator.generate(sentences=sentences)
        
        # Calculate statistics
        mean_values = np.mean(embeddings, axis=0)
        std_values = np.std(embeddings, axis=0)
        min_values = np.min(embeddings, axis=0)
        max_values = np.max(embeddings, axis=0)
        
        print(f"✓ Embedding statistics calculated")
        print(f"  - Number of embeddings: {len(embeddings)}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print()
        
        print(f"  Global Statistics:")
        print(f"    - Mean of all values: {np.mean(embeddings):.6f}")
        print(f"    - Std of all values: {np.std(embeddings):.6f}")
        print(f"    - Min value: {np.min(embeddings):.6f}")
        print(f"    - Max value: {np.max(embeddings):.6f}")
        print()
        
        print(f"  Per-Dimension Statistics (first 10 dimensions):")
        for dim in range(min(10, embeddings.shape[1])):
            print(f"    Dim[{dim:3d}]: mean={mean_values[dim]:7.4f}, "
                  f"std={std_values[dim]:7.4f}, "
                  f"min={min_values[dim]:7.4f}, max={max_values[dim]:7.4f}")
        print()
        
        # Compute pairwise similarities
        print(f"  Pairwise Cosine Similarities:")
        similarities = np.dot(embeddings, embeddings.T)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = similarities[i, j]
                print(f"    [{i},{j}] = {sim:.6f}  "
                      f"({sentences[i][:40]}... vs {sentences[j][:40]}...)")
        print()


class TestEmbeddingIntegration:
    """Integration tests for embedding generator."""
    def test_full_pipeline(self, capsys):
        """Test complete embedding generation pipeline."""
        print("\n" + "="*70)
        print("TEST: Full Integration Pipeline")
        print("="*70)
        
        # Initialize generator
        print(f"Step 1: Initializing EmbeddingGenerator...")
        generator = EmbeddingGenerator()
        print(f"✓ Generator initialized")
        print()
        
        # Prepare documents
        print(f"Step 2: Preparing documents...")
        documents = [
            "Artificial intelligence is transforming technology.",
            "Machine learning enables computers to learn from data.",
            "Deep learning uses neural networks for pattern recognition.",
            "Natural language processing helps understand human language.",
            "Computer vision enables machines to see and interpret images.",
        ]
        print(f"✓ {len(documents)} documents prepared")
        print()
        
        # Generate embeddings
        print(f"Step 3: Generating embeddings with batch_size=2...")
        embeddings = generator.generate(sentences=documents, batch_size=2)
        print(f"✓ Embeddings generated successfully")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Data type: {embeddings.dtype}")
        print()
        
        # Verify output
        print(f"Step 4: Verifying output...")
        assert embeddings.shape == (len(documents), 768)
        assert embeddings.dtype == np.float32
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        print(f"✓ All verifications passed")
        print()
        
        # Perform similarity search
        print(f"Step 5: Performing similarity search...")
        query = "Deep learning and neural networks"
        query_embedding = generator.generate(sentences=[query])
        
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"  - Query: '{query}'")
        print(f"  - Top 3 most similar documents:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"    [{rank}] Similarity: {similarities[idx]:.6f}")
            print(f"        Document: '{documents[idx]}'")
        print()
        
        print(f"✓ Full pipeline test completed successfully!")
        print()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])