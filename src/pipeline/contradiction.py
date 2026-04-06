from typing import Any, Dict

from loguru import logger

from src.chunking.strategies import TextChunker
from src.contradiction_detector.detector import ContradictionDetector
from src.embeddings.generator import EmbeddingGenerator
from src.fact_extractor.extractor import FactExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.pipeline.base import BasePipeline
from src.report_generator.generator import ReportGenerator
from src.vector_db.builder import VectorDBBuilder


class ContradictionDetectionPipeline(BasePipeline):
    """
    Полный пайплайн RAG-системы выявления противоречий
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Инициализация компонентов
        self.ingestion = IngestionPipeline()
        self.chunker = TextChunker()
        self.extractor = FactExtractor()
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorDBBuilder()
        self.detector = ContradictionDetector()
        self.report_gen = ReportGenerator()

    def run(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Запуск полного цикла обработки
        """
        logger.info(f"Starting pipeline for file: {file_path}")

        try:
            # Шаг 1: Ingestion
            self._log_step("ingestion")
            documents = self.ingestion.run(file_path)
            self.state["documents"] = documents

            # Шаг 2: Chunking
            self._log_step("chunking")
            chunks = []
            for doc in documents:
                chunks.extend(self.chunker.chunk(doc["content"]))
            self.state["chunks"] = chunks

            # Шаг 3: Triplet Extraction
            self._log_step("triplet_extraction")
            triplets = []
            for chunk in chunks:
                triplets.extend(self.extractor.extract(chunk))
            self.state["triplets"] = triplets

            # Шаг 4: Embedding
            self._log_step("embedding")
            texts = [f"{t['s']} {t['p']} {t['o']}" for t in triplets]
            embeddings = self.embedder.generate(texts)
            self.state["embeddings"] = embeddings

            # Шаг 5: Vector DB Indexing
            self._log_step("vector_indexing")
            self.vector_db.build(embeddings, triplets)

            # Шаг 6: Contradiction Detection
            self._log_step("contradiction_detection")
            contradictions = self.detector.detect_all(triplets, embeddings)
            self.state["contradictions"] = contradictions

            # Шаг 7: Report Generation
            self._log_step("report_generation")
            report_path = output_path or "reports/output.json"
            self.report_gen.generate(contradictions, report_path)

            self._log_step("complete", "success")
            return {
                "status": "success",
                "report_path": report_path,
                "contradictions_count": len(contradictions),
                "triplets_count": len(triplets),
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {"status": "error", "error": str(e)}
