from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from src.chunking.paragraph_chunker import ParagraphChunker
from src.contradiction_detector.detector import ContradictionDetector
from src.embeddings.generator import EmbeddingGenerator
from src.fact_extractor import Llama31InstructChatCompletionFactExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.pipeline.base import BasePipeline
from src.report_generator.generator import ReportGenerator
from src.vector_db.builder import VectorDBBuilder


class ContradictionDetectionPipeline(BasePipeline):
    """
    Полный пайплайн выявления противоречий:
    ingestion -> chunking -> fact extraction -> contradiction detection
    -> embedding -> vector indexing -> report generation.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.ingestion = IngestionPipeline()
        self.chunker = ParagraphChunker()
        self.extractor = Llama31InstructChatCompletionFactExtractor()
        self.detector = ContradictionDetector()
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorDBBuilder()
        self.report_gen = ReportGenerator()

    def run(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Запуск полного цикла обработки одного файла.
        """
        logger.info(f"Starting pipeline for file: {file_path}")

        try:
            # Шаг 1: Ingestion
            self._log_step("ingestion")
            document = self.ingestion.run(file_path)
            self.state["document"] = document
            source_text, source_metadata = next(iter(document.items()))

            # Шаг 2: Chunking
            self._log_step("chunking")
            chunking_result = self.chunker.chunk(
                {"data": source_text, "metadata": source_metadata}
            )
            chunks = chunking_result["chunks"]
            self.state["chunks"] = chunks

            # Шаг 3: извлечение атомарных фактов из чанков
            self._log_step("fact_extraction")
            facts: list[str] = []
            for chunk in chunks:
                facts.extend(self.extractor.extract(chunk))
            facts = [fact.strip() for fact in facts if fact and fact.strip()]
            self.state["facts"] = facts

            if len(facts) < 2:
                raise ValueError("Need at least 2 extracted facts for contradiction detection")

            # Шаг 4: Embedding
            self._log_step("embedding")
            embeddings = self.embedder.generate(sentences=facts)
            self.state["embeddings"] = embeddings

            # Шаг 5: Vector DB Indexing
            self._log_step("vector_indexing")
            indexed_records = self.vector_db.build(embeddings, facts)
            self.state["indexed_records_count"] = len(indexed_records)

            # Шаг 6: Retrieval + Contradiction Detection
            self._log_step("retrieval_and_contradiction_detection")
            contradictions = []
            for record in indexed_records:
                neighbors = self.vector_db.search_neighbors(
                    record.embedding,
                    exclude_point_ids=[str(record.point_id)],
                )
                neighbor_facts = [hit.text for hit in neighbors if hit.text.strip()]
                if not neighbor_facts:
                    continue

                contradictions.append(
                    self.detector.detect_all(
                        {
                            "main_fact_what_is_going_to_be_checked": record.text,
                            "list_of_facts": neighbor_facts,
                        }
                    )
                )

            if not contradictions:
                raise ValueError("No neighbor facts found for contradiction detection")
            self.state["contradictions"] = contradictions

            # Шаг 7: Report Generation
            self._log_step("report_generation")
            report_name = output_path or None
            report_result = self.report_gen.generate(contradictions, report_name)
            self.state["report"] = report_result

            self._log_step("complete", "success")
            result: Dict[str, Any] = {
                "status": "success",
                "contradictions_count": len(contradictions),
                "facts_count": len(facts),
                "chunks_count": len(chunks),
                "metadata": source_metadata,
                "contradictions": contradictions,
                "report": report_result,
            }
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {"status": "error", "error": str(e)}
