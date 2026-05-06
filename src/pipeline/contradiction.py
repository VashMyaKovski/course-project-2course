from __future__ import annotations

from typing import Any, Dict, List

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

    @staticmethod
    def _build_pairwise_contradiction_inputs(facts: List[str]) -> List[Dict[str, object]]:
        payloads: List[Dict[str, object]] = []
        for idx, main_fact in enumerate(facts):
            other_facts = [fact for j, fact in enumerate(facts) if j != idx]
            if not other_facts:
                continue
            payloads.append(
                {
                    "main_fact_what_is_going_to_be_checked": main_fact,
                    "list_of_facts": other_facts,
                }
            )
        return payloads

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

            if not facts:
                raise ValueError("No facts extracted from document chunks")

            # Шаг 4: Contradiction Detection (pairwise by fact)
            self._log_step("contradiction_detection")
            contradictions = [
                self.detector.detect_all(payload)
                for payload in self._build_pairwise_contradiction_inputs(facts)
            ]
            self.state["contradictions"] = contradictions

            # Шаг 5: Embedding
            self._log_step("embedding")
            embeddings = self.embedder.generate(sentences=facts)
            self.state["embeddings"] = embeddings

            # Шаг 6: Vector DB Indexing
            self._log_step("vector_indexing")
            self.vector_db.build(embeddings, facts)

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
