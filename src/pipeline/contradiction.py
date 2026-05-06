import os
from typing import Any, Dict, List
import httpx
from loguru import logger
from dotenv import load_dotenv
from src.chunking.paragraph_chunker import ParagraphChunker
from src.ingestion.pipeline import IngestionPipeline
from src.fact_extractor import Llama31InstructChatCompletionFactExtractor
from src.report_generator.generator import ReportGenerator
from src.pipeline.base import BasePipeline

load_dotenv()

class ContradictionDetectionPipeline(BasePipeline):
    """
    Полный пайплайн RAG-системы выявления противоречий (распределенный вариант с HTTP-вызовами)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.ingestion = IngestionPipeline()
        self.chunker = ParagraphChunker()
        self.extractor = Llama31InstructChatCompletionFactExtractor()
        self.report_gen = ReportGenerator()
        self.embedding_url = os.getenv("EMBEDDING_URL", "http://localhost:8001")
        self.llm_url = os.getenv("LLM_URL", "http://localhost:11434")
        self.nli_url = os.getenv("NLI_URL", "http://localhost:8002")
        self.vector_db_url = os.getenv("VECTOR_DB_URL", "http://localhost:6333")

    async def run(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Запуск полного цикла обработки (асинхронный)
        """
        logger.info(f"Starting pipeline for file: {file_path}")
        async with httpx.AsyncClient(timeout=60.0) as client:
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

                # Шаг 3: Извлечение фактов (LLM — локальный, TODO: адаптировать для HTTP)
                self._log_step("fact_extraction")
                facts: List[str] = []
                for chunk in chunks:
                    facts.extend(self.extractor.extract(chunk))
                self.state["facts"] = facts

                # Шаг 4: Embedding (HTTP)
                self._log_step("embedding")
                resp = await client.post(f"{self.embedding_url}/embed", json={"texts": facts})
                if resp.status_code != 200:
                    raise Exception(f"Embedding failed: {resp.text}")
                embeddings = resp.json()["embeddings"]
                self.state["embeddings"] = embeddings

                # Шаг 5: Vector DB Indexing (HTTP)
                self._log_step("vector_indexing")
                resp = await client.put(f"{self.vector_db_url}/index", json={"embeddings": embeddings, "facts": facts})
                if resp.status_code != 200:
                    raise Exception(f"Vector DB index failed: {resp.text}")

                # Шаг 6: Contradiction Detection (HTTP к NLI)
                self._log_step("contradiction_detection")
                contradictions = []
                for fact in facts:
                    resp = await client.post(f"{self.nli_url}/classify", json={"main_fact": fact, "facts": [f for f in facts if f != fact]})
                    if resp.status_code != 200:
                        raise Exception(f"NLI failed: {resp.text}")
                    results = resp.json()["results"]
                    contradictions.extend([r for r in results if r["relation"] == "contradiction"])
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
                    "facts_count": len(facts),
                }

            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                return {"status": "error", "error": str(e)}
