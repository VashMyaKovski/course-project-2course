import json
import os
import time
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import get_settings
from src.embeddings.generator import EmbeddingGenerator
from src.pipeline.contradiction import ContradictionDetectionPipeline


class TestFullPipelineLive(unittest.TestCase):
    """Live integration test: file ingestion -> report output."""

    def test_run_full_pipeline_from_file_to_report(self) -> None:
        unique_report_name = f"test_full_pipeline_{int(time.time())}"
        old_qdrant_vector_size = os.getenv("QDRANT_VECTOR_SIZE")
        old_qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        temp_path: str | None = None

        try:
            embedder = EmbeddingGenerator()
            vector_size = int(embedder.generate(sentences=["probe"]).shape[1])
            os.environ["QDRANT_VECTOR_SIZE"] = str(vector_size)
            os.environ["QDRANT_COLLECTION_NAME"] = (
                f"atomic_statements_full_pipeline_{int(time.time())}"
            )
            get_settings.cache_clear()

            # Текст должен дать >= 2 фактов, иначе стадия report_generation получит пустой список.
            source_text = (
                "Солнце встает на востоке. "
                "Солнце не встает на востоке. "
                "Земля вращается вокруг Солнца."
            )
            with NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
                f.write(source_text)
                temp_path = f.name

            pipeline = ContradictionDetectionPipeline()
            result = pipeline.run(temp_path, unique_report_name)
        finally:
            if old_qdrant_vector_size is None:
                os.environ.pop("QDRANT_VECTOR_SIZE", None)
            else:
                os.environ["QDRANT_VECTOR_SIZE"] = old_qdrant_vector_size

            if old_qdrant_collection_name is None:
                os.environ.pop("QDRANT_COLLECTION_NAME", None)
            else:
                os.environ["QDRANT_COLLECTION_NAME"] = old_qdrant_collection_name
            get_settings.cache_clear()

            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "success", msg=str(result))
        self.assertGreater(result.get("chunks_count", 0), 0, msg=str(result))
        self.assertGreater(result.get("facts_count", 0), 0, msg=str(result))
        self.assertIn("contradictions", result, msg=str(result))
        self.assertIn("report", result, msg=str(result))

        report_info = result["report"]
        self.assertIsInstance(report_info, dict, msg=str(result))
        self.assertEqual(report_info.get("status"), "success", msg=str(report_info))

        report_path = Path(report_info.get("file_path", ""))
        self.assertTrue(report_path.exists(), f"Report file not created: {report_path}")

        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("timestamp", report_data)
        self.assertIn("model", report_data)
        self.assertIn("num_contradictions", report_data)
        self.assertIn("report", report_data)
        self.assertIsInstance(report_data["report"], str)
        self.assertTrue(report_data["report"].strip())


if __name__ == "__main__":
    unittest.main()
