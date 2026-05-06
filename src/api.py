import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from loguru import logger
from dotenv import load_dotenv
from src.pipeline.contradiction import ContradictionDetectionPipeline  # Модифицированный пайплайн

load_dotenv()

app = FastAPI(title="Contradiction Detection API", version="1.0.0")

class PipelineRequest(BaseModel):
    file_path: str
    output_path: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    services: dict

@app.post("/run_pipeline")
async def run_pipeline(request: PipelineRequest):
    """
    Запуск полного пайплайна обнаружения противоречий.
    Пример запроса: POST /run_pipeline {"file_path": "/path/to/doc.pdf", "output_path": "/path/to/report.json"}
    Пример ответа: {"status": "success", "report_path": "/path/to/report.json", "contradictions_count": 2, ...}
    """
    try:
        logger.info(f"Starting pipeline for {request.file_path}")
        pipeline = ContradictionDetectionPipeline()
        result = await pipeline.run(request.file_path, request.output_path)
        return result
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/health")
async def health_check():
    """
    Проверка здоровья сервисов.
    Пример ответа: {"status": "ok", "services": {"embedding": "ok", "llm": "ok", ...}}
    """
    services = {}
    urls = {
        "embedding": os.getenv("EMBEDDING_URL"),
        "llm": os.getenv("LLM_URL"),
        "nli": os.getenv("NLI_URL"),
        "vector_db": os.getenv("VECTOR_DB_URL"),
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in urls.items():
            try:
                if url:
                    resp = await client.get(f"{url}/health" if name != "llm" else f"{url}/api/tags")  # Ollama без /health
                    services[name] = "ok" if resp.status_code == 200 else "error"
                else:
                    services[name] = "not_configured"
            except Exception:
                services[name] = "error"
    status = "ok" if all(s == "ok" for s in services.values()) else "partial"
    return HealthResponse(status=status, services=services)