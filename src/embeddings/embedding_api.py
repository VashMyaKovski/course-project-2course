from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.embeddings.generator import EmbeddingGenerator

app = FastAPI(title="Embedding Service", version="1.0.0")

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

generator = EmbeddingGenerator()

@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    """
    Генерация эмбеддингов для списка текстов.
    Пример запроса: POST /embed {"texts": ["Факт 1", "Факт 2"]}
    Пример ответа: {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}
    """
    try:
        logger.info(f"Generating embeddings for {len(request.texts)} texts")
        embeddings = generator.generate(request.texts)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok"}