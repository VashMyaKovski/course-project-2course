from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Vector DB Service (Qdrant)", version="1.0.0")

class IndexRequest(BaseModel):
    embeddings: List[List[float]]
    facts: List[str]

class SearchRequest(BaseModel):
    query_embedding: List[float]
    limit: int = 10

class SearchResponse(BaseModel):
    results: List[Dict]

qdrant_url = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
collection = "facts"

@app.put("/index")
async def index_embeddings(request: IndexRequest):
    """
    Индексация эмбеддингов и фактов в Qdrant.
    Пример запроса: PUT /index {"embeddings": [[0.1, ...]], "facts": ["Факт 1"]}
    Пример ответа: {"status": "indexed", "count": 1}
    """
    try:
        logger.info(f"Indexing {len(request.embeddings)} embeddings")
        points = [{"id": i, "vector": emb, "payload": {"fact": fact}} for i, (emb, fact) in enumerate(zip(request.embeddings, request.facts))]
        async with httpx.AsyncClient() as client:
            resp = await client.put(f"{qdrant_url}/collections/{collection}/points", json={"points": points})
            if resp.status_code not in [200, 201]:
                raise Exception(f"Qdrant index failed: {resp.text}")
        return {"status": "indexed", "count": len(points)}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/search")
async def search_similar(request: SearchRequest):
    """
    Поиск похожих фактов по эмбеддингу.
    Пример запроса: POST /search {"query_embedding": [0.1, ...], "limit": 5}
    Пример ответа: {"results": [{"fact": "Факт 1", "score": 0.95}, ...]}
    """
    try:
        logger.info(f"Searching with limit {request.limit}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{qdrant_url}/collections/{collection}/points/search", json={"vector": request.query_embedding, "limit": request.limit})
            if resp.status_code != 200:
                raise Exception(f"Qdrant search failed: {resp.text}")
            data = resp.json()
            results = [{"fact": point["payload"]["fact"], "score": point["score"]} for point in data.get("result", [])]
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok"}