from fastapi import FastAPI

from api.schemas.schemas import DetectionRequest

app = FastAPI(
    title="Contradiction Detection API",
    description="API для выявления фактологических противоречий",
    version="0.1.0",
)


@app.get("/")
async def healthcheck():
    return {"status": "ok"}


@app.post("/detect_contradictions")
async def detect_contradictions(request: DetectionRequest): ...
