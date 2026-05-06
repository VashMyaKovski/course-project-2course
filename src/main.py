import uvicorn
from fastapi import FastAPI

from src.api.endpoints.contradiction import router as contradiction_router
from src.api.endpoints.healthcheck import router as healthcheck_router

app = FastAPI()

app.include_router(healthcheck_router, tags=["Healthcheck"])
app.include_router(contradiction_router, tags=["Contradiction"], prefix="/contradiction")

uvicorn.run(app, host="0.0.0.0", port=8000)