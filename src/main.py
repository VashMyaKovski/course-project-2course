import uvicorn
from fastapi import FastAPI

from src.api.endpoints.contradiction import router as contradiction_router
from src.api.endpoints.healthcheck import router as healthcheck_router
from src.api.endpoints.ui_contradiction import (
    mount_ui_contradiction_static,
    router as ui_contradiction_router,
)

app = FastAPI()

app.include_router(healthcheck_router, tags=["Healthcheck"])
app.include_router(contradiction_router, tags=["Contradiction"], prefix="/contradiction")
# Mount статики раньше роутера с prefix=/ui_contradiction, иначе /ui_contradiction/static/* ловит подроутер и даёт 404.
mount_ui_contradiction_static(app)
app.include_router(ui_contradiction_router, prefix="/ui_contradiction", tags=["UI"])

uvicorn.run(app, host="0.0.0.0", port=8000)
