"""Раздача веб-интерфейса противоречий по префиксу ``/ui_contradiction``."""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_STATIC_DIR = _PROJECT_ROOT / "static"

# Полный префикс статики (используйте в HTML); mount на APIRouter с prefix часто не отдаёт файлы.
UI_STATIC_URL_PREFIX = "/ui_contradiction/static"

router = APIRouter()


@router.get("/", response_class=FileResponse)
@router.get("", response_class=FileResponse)
async def ui_contradiction_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


def mount_ui_contradiction_static(app: FastAPI) -> None:
    """Вешает StaticFiles на приложение — так стабильнее, чем router.mount под include_router(prefix=…)."""
    if _STATIC_DIR.is_dir():
        app.mount(
            UI_STATIC_URL_PREFIX,
            StaticFiles(directory=str(_STATIC_DIR)),
            name="ui_contradiction_static",
        )
