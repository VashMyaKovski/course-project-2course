from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile


async def save_upload_to_temp_file(upload_file: UploadFile) -> Path:
    """
    Save FastAPI UploadFile into a temporary file and return its path.
    """
    suffix = Path(upload_file.filename or "").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
    await upload_file.seek(0)
    return Path(temp_file.name)
