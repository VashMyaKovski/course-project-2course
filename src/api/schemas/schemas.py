from fastapi import UploadFile
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    file: UploadFile


class DetectionResponse(BaseModel): ...
