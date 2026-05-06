from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.schemas.schemas import DetectionRequest, DetectionResponse
from src.pipeline.contradiction import ContradictionDetectionPipeline
from src.utils import save_upload_to_temp_file

app = FastAPI(
    title="Contradiction Detection API",
    description="API для выявления фактологических противоречий",
    version="0.1.0",
)


@app.get("/")
async def healthcheck():
    return {"status": "ok"}


@app.post("/detect_contradictions", response_model=DetectionResponse)
async def detect_contradictions(
    file: UploadFile = File(...),
    report_name: str | None = Form(None),
):
    request = DetectionRequest(report_name=report_name)
    temp_file_path = await save_upload_to_temp_file(file)

    try:
        pipeline = ContradictionDetectionPipeline()
        pipeline_result = pipeline.run(str(temp_file_path), request.report_name)
    finally:
        Path(temp_file_path).unlink(missing_ok=True)

    if pipeline_result.get("status") != "success":
        raise HTTPException(status_code=500, detail=pipeline_result.get("error", "Pipeline failed"))

    return DetectionResponse(**pipeline_result)
