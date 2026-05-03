from fastapi import FastAPI, HTTPException
from nli_model import ContradictionDetector
from pydantic import BaseModel

app = FastAPI(title="NLI Contradiction Detector")

# Инициализация модели при старте
detector = ContradictionDetector()


class PredictionRequest(BaseModel):
    sentence1: str
    sentence2: str


class PredictionResponse(BaseModel):
    contradiction_probability: float
    is_contradiction: bool
    predicted_label: str
    all_probabilities: dict


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Определяет степень противоречия между двумя предложениями.

    - **sentence1**: первое предложение (premise)
    - **sentence2**: второе предложение (hypothesis)
    - **return**: уверенность в противоречии и дополнительные метрики
    """
    if not request.sentence1.strip() or not request.sentence2.strip():
        raise HTTPException(
            status_code=400, detail="Оба предложения должны быть непустыми"
        )

    result = detector.detect(request.sentence1, request.sentence2)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
