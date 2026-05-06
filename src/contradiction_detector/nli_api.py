from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.contradiction_detector.detector import ContradictionDetector

app = FastAPI(title="NLI Service", version="1.0.0")

class ClassifyRequest(BaseModel):
    main_fact: str
    facts: List[str]

class ClassifyResponse(BaseModel):
    results: List[Dict[str, str]]

detector = ContradictionDetector()

@app.post("/classify")
async def classify_relations(request: ClassifyRequest):
    """
    Классификация отношений между главным фактом и списком фактов.
    Пример запроса: POST /classify {"main_fact": "Главный факт", "facts": ["Факт 1", "Факт 2"]}
    Пример ответа: {"results": [{"fact": "Факт 1", "relation": "entailment"}, {"fact": "Факт 2", "relation": "contradiction"}]}
    """
    try:
        logger.info(f"Classifying relations for main fact and {len(request.facts)} facts")
        dict_of_facts = {"main_fact_what_is_going_to_be_checked": request.main_fact, "list_of_facts": request.facts}
        result = detector.detect_all(dict_of_facts)
        results = [{"fact": ref_fact, "relation": rel["rel_class"]} for ref_fact, rel in result.get("nli_results", {}).items()]
        return ClassifyResponse(results=results)
    except Exception as e:
        logger.error(f"NLI classification failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok"}