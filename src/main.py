"""Main entry point for the FactChecker API."""

import logging
from fastapi import FastAPI
from src.api.endpoints.verification import router as verification_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="FactChecker API",
    description="API for automatic detection of factual contradictions",
    version="0.1.0"
)

# Include routers
app.include_router(verification_router, prefix="/api/v1", tags=["verification"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FactChecker API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}