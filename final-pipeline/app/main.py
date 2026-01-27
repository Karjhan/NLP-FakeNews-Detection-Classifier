from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from .pipeline import FakeNewsPipeline, PipelineInput

app = FastAPI(
    title="Romanian Fake News Detector",
    version="1.0.0",
)

PIPELINE = FakeNewsPipeline()


class PredictRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Article/post title (optional)")
    claim: Optional[str] = Field(default=None, description="Claim statement (optional)")
    body: Optional[str] = Field(default=None, description="Full text/body (optional)")
    source_url: Optional[str] = Field(default=None, description="URL of the source/article/post (optional)")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.on_event("startup")
def _startup():
    PIPELINE.load()


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    inp = PipelineInput(
        title=req.title,
        claim=req.claim,
        body=req.body,
        source_url=req.source_url,
    )
    return PIPELINE.predict(inp)
