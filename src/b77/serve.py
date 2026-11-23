"""Expose redacted, bounded intent routing; this API never makes a banking decision."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .baselines import Baseline
from .guardrails import mask


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000)


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    needs_human_review: bool
    latency_ms: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Baseline()
    yield


app = FastAPI(title="Banking77 Distill — baseline service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    # Avoid echoing raw customer identifiers into captured validation responses.
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid query: provide text between 1 and 2000 characters, without extra fields"
        },
    )


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "system": "tfidf",
        "labels": len(request.app.state.model.labels),
        "model_release": "Qwen training and Hub release are not executed",
    }


@app.post("/classify", response_model=Classification)
def classify(query: Query, request: Request) -> Classification:
    start = time.perf_counter()
    model = request.app.state.model
    probability = model.probabilities([mask(query.text)])[0]
    confidence = float(probability.max())
    review = confidence < model.threshold
    return Classification(
        label="needs_human_review" if review else model.labels[int(probability.argmax())],
        confidence=confidence,
        needs_human_review=review,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
