import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from main import index_document
from rag import answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_document("doc.txt")
    yield


app = FastAPI(title="API", lifespan=lifespan)
APP_API_KEY = os.getenv("APP_API_KEY")  # اختياري
RATE_LIMIT = 10  # عدد الطلبات المسموح في الدقيقة لكل IP
_hits: dict[str, deque] = defaultdict(deque)


def protect(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
    now = time.time()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, try again in a minute")
    window.append(now)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=7)


class Source(BaseModel):
    chunk: int
    distance: float
    text: str


class Reply(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Reply, dependencies=[Depends(protect)])
def ask(body: Question) -> Reply:
    text, top = answer(body.question, top_k=body.top_k)
    return Reply(
        answer=text,
        sources=[
            Source(chunk=meta["chunk"], distance=dist, text=doc)
            for doc, meta, dist in top
        ],
    )