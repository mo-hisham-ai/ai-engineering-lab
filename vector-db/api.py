from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from main import index_document
from rag import answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_document("doc.txt")
    yield


app = FastAPI(title="Handbook RAG API", lifespan=lifespan)


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


@app.post("/ask", response_model=Reply)
def ask(body: Question) -> Reply:
    text, top = answer(body.question, top_k=body.top_k)
    return Reply(
        answer=text,
        sources=[
            Source(chunk=meta["chunk"], distance=dist, text=doc)
            for doc, meta, dist in top
        ],
    )