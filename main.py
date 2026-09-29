from fastapi import FastAPI
from pydantic import BaseModel

from rag_engine import ask_rag


app = FastAPI()


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):

    answer, sources = ask_rag(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }