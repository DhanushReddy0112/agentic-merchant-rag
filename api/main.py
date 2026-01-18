from fastapi import FastAPI
from pydantic import BaseModel

from agent.merchant_agent import run_agent

app = FastAPI(title="Agentic Merchant RAG API")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = run_agent(request.question)
    return {"answer": answer}

