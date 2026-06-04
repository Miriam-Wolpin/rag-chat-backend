import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.rag import answer_chat

load_dotenv()

app = FastAPI(title="Agentic RAG Chat API")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_origin,
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "Agentic RAG Chat API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, sources = answer_chat(
        message=request.message,
        history=request.history,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
    )