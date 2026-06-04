from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatMessage] = []


class Source(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []