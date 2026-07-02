from pydantic import BaseModel, Field
from typing import List, Literal

class Message(BaseModel):
    """
    Represents an individual turn in the stateless conversation trace.
    """
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    """
    The strict stateless payload wrapper received by POST /chat.
    """
    messages: List[Message]

class Recommendation(BaseModel):
    """
    A single structured assessment selection extracted directly from the SHL catalog.
    """
    name: str
    url: str
    test_type: str  # Typically 'K' for Knowledge/Skills, 'P' for Personality/Behavior based on traces

class ChatResponse(BaseModel):
    """
    The final mandated output schema that the automated evaluation harness parses.
    """
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool