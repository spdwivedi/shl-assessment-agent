from fastapi import FastAPI, status
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    description="Automated Agent Agentic Selection Engine for Individual Test Solutions",
    version="1.0.0"
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Readiness probe for cold-start web hosting architectures.
    Must return {"status": "ok"} with an HTTP 200 code.
    """
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Stateless main router processing the full conversation loop history.
    Caps timeouts strictly below the evaluation limit.
    """
    # Placeholder block testing schema structural integrity
    # We will hook our advanced hybrid search and prompt routing logic into this block next
    test_response = {
        "reply": "System schema validation successful. Core engine parsing ready.",
        "recommendations": [],
        "end_of_conversation": False
    }
    return test_response