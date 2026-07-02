from fastapi import FastAPI, status, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.core_agent import SHLConversationAgent

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    description="Automated Agent Agentic Selection Engine for Individual Test Solutions",
    version="1.0.0"
)

# Initialize the core agent into server memory once during lifecycle startup
agent = SHLConversationAgent()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Readiness probe for evaluation harness testing.
    Must return {"status": "ok"} with an HTTP 200 code.
    """
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """
    Stateless router processing incoming conversation loops.
    Delegates synchronous calls to internal FastAPI thread workers safely.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Conversation message list cannot be empty.")
        
    try:
        # Synchronous invocation bypasses loop deadlocks entirely
        response = agent.process_chat(request.messages)
        return response
    except Exception as e:
        print(f"❌ Critical runtime routing exception: {e}")
        raise HTTPException(status_code=500, detail="Internal processing sequence anomaly.")