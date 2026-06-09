from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Rahil's portfolio chatbot is running!"}

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    answer = get_answer(request.question)
    return ChatResponse(answer=answer)
