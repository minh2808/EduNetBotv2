from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory

app = FastAPI()

# Thêm đoạn này ngay sau khi tạo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc giới hạn domain WebUI của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tạo retriever & model ---
docs = load_txt_files()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = ensemble_retriever_from_docs(docs, embeddings=embeddings)

# Request body OpenAI
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


# Endpoint khai báo model
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-model",
                "object": "model",
                "owned_by": "me",
            }
        ]
    }
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Rebuild memory từ messages
    chat_history = ChatMessageHistory()
    for m in req.messages:
        chat_history.add_message({"role": m["role"], "content": m["content"]})

    # Tạo chain có memory
    chain = create_full_chain(retriever, chat_memory=chat_history)

    # Lấy message cuối cùng của user
    user_msg = req.messages[-1]["content"]

    # Hỏi RAG
    response = ask_question(chain, user_msg)

    if hasattr(response, "content"):
        answer = response.content
    elif isinstance(response, dict) and "answer" in response:
        answer = response["answer"]
    else:
        answer = str(response)

    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }
        ]
    }
