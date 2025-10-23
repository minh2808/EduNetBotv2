# rag_backend_openwebui.py

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

# 🧩 Import DialogueManager
from dialogue.manager import DialogueManager


# -------------------------------
# 🚀 Tạo FastAPI app
# -------------------------------
app = FastAPI(title="Smart RAG + Dialogue Backend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 📘 Load tài liệu & retriever
# -------------------------------
print("🔄 Đang load tài liệu & embeddings ...")

docs = load_txt_files()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = ensemble_retriever_from_docs(docs, embeddings=embeddings)

print("✅ Retriever sẵn sàng!")

# -------------------------------
# 🧠 DialogueManager
# -------------------------------
dm = DialogueManager(user_name="Bạn")


# -------------------------------
# 🧩 Request body
# -------------------------------
class ChatRequest(BaseModel):
    model: str = "rag-model"
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


# -------------------------------
# 🧠 Endpoint: Model list
# -------------------------------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "rag-model", "object": "model", "owned_by": "local"},
        ],
    }


# -------------------------------
# 💬 Endpoint: Chat completions
# -------------------------------
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # --- Build lại chat history ---
    chat_history = ChatMessageHistory()
    for msg in req.messages:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            chat_history.add_ai_message(msg["content"])

    # --- Lấy prompt cuối cùng ---
    user_prompt = req.messages[-1]["content"]

    # --- DialogueManager xử lý trước ---
    dm_response = dm.handle_input(user_prompt)

    # --- Nếu DialogueManager chưa hiểu → fallback RAG ---
    if not dm_response or "chưa hiểu" in dm_response.lower() or "xin lỗi" in dm_response.lower():
        chain = create_full_chain(retriever, chat_memory=chat_history)
        response = ask_question(chain, user_prompt)
        if hasattr(response, "content"):
            answer = response.content
        elif isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            answer = str(response)
    else:
        answer = dm_response

    # --- Trả kết quả về Open WebUI ---
    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "created": 1234567890,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }
        ]
    }


