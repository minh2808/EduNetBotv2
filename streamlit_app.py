# streamlit_rag_smart_full.py
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from local_loader import load_all_local_docs
from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from dialogue.manager import DialogueManager
from basic_chain import get_model, basic_chain

st.set_page_config(page_title="Smart RAG Chatbot")
st.title("Smart RAG Chatbot")

# --- DialogueManager ---
if "dm" not in st.session_state:
    st.session_state.dm = DialogueManager(user_name="Bạn")

# --- Load retriever (PDF + embeddings) chỉ 1 lần ---
@st.cache_resource
def get_retriever():
    docs = load_all_local_docs()
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)

# --- Load chain LLM + RAG ---
@st.cache_resource
def get_chain():
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever()
    # LLM GPT-4 fallback GPT-3.5
    llm = get_model()
    chain_llm = basic_chain(model=llm)
    # RAG chain
    chain_rag = create_full_chain(
        st.session_state.retriever,
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages")
    )
    return chain_llm, chain_rag

chain_llm, chain_rag = get_chain()

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn muốn hỏi gì hôm nay?"}]

# Hiển thị chat hiện tại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Nhập prompt từ user
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- DialogueManager xử lý trước ---
    dm_response = st.session_state.dm.handle_input(prompt)

    # --- Nếu DM chưa hiểu → fallback RAG ---
    if dm_response.startswith("Xin lỗi") or "chưa hiểu" in dm_response:
        with st.spinner("Thinking with RAG + LLM..."):
            rag_response = ask_question(chain_rag, prompt)
            if hasattr(rag_response, "content"):
                dm_response = rag_response.content
            elif isinstance(rag_response, dict) and "answer" in rag_response:
                dm_response = rag_response["answer"]
            else:
                dm_response = str(rag_response)

    # Hiển thị phản hồi
    with st.chat_message("assistant"):
        st.markdown(dm_response)
    st.session_state.messages.append({"role": "assistant", "content": dm_response})
