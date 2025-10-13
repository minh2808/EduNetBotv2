
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from memory import create_memory_chain
from rag_chain import make_rag_chain

# Hàm tạo một full chain kết hợp giữa retriever, mô hình và bộ nhớ
def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model()  # Lấy mô hình LLM từ basic_chain

    # Định nghĩa hệ thống prompt cho mô hình
    system_prompt = """You are a helpful AI assistant for busy professionals trying to improve their health.
    Use the following context and the users' chat history to help the user:
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: """

    # Tạo template cho prompt với hệ thống và câu hỏi của người dùng
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # Prompt hệ thống cung cấp ngữ cảnh
            ("human", "{question}"),  # Câu hỏi của người dùng
        ]
    )

    # Tạo RAG chain (Retrieval-Augmented Generation Chain) với mô hình và bộ truy xuất
    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)

    # Tạo chain kết hợp với bộ nhớ
    chain = create_memory_chain(model, rag_chain, chat_memory)

    return chain  # Trả về chain đầy đủ có thể sử dụng để trả lời câu hỏi

# Hàm gửi câu hỏi đến chain và nhận câu trả lời
def ask_question(chain, query):
    # Gửi câu hỏi và nhận phản hồi
    response = chain.invoke(
        {"question": query},  # Truyền câu hỏi vào chuỗi
        config={"configurable": {"session_id": "foo"}}  # Cấu hình session ID cho lần trò chuyện
    )

    # Kiểm tra và in ra context nếu có
    if "context" in response:
        print("\n--- Context Retrieved ---\n")
        print(response["context"])  # In context được truy xuất từ tài liệu
    else:
        print("\n[⚠️] No context found in response!\n")  # Thông báo nếu không có context
    return response  # Trả về câu trả lời


