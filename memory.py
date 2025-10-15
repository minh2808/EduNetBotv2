import os
from typing import List, Iterable, Any

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from basic_chain import get_model
from rag_chain import make_rag_chain

# Hàm tạo chuỗi bộ nhớ kết hợp với mô hình ngôn ngữ và truy vấn lịch sử trò chuyện
def create_memory_chain(llm, base_chain, chat_memory):
    # Mô tả chức năng của hệ thống là cải tiến câu hỏi từ lịch sử trò chuyện mà không cần trả lời
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    
    # Tạo template cho prompt để cải tiến câu hỏi
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),  # Hướng dẫn hệ thống
            MessagesPlaceholder(variable_name="chat_history"),  # Thay thế bằng lịch sử trò chuyện
            ("human", "{question}"),  # Câu hỏi của người dùng
        ]
    )

    # Tạo một chuỗi để xử lý các câu hỏi và trả về kết quả từ mô hình ngôn ngữ và chuỗi cơ bản
    runnable = contextualize_q_prompt | llm | base_chain

    # Hàm lấy lịch sử trò chuyện từ một session_id
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    # Kết hợp lịch sử trò chuyện vào chuỗi, đảm bảo rằng câu hỏi và lịch sử đều được xem xét
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",  # Câu hỏi người dùng nhập
        history_messages_key="chat_history",  # Lịch sử trò chuyện
    )
    return with_message_history


# Lớp truy xuất tài liệu từ văn bản đơn giản
class SimpleTextRetriever(BaseRetriever):
    docs: List[Document]
    """Danh sách các tài liệu."""  # Chứa danh sách các tài liệu dưới dạng Document

    @classmethod
    def from_texts(
            cls,
            texts: Iterable[str],
            **kwargs: Any,
    ):
        # Chuyển danh sách các chuỗi thành đối tượng Document
        docs = [Document(page_content=t) for t in texts]
        return cls(docs=docs, **kwargs)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Trả về tất cả các tài liệu (có thể cải tiến để chỉ trả về các tài liệu liên quan)
        return self.docs

