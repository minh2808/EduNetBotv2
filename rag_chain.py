# rag_chain.py

import os
from dotenv import load_dotenv
from langchain import hub  # Dùng để lấy các prompt từ Langchain hub
from langchain_core.output_parsers import StrOutputParser  # Phân tích đầu ra dưới dạng chuỗi
from langchain_core.prompts import ChatPromptTemplate  # Định nghĩa mẫu prompt cho chatbot
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # Các chạy công việc có thể kết hợp
from langchain_core.messages.base import BaseMessage  # Định nghĩa tin nhắn trong LangChain

from basic_chain import basic_chain, get_model  
from remote_loader import get_wiki_docs  
from splitter import split_documents  
from vector_store import create_vector_db 


def find_similar(vs, query):
    docs = vs.similarity_search(query) 
    return docs 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

# Hàm lấy câu hỏi từ đầu vào (input), có thể là chuỗi, dict, hoặc BaseMessage
def get_question(input):
    if not input:
        return None 
    elif isinstance(input, str):
        return input  
    elif isinstance(input, dict) and 'question' in input:
        return input['question']  
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")  # Nếu không hợp lệ

def make_rag_chain(model, retriever, rag_prompt=None):
    # Nếu không có prompt, tải prompt mặc định từ LangChain Hub
    if not rag_prompt:
        rag_prompt = hub.pull("rlm/rag-prompt")  # Lấy prompt từ LangChain Hub

    # Sử dụng RunnablePassthrough để thêm các bước tùy chỉnh vào chuỗi RAG
    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,  
                "question": RunnablePassthrough()  # Truyền qua câu hỏi
            }
            | rag_prompt  # Áp dụng prompt vào câu hỏi
            | model 
    )

    return rag_chain  # Trả về chuỗi RAG
