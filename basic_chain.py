# basic_chain.py
import os
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Thêm biến môi trường để override OpenAI endpoint
os.environ["OPENAI_API_KEY"] = "sk-w468H09Lz4NbERIUBcB25b47Fd454577A1F52498Fa377860"
#đăng kí tài khoản dịch vụ của bạn
os.environ["OPENAI_API_BASE"] = "https://api.llm.ai.vn/v1"
#nơi gửi yêu cầu


def get_model(model_name="gpt-4o"):
    """Load model qua dịch vụ OpenAI-compatible"""
    return ChatOpenAI(
        model_name=model_name,
        temperature=0,
    )

def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template(
            "Tell me the most noteworthy books by the author {author}"
        )
    chain = prompt | model
    return chain

