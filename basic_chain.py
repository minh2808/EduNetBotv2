import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Thêm biến môi trường để override OpenAI endpoint
os.environ["OPENAI_API_KEY"] = "sk-w468H09Lz4NbERIUBcB25b47Fd454577A1F52498Fa377860"
os.environ["OPENAI_API_BASE"] = "https://api.llm.ai.vn/v1"


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


def main():
    load_dotenv()  # nếu bạn có file .env, nhưng Ollama thì không cần API token

    prompt = ChatPromptTemplate.from_template(
        "Tell me the most noteworthy books by the author {author}"
    )
    chain = basic_chain(prompt=prompt) | StrOutputParser()

    results = chain.invoke({"author": "William Faulkner"})
    print(results)


if __name__ == "__main__":
    main()
