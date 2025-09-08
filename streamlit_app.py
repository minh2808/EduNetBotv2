import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files
from splitter import split_documents

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                # Xử lý cả str, AIMessage, ChatResult
                if hasattr(response, "content"):
                    answer = response.content
                elif isinstance(response, dict) and "answer" in response:
                    answer = response["answer"]
                else:
                    answer = str(response)

                st.markdown(answer)
        message = {"role": "assistant", "content": answer}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    docs = load_txt_files()
    # Thay OpenAIEmbeddings bằng HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return ensemble_retriever_from_docs(docs, embeddings=embeddings)


@st.cache_resource
def get_chain():
    retriever = get_retriever()

    st.write("Retriever type:", type(retriever))
    chain = create_full_chain(
        retriever,
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages")
    )

    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


if __name__ == "__main__":
    # Tạo chain
    chain = get_chain()

    st.subheader("Ask me questions about this week's meal plan")
    show_ui(chain, "What would you like to know?")

