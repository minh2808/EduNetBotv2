import logging
import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from local_loader import get_document_text
from remote_loader import download_file
from splitter import split_documents
from dotenv import load_dotenv
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds
logging.basicConfig(level=logging.DEBUG)

# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)



# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    persist_dir = os.path.join("store", collection_name)
    if not texts:
        logging.warning("Empty texts passed in to create vector database")

    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    proxy_embeddings = EmbeddingProxy(embeddings)

    # Nếu đã có DB thì load lại, không cần embed lại
    if os.path.exists(persist_dir):
        logging.info(f"Loading existing Chroma DB from {persist_dir}")
        print("Loading existing Chroma DB from")
        db = Chroma(
            collection_name=collection_name,
            embedding_function=proxy_embeddings,
            persist_directory=persist_dir
        )
    else:
        logging.info(f"Creating new Chroma DB at {persist_dir}")
        print("Creating new Chroma DB at")
        db = Chroma.from_documents(
            documents=texts,
            embedding=proxy_embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        db.persist()

    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()

    pdf_filename = "examples/mal_boole.pdf"

    if not os.path.exists(pdf_filename):
        math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
    else:
        local_pdf_path = pdf_filename

    print(f"PDF path is {local_pdf_path}")

    with open(local_pdf_path, "rb") as pdf_file:
        docs = get_document_text(pdf_file, title="Analysis of Logic")

    texts = split_documents(docs)
    vs = create_vector_db(texts)

    results = find_similar(vs, query="What is meant by the simple conversion of a proposition?")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        # cap to max length but split by words.
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")


if __name__ == "__main__":
    main()
