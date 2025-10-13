import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def list_txt_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_txt_files(data_dir="./data"):
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path, encoding='utf-8')
        docs.extend(loader.load())
    return docs

def load_pdf_files(data_dir="./data") -> List[Document]:
    docs = []
    paths = Path(data_dir).glob('**/*.pdf')
    for path in paths:
        print(f"Loading PDF {path}")
        try:
            pdf_reader = PdfReader(str(path))
            for num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(page_content=text, metadata={'source': path.name, 'page': num + 1}))
        except Exception as e:
            print(f"Không thể đọc {path}: {e}")
    return docs

def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None) -> List[Document]:
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            docs.append(Document(page_content=text, metadata={'title': title, 'page': (num + 1)}))
    else:
        doc_text = uploaded_file.read().decode()
        docs.append(Document(page_content=doc_text, metadata={'title': title}))
    return docs

def load_all_local_docs(data_dir="./data") -> List[Document]:
    """
    Tải tất cả PDF, TXT, CSV trong thư mục data.
    """
    docs = []
    docs.extend(load_pdf_files(data_dir))
    docs.extend(load_txt_files(data_dir))
    docs.extend(load_csv_files(data_dir))
    print(f"Tổng số tài liệu đã load: {len(docs)}")
    return docs

if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    for doc in docs:
        print(doc)
    docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    for doc in docs:
        print(doc)
    txt_docs = load_txt_files("examples")
    for doc in txt_docs:
        print(doc)
    csv_docs = load_csv_files("examples")
    for doc in csv_docs:
        print(doc)

