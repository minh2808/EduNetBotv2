# remote_loader.py
import requests
import os
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from local_loader import get_document_text
from langchain_community.document_loaders import OnlinePDFLoader


# Đặt thư mục lưu trữ nội dung là thư mục hiện tại của tệp này.
CONTENT_DIR = os.path.dirname(__file__)

# Nếu muốn lưu trữ trong /tmp hoặc tương đương, có thể sử dụng:
# CONTENT_DIR = tempfile.gettempdir()


def load_web_page(page_url):
    """
    Hàm tải nội dung của một trang web từ URL.
    """
    loader = WebBaseLoader(page_url)
    data = loader.load()  # Tải dữ liệu từ trang web
    return data  # Trả về dữ liệu đã tải

def load_online_pdf(pdf_url):
    """
    Hàm tải tài liệu PDF trực tuyến từ URL.
    """
    loader = OnlinePDFLoader(pdf_url)
    data = loader.load()  # Tải PDF từ URL
    return data  # Trả về dữ liệu PDF đã tải

def filename_from_url(url):
    """
    Hàm lấy tên tệp từ URL.
    """
    filename = url.split("/")[-1]
    return filename

def download_file(url, filename=None):
    """
    Hàm tải tệp từ URL và lưu trữ vào thư mục CONTENT_DIR.
    """
    response = requests.get(url)  # Gửi yêu cầu GET để tải tệp
    if not filename:
        filename = filename_from_url(url)  # Lấy tên tệp từ URL nếu không có filename

    full_path = os.path.join(CONTENT_DIR, filename)  # Lưu tệp tại thư mục CONTENT_DIR

    # Lưu tệp vào thư mục đã xác định
    with open(full_path, mode="wb") as f:
        f.write(response.content)  # Ghi dữ liệu vào tệp
        download_path = os.path.realpath(f.name)  # Lấy đường dẫn đầy đủ của tệp đã lưu
    print(f"Downloaded file {filename} to {download_path}")
    return download_path  # Trả về đường dẫn của tệp đã tải xuống

def get_wiki_docs(query, load_max_docs=2):
    """
    Hàm tải tài liệu từ Wikipedia theo câu truy vấn.
    """
    wiki_loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)  # Tạo WikipediaLoader
    docs = wiki_loader.load()  # Tải tài liệu từ Wikipedia
    for d in docs:
        print(d.metadata["title"])  # In tiêu đề của từng tài liệu
    return docs  # Trả về danh sách tài liệu tải được
def load_multiple_pdfs(pdf_urls):
    """
    Hàm tải nhiều PDF từ danh sách URL.
    """
    all_docs = []  # lưu toàn bộ văn bản từ các PDF
    for url in pdf_urls:
        print(f"Đang tải: {url}")
        loader = OnlinePDFLoader(url)
        docs = loader.load()  # mỗi file trả về 1 list Document
        all_docs.extend(docs)
    return all_docs
