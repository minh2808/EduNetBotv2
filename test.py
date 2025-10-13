# test.py
import os
# Thiết lập biến môi trường USER_AGENT để tránh cảnh báo
# Thay thế 'YourAppName/v1.0' bằng tên ứng dụng của bạn
os.environ['USER_AGENT'] = 'rag-application/v1.0' 

from local_loader import load_txt_files
from splitter import split_documents
from vector_store import create_vector_db, find_similar

docs = load_txt_files()
print("Các tài liệu đã tải lên:")
for doc in docs:
    print(doc.page_content)

splits = split_documents(docs)
print("Các phần tài liệu đã chia nhỏ:")
for split in splits:
    print(split.page_content)

vector_db = create_vector_db(splits)

print("Vector database:")
print(vector_db)

query ="Tong quan mon hoc"
similar_docs = find_similar(vector_db, query)

if similar_docs:
    print("Tài liệu tương tự tìm được:")
    for doc in similar_docs:
        print(doc.page_content)
else:
    print("Không tìm thấy tài liệu phù hợp.")