#Test.py
from local_loader import load_txt_files
from splitter import split_documents
from vector_store import create_vector_db, find_similar

# Load tài liệu
docs = load_txt_files()

# Kiểm tra nội dung tài liệu đã tải
print("Các tài liệu đã tải lên:")
for doc in docs:
    print(doc.page_content)

# Chia tài liệu thành các phần nhỏ hơn
splits = split_documents(docs)
print("Các phần tài liệu đã chia nhỏ:")
for split in splits:
    print(split.page_content)

# Tạo cơ sở dữ liệu vector
vector_db = create_vector_db(splits)

# Kiểm tra cơ sở dữ liệu vector
print("Vector database:")
print(vector_db)

# Tìm tài liệu tương tự
query = "IP"
similar_docs = find_similar(vector_db, query)

print("Độ dài" ,  len(similar_docs))

# In ra tài liệu tương tự tìm được
if similar_docs:
    print("Tài liệu tương tự tìm được:")
    for doc in similar_docs:
        print(doc.page_content)
else:
    print("Không tìm thấy tài liệu phù hợp.")