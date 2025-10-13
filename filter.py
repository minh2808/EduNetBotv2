from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MergerRetriever
from langchain.chains import RetrievalQA

from basic_chain import get_model
from ensemble import ensemble_retriever_from_docs
from remote_loader import load_web_page
from vector_store import create_vector_db

from dotenv import load_dotenv

# Hàm này tạo ra bộ truy xuất với các kỹ thuật nén tài liệu (compression) và lọc dư thừa embedding
def create_retriever(texts):
    
    #sử dụng 2 mô hình phát triển  dense và sparse ( dày đặc và thưa thớt) hay chính xác theo(ngữ nghĩa và từ khóa)
    dense_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en", encode_kwargs={'normalize_embeddings': False}
)


    # Tạo vector store cho các embedding dense và sparse
    dense_vs = create_vector_db(texts, collection_name="dense", embeddings=dense_embeddings)
    sparse_vs = create_vector_db(texts, collection_name="sparse", embeddings=sparse_embeddings)
    vector_stores = [dense_vs, sparse_vs]  # Liệt kê các vector store
    base_retrievers = [vs.as_retriever() for vs in vector_stores]
    lotr = MergerRetriever(retrievers=base_retrievers)



    #Luồng xử lý
    emb_filter = EmbeddingsRedundantFilter(embeddings=sparse_embeddings) #loai du thừa
    reordering = LongContextReorder() # sắp xếp lại
    pipeline = DocumentCompressorPipeline(transformers=[emb_filter, reordering])





    # Tạo bộ truy xuất nén và tái cấu trúc tài liệu theo ngữ cảnh
    compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr, search_kwargs={"k": 5, "include_metadata": True}
    )

    # Trả về bộ truy xuất đã nén và tái cấu trúc
    return compression_retriever_reordered


#vd
    load_dotenv()  # Tải các biến môi trường từ file .env (thường dùng để bảo mật thông tin API)

    # Đường dẫn tới cuốn sách "Problems of Philosophy" của Bertrand Russell trên Project Gutenberg
    problems_of_philosophy_by_russell = "https://www.gutenberg.org/ebooks/5827.html.images"

    # Tải tài liệu từ URL
    docs = load_web_page(problems_of_philosophy_by_russell)

    # Tạo bộ truy xuất kết hợp từ tài liệu
    ensemble_retriever = ensemble_retriever_from_docs(docs)

    # Lấy mô hình ngôn ngữ (LLM)
    llm = get_model()

    # Tạo chuỗi RAG (Retrieval-Augmented Generation) để trả lời câu hỏi với bộ truy xuất kết hợp
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=ensemble_retriever)

    # Gửi câu hỏi về các vấn đề chính của triết học theo Russell
    results = qa.invoke("What are the key problems of philosophy according to Russell?")

    # In kết quả trả lời từ mô hình
    print(results)
