from langchain_chroma import Chroma
from crawler import pdf_to_text
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

def create_ChromaDB(pdf_file: str, embed_model: str, collection_name: str = "collection1", CHROMA_PATH: str = "./chroma_test"):

    # chọn model embedding text
    if embed_model == "nomic-embed-text":
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    else:
        embedding_model = OpenAIEmbeddings(mode="text-embedding-3-large")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    # chia file pdf ban đầu thành các đoạn chunk nhỏ
    chunks = pdf_to_text(pdf_file=pdf_file)
    # khởi tạo kho lưu trữ vector với Chroma (mode in-memory save disk)
    vector_store = Chroma(
    collection_name=collection_name, # tên collection
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH,  # nơi lưu data local
    )
    # xóa collection và tạo lại collection rỗng
    vector_store.reset_collection()
    # Embedding các đoạn chunk
    vector_store.add_texts(texts=chunks)
    print("vector save to disk")

    return vector_store


def connect_Chroma(embed_model: str, collection_name: str = "collection1", persist_directory: str="./chroma_test"):


    # load vector store from disk
    vector_store = Chroma(collection_name=collection_name,
    embedding_function=embed_model,
    persist_directory=persist_directory
    )
    print("connected chromaDB")

    return vector_store


def main():
    
    json_path = r'data\1506.02640v5.json'
    create_ChromaDB(json_path,"collection_1")
    # test tạo vector store và connect tới vector store sử dụng chromaDB

    vectorstore = connect_Chroma("collection_1")
    print(vectorstore._collection.count())
    # Kết nối với client Chroma từ vectorstore
    # client = vectorstore._client

    # # Lấy danh sách collections
    # collections = client.list_collections()

    # for collection in collections:

    #     print(f"Collection Name: {collection.name}")
    #     print(f"Collection id: {collection.id}")
    #     print(f"the length vector of collection {collection.count()}")
              

if __name__ == "__main__":
    main()