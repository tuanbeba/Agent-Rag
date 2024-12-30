from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings

def create_ChromaDB(path_json: str, collection_name: str, CHROMA_PATH: str = "./chroma_test"):

    # đọc file json
    with open(path_json, mode='r') as f:
        data=json.load(f)
    print("loaded data from json file")

    # chuyển đổi sang định dạng document
    docs = []
    for doc in data:
        new_doc = Document(page_content=doc["page_content"], metadata = doc["metadata"])
        docs.append(new_doc)
    # tạo id cho document
    uuids = [str(uuid4()) for _ in range(len(docs))]
    # khởi tạo model embedding
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    # khởi tạo kho lưu trữ vector với Chroma (mode in-memory save disk)
    vector_store = Chroma(
    collection_name=collection_name, # tên collection
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH,  # nơi lưu data local
    )
    # xóa collection và tạo lại collection rỗng
    vector_store.reset_collection()
    # Embedding các document thành vector và lưu vào db, thêm id cho từng document
    vector_store.add_documents(documents= docs, ids = uuids)
    print("vector save to disk")

    return vector_store



def connect_Chroma(collection_name: str, persist_directory: str="./chroma_test"):

    # khởi tạo model embedding
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    # load vector store from disk
    vector_store = Chroma(collection_name=collection_name,
    embedding_function=embedding_model,
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