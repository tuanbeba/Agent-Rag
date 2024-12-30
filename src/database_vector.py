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

    vectorstore = connect_Chroma("collection_1","./chroma_test")
    # vectorstore.delete_collection()

    # Kết nối với client Chroma từ vectorstore
    client = vectorstore._client

    # Lấy danh sách collections
    collections = client.list_collections()

    for collection in collections:

        print(f"Collection Name: {collection.name}")
        print(f"Collection id: {collection.id}")
        print(f"the length vector of collection {collection.count()}")
              

    # Initialization from client
    # persistent_client = chromadb.PersistentClient()
    # # lấy collection
    # collection = persistent_client.get_or_create_collection("collection_name")


    # # Lấy số lượng vector trong vector store
    # num_vectors = collection.count()
    # print(f"Number of vectors in the store: {num_vectors}")
    # # Truy xuất toàn bộ thông tin từ vector store
    # data = collection.get(include=["documents", "embeddings", "metadatas"])

    # # Kiểm tra nếu không có data['ids'] thì truy xuất từ dữ liệu
    # ids = data.get("ids", [])

    # # In các thông tin chi tiết về các vector
    # for idx, doc_id in enumerate(ids):
    #     print(f"ID: {doc_id}")
    #     print(f"Document: {data['documents'][idx]}")
    #     print(f"Dimention of Embedding: {len(data['embeddings'][idx])}")
    #     print(f"Metadata: {data['metadatas'][idx]}")
    #     print("-" * 50)

if __name__ == "__main__":
    main()