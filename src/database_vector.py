from langchain_chroma import Chroma
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
import os, shutil
from uuid import uuid4
import chromadb
from chromadb.config import Settings


def db_vector_from_local(path_json: str, CHROMA_PATH: str = "./chroma_langchain_db"):

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
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # # khởi tạo kho lưu trữ vector với Chroma (mode in-memory)
    # vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model, ids=uuids,collection_name='my_collection')
    # # lưu các document đã embedding vào vector store
    # print("created vector store")

    # return vector_store

    # khởi tạo kho lưu trữ vector với Chroma (mode in-memory save disk)
    # vector_store = Chroma(
    # collection_name="my_collection",
    # embedding_function=embedding_model,
    # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    # )
    # vector_store.add_documents(documents= docs, ids = uuids)



    # # Initialization from client
    # persistent_client = chromadb.PersistentClient()
    # collection = persistent_client.get_or_create_collection("collection_name")
    # collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

    # vector_store_from_client = Chroma(
    #     client=persistent_client,
    #     collection_name="collection_name",
    #     embedding_function=embedding_model,
    # )



def connect_to_database(collection_name: str, persist_directory: str):
    embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory=persist_directory
    )

    return vector_store



def main():
    
    json_path = 'data/2312.16862v3.json'
    vector_store = db_vector_from_local(json_path)

    # # Lấy số lượng vector trong vector store
    # num_vectors = vector_store._collection.count()
    # print(f"Number of vectors in the store: {num_vectors}")
    # # Truy xuất toàn bộ thông tin từ vector store
    # data = vector_store._collection.get(include=["documents", "embeddings", "metadatas"])

    # # Kiểm tra nếu không có data['ids'] thì truy xuất từ dữ liệu
    # ids = data.get("ids", [])
    # # print("Vector IDs:", ids)
    # print(f"Number of ids in the store: {len(ids)}")

    # # In các thông tin chi tiết về các vector
    # for idx, doc_id in enumerate(ids):
    #     print(f"ID: {doc_id}")
    #     print(f"Document: {data['documents'][idx]}")
    #     print(f"Embedding: {data['embeddings'][idx]}")
    #     print(f"Metadata: {data['metadatas'][idx]}")
    #     print("-" * 50)
    #     break

    # save and load from memory

    # db_vector_from_local(json_path)

    # load from disk
    # embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vector_store = Chroma(collection_name="my_collection",
    # embedding_function=embedding_function,
    # persist_directory="./chroma_langchain_db")
    # # Lấy số lượng vector trong vector store
    # num_vectors = vector_store._collection.count()
    # print(f"Number of vectors in the store: {num_vectors}")
    # # Truy xuất toàn bộ thông tin từ vector store
    # data = vector_store._collection.get(include=["documents", "embeddings", "metadatas"])

    # # Kiểm tra nếu không có data['ids'] thì truy xuất từ dữ liệu
    # ids = data.get("ids", [])

    # # In các thông tin chi tiết về các vector
    # for idx, doc_id in enumerate(ids):
    #     print(f"ID: {doc_id}")
    #     # print(f"Document: {data['documents'][idx]}")
    #     print(f"Dimention of Embedding: {len(data['embeddings'][idx])}")
    #     print(f"Metadata: {data['metadatas'][idx]}")
    #     print("-" * 50)
    #     break

    # db_vector_from_local(json_path)

        

if __name__ == "__main__":
    main()