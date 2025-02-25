from langchain_chroma import Chroma
from crawler import pdf_to_text, chunk_docs
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
    chunks = chunk_docs(pdf_file=pdf_file)
    # khởi tạo kho lưu trữ vector với Chroma (mode in-memory save disk)
    vector_store = Chroma(
    collection_name=collection_name, # tên collection
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH,  # nơi lưu data local
    )
    # xóa collection và tạo lại collection rỗng
    vector_store.reset_collection()
    # Embedding các đoạn chunk
    vector_store.add_documents(documents=chunks)
    print("vector save to disk")

    return vector_store


def connect_Chroma(embed_model: str="nomic-embed-text", collection_name: str = "collection1", persist_directory: str="./chroma_test"):

    # chọn model embedding text
    if embed_model == "nomic-embed-text":
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    else:
        embedding_model = OpenAIEmbeddings(mode="text-embedding-3-large")

    # load vector store from disk
    vector_store = Chroma(collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory
    )
    print("connected chromaDB")

    return vector_store


def main():
    #test function
    # db =create_ChromaDB(pdf_file=r"data\1506.02640v5.pdf", embed_model = "nomic-embed-text")
    collection = connect_Chroma(embed_model="nomic-embed-text")
    print(len(collection.get()["metadatas"]))
              

if __name__ == "__main__":
    main()