from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json


def db_vector(json_file: str):

    # đọc file json
    with open(json_file, mode='r') as f:
        data=json.load(f)
    print("loaded data")

    # chuyển đổi sang định dạng document
    docs = []
    for doc in data:
        new_doc = Document(page_content=doc["page_content"], metadata = doc["metadata"])
        docs.append(new_doc)

def main():
    pass

if __name__ == "__main__":
    main()