from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re
import os
import json

def processing_text(text: str):
    # thay thế "\n\n+" thành "\n\n" và loại bỏ khoảng trắng đầu và cuối câu
    text = re.sub(r"\n\n+", "\n\n", text).strip()
    return text

def loadPDF_from_file(file_pdf_local: str):

    #tải file fpd từ đường dẫn local
    pdf_loader = PyPDFLoader(file_pdf_local)
    docs = pdf_loader.load()
    for doc in docs:
        doc.page_content = processing_text(doc.page_content)
    print(f"len document: {len(docs)}")
    
    # chia nhỏ các document thành các chunks
    separators: List[str]  = ["\n\n", "\n", " ", ""]
    char_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300, separators=separators)
    docs_splitted = char_splitter.split_documents(docs)
    print(f"len document splitted: {len(docs_splitted)}")
    return docs_splitted

def save_documents(file_pdf: str, directory: str):

    # tải file pdf từ đường dẫn
    docs = loadPDF_from_file(file_pdf)
    # tạo dic nếu dic chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # chuyển đổi data để có thê lưu dưới dạng json
    save_data = []
    for doc in docs:
        data = {
            "metadata": doc.metadata,
            "page_content": doc.page_content
        }
        save_data.append(data)

    # save dữ liệu vào fiel json
    file_json = os.path.join(directory,"document.json")
    with open(file = file_json, mode="w") as f:
        json.dump(save_data, f, indent = 4)

    print("saved data")

def main():
    file_pdf_local = r"data\2312.16862v3.pdf"

    save_documents(file_pdf_local, "data")

if __name__ == "__main__":
    main()