from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re
import os
import json

def processing_text(text: str):
    # thay thế "\n\n+" thành "\n\n" và loại bỏ khoảng trắng đầu và cuối câu
    text = re.sub(r"\n\n+", "\n\n", text).strip()
    # xóa các ký tự chứa surrogate
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text

def loadPDF_from_file(path_pdf: str):

    #tải file fpd từ đường dẫn local
    pdf_loader = PyPDFLoader(path_pdf)
    docs = pdf_loader.load()
    for doc in docs:
        doc.page_content = processing_text(doc.page_content)
    print(f"len document: {len(docs)}") 
    
    # chia nhỏ các document thành các chunks
    separators: List[str]  = ["\n\n", "\n", " ", ""]
    char_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100, separators=separators)
    docs_splitted = char_splitter.split_documents(docs)
    print(f"len document splitted: {len(docs_splitted)}") 
    return docs_splitted

def save_documents(path_pdf: str, directory: str):

    # tải file pdf từ đường dẫn
    docs = loadPDF_from_file(path_pdf)
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

    # lấy tên file pdf
    file_pdf = os.path.split(path_pdf)[1]
    # thay phần đuôi mở rộng pdf thành json
    file_json = os.path.splitext(file_pdf)[0] + '.json'
    # tạo đường dẫn tới file json
    path_json = os.path.join(directory,file_json)
    with open(file = path_json, mode="w") as f:
        json.dump(save_data, f, indent = 4)

    print("saved to json file")

def main():
    file_pdf_local = r"data\1506.02640v5.pdf"

    save_documents(file_pdf_local, "data")

if __name__ == "__main__":
    main()