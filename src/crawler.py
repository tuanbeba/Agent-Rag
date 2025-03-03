from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List
import re
import os, json


def processing_text(text: str):
    # định nghĩa regex
    pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
    # xóa các ký tự không nằm trong pattern
    sub_text = re.findall(pattern, text)
    # join các chuỗi con thành một chuỗi duy nhất 
    text = ' '.join(sub_text)
    # xóa các ký tự khoảng trắng thừa
    # text = re.sub(r'\s+', ' ', text.strip())

    return text

def pdf_to_text(pdf_file: str):

    #trích xuất văn bản từ file pdf
    pdf_reader = PdfReader(pdf_file)
    print(f"len pages: {len(pdf_reader.pages)}")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # chia nhỏ text thành các chunks
    separators: List[str]  = ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators=separators)
    text_splitted = text_splitter.split_text(processing_text(text))
    print(f"len document splitted: {len(text_splitted)}") 

    return text_splitted

def chunk_docs(pdf_file: str):

    #trích xuất văn bản từ file pdf
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load()
    print(f"len pages: {len(pages)}")
    for page in pages:
        page.page_content = processing_text(page.page_content)
    # chia nhỏ text thành các chunks
    separators: List[str]  = ["\n\n", "\n", " ", ""]
    char_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators=separators)
    docs_splitted = char_splitter.split_documents(pages)
    print(f"len document splitted: {len(docs_splitted)}") 

    return docs_splitted

def save_documents(path_pdf: str, directory: str):

    # tải file pdf từ đường dẫn
    docs = chunk_docs(path_pdf)
    # tạo dic nếu dic chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # # chuyển đổi data để có thê lưu dưới dạng json
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
    path_pdf = r"data\1506.02640v5.pdf"

    # tải file pdf từ đường dẫn
    save_documents(path_pdf=path_pdf, directory="data")

if __name__ == "__main__":
    main()