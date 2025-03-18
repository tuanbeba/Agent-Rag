from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from setting import ChromaSetting
import re
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class LocalVectorStore:
    def __init__(self, is_local: bool, embedding_model: str):
        self.is_local = is_local
        if self.is_local:
            self.embedding_model = OllamaEmbeddings(model=embedding_model)
        else:
            self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        # self.k_retriever = k_retriever
        # self.chunk_size = chunk_size
        # self.chunk_overlap = chunk_overlap

    def _clean_text(self, text: str)-> str:
        # định nghĩa regex
        pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
        # xóa các ký tự không nằm trong pattern
        sub_text = re.findall(pattern, text)
        # join các chuỗi con thành một chuỗi duy nhất 
        text = ' '.join(sub_text)
        # xóa các ký tự khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def _parse_pdf(self, input_files):
        chunks = []
        for input_file in input_files:
        #trích xuất văn bản từ file pdf
            pdf_loader = PyPDFLoader(input_file)
            pages = pdf_loader.load()
            print(f"len pages: {len(pages)}")
            for page in pages:
                page.page_content = self._clean_text(page.page_content)
            # chia nhỏ text thành các chunks
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size = ChromaSetting().chunk_size,
                chunk_overlap = ChromaSetting().chunk_overlap,)
            chunk = char_splitter.split_documents(pages)
            chunks.extend(chunk)
        
        return  chunks
    
    
    def set_vectorstore(self, input_files):
        all_docs = self._parse_pdf(input_files=input_files)
        # khởi tạo kho lưu trữ vector với Chroma
        vector_store = Chroma(
        collection_name=ChromaSetting().collection_name, # tên collection
        embedding_function=self.embedding_model,
        persist_directory=ChromaSetting().chroma_path,  # nơi lưu data local
        )
        # xóa collection và tạo lại collection rỗng
        vector_store.reset_collection()
        vector_store.add_documents(documents=all_docs)
        print(f"save {len(all_docs)} vectors to disk")

    def get_vectorstore(self):
        # load vector store from disk
        vector_store = Chroma(collection_name=ChromaSetting().collection_name,
        embedding_function=self.embedding_model,
        persist_directory=ChromaSetting().chroma_path
        )
        print("connected chromaDB")

        return vector_store
