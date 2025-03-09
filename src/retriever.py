from vectorstore import LocalVectorStore
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever

class Retriever():
    def __init__(self, ):
        pass
        
    def hybrid_retriever(self,):

        vectordb = LocalVectorStore.get_vectorstore()
        # tạo retrival chroma
        chroma_retriever = vectordb.similarity_search(k=10)
        # tạo retriver BM25
        all_ids = vectordb._collection.get()["ids"]
        documents = vectordb.get_by_ids(all_ids)
        bm25_retriever = BM25Retriever.from_documents(documents)
        # tạo retrievel kết hợp BM25 và similarity
        ensemble_retriever = EnsembleRetriever(
             retrievers=[chroma_retriever, bm25_retriever],
             weights=[0.7, 0.3]
        )

        
