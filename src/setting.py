from pydantic import BaseModel, Field
from typing import List
class ChromaSetting(BaseModel):
    collection_name: str = Field(
        default="test_collection", description="name collection embedding documents"
    )
    chroma_path: str = Field(
        default="./chroma_test", description="path folder"
    )
    chunk_size: int = Field(
        default=1000, description="chunk size documents"
    )
    chunk_overlap: int = Field(
        default=100, description="chunk overlap documents"
    )
class RetrieverSetting(BaseModel):
    similarity_top_k: int = Field(
        default=10, description="top k documents search by similarity search"
    )
    bm25_top_k: int =  Field(
        default=10, description="top k documents search by similarity search"
    )
    retriever_weight: List[float] = Field(
        default=[0.7, 0.3], description="weights for retriever"
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="Rerank model"
    )
    rerank_top_k: int = Field(
        default=5, description="top k documents for rerank"
    )
    use_rerank: bool = Field(
        default=True, description="using reranking or not"
    )