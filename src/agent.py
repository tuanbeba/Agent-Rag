from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from vectorstore import LocalVectorStore
from langchain.retrievers import EnsembleRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
from dotenv import load_dotenv
import asyncio
from setting import RetrieverSetting
from FlagEmbedding import FlagReranker
import torch

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class RAG_pipeline():
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store
        self.similarity_top_k = RetrieverSetting().similarity_top_k
        self.bm25_top_k = RetrieverSetting().bm25_top_k
        self.rerank_top_k = RetrieverSetting().rerank_top_k
        self.retriever_weight = RetrieverSetting().retriever_weight
        self.device=["cuda"] if torch.cuda.is_available() else ["cpu"]
        self.rerank_model = FlagReranker(RetrieverSetting().rerank_model, use_fp16=True, devices=self.device)
        self.use_rerank = RetrieverSetting().use_rerank


    def rerank(self, query, list_docs, top_k):

        pairs = [[query, doc.page_content] for doc in list_docs]
        scores = self.rerank_model.compute_score(sentence_pairs=pairs, normalize = True)
        top_docs = sorted(zip(scores, list_docs), key=lambda x: x[0], reverse=True)[:top_k]
        top_chunks_rerank = [doc for _, doc in top_docs]
        print(f"Lenght docs reranked : {len(top_chunks_rerank)}")

        return top_chunks_rerank
        
    def hybrid_search(self, query):

        vectordb = self.vector_store.get_vectorstore()
        # tạo retrival chroma
        chroma_retriever = vectordb.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": RetrieverSetting().similarity_top_k}
        )
        # tạo retriver BM25
        all_ids = vectordb._collection.get()["ids"]
        if not all_ids:
            raise ValueError(" Not found documents in collection")
        documents = vectordb.get_by_ids(all_ids)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = RetrieverSetting().bm25_top_k

        if self.use_rerank:
            chroma_result = chroma_retriever.invoke(query)
            bm25_result = bm25_retriever.invoke(query)
            page_content = set()
            docs_retriever = []
            for doc in chroma_result:
                if doc.page_content not in page_content:
                    page_content.add(doc.page_content)
                    docs_retriever.append(doc)
            for doc in bm25_result:
                if doc.page_content not in page_content:
                    page_content.add(doc.page_content)
                    docs_retriever.append(doc)
            if not docs_retriever:
                raise ValueError("No documents retrieved for reranking.")
            reranked_docs = self.rerank(query=query,list_docs=docs_retriever, top_k= self.rerank_top_k)

            return reranked_docs
        else:
            emsemble_retreiever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever],
                weights=self.retriever_weight
            )
            return emsemble_retreiever.invoke(query) 
    
        
    def process_query(self, query):


        pass


class Agent():

    def __init__(self, is_local: bool, chat_model: str, vector_store: LocalVectorStore):
        self.is_local = is_local
        if self.is_local:
            self.chat_model = ChatOllama(model=chat_model)
        else:
            self.chat_model = ChatOpenAI(model=chat_model, streaming=True)
        self.vector_store = vector_store
        self.rag_retreiver = RAG_pipeline(self.vector_store)
        self.system="""
            ## Task &amp; Context
            You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.
            ## Style Guide
            Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
            ## Guidelines
            You are an expert who answers the user's question.
            You have access to a search tool that will use your query to search through documents and find the relevant answer.
            """
        self.get_tools()
        self.build_agent()
        
    def get_tools(self,):
        @tool
        def search(query: str):
            "Uses the query to search through a list of documents and return the most relevant documents."
            # tạo retrival chroma
            final_docs = self.rag_retreiver.hybrid_search(query=query)
            if final_docs is None:
                return "No documents found."

            content_docs = ""
            for doc in final_docs:
                content_docs += doc.page_content + "\n"

            return content_docs
        search.name ="hybridsearch"
        search.description = "Uses the query to search through a list of documents and return the most relevant documents."
        class SearchInput(BaseModel):
            query: str = Field(description="The users query")
        search.args_schema = SearchInput

        self.tools = [search]
    
    def build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        agent = create_tool_calling_agent(llm=self.chat_model, tools=self.tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def run_agent(self, query, history):
        events = []
        async for event in self.agent_executor.astream_events(
            {
                "input": query,
                "chat_history": history
            },
            version="v1"
        ):
                events.append(event)
                if event["event"] == "on_chat_model_stream":
                    yield event["data"]["chunk"].content
        event_types = {event["event"] for event in events}
        print("Unique event types:", event_types)
    

async def main():
    pdf1 = r"data\1506.02640v5.pdf"
    pdf2 = r"data\2312.16862v3.pdf"
    pdfs = [pdf1,pdf2]
    vectorstore = LocalVectorStore(is_local=False, embedding_model="text-embedding-3-small")
    vectorstore.set_vectorstore(pdfs)
    agent1 = Agent(is_local=False, chat_model="gpt-4o-mini", vector_store=vectorstore)
    chat_history = []
    while (user_input := input("user input: ")):
        if user_input.lower() == "quit":
            break

        async for response in agent1.run_agent(user_input, chat_history):  # Dùng async for
            print(response, end="", flush=True)
        print()
if __name__ == "__main__":

    asyncio.run(main())