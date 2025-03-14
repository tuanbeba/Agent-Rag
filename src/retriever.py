from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from vectorstore import LocalVectorStore
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
from dotenv import load_dotenv
from vectorstore import LocalVectorStore

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class Agent():
    def __init__(self, is_local: bool, chat_model: str, embedding_model: str, vector_store: LocalVectorStore):
        self.is_local = is_local
        if self.is_local:
            self.embedding_model = OllamaEmbeddings(model=embedding_model)
            self.chat_model = ChatOllama(model=chat_model)
        else:
            self.embedding_model = OpenAIEmbeddings(model=embedding_model)
            self.chat_model = ChatOpenAI(model=chat_model)
        self.vector_store = vector_store
        self.preamble="""
## Task &amp; Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.
## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
## Guidelines
You are an expert who answers the user's question.
You have access to a hybrid_search tool that will use your query to search through documents and find the relevant answer.
"""
        self.get_tools()
        self.build_agent()
        
    def get_tools(self,):
        @tool
        def hybrid_search(query: str):
            "Uses the query to search through a list of documents and return the most relevant documents."
            vectordb = self.vector_store.get_vectorstore()
            # tạo retrival chroma
            chroma_retriever = vectordb.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 10}
            )
            # tạo retriver BM25
            all_ids = vectordb._collection.get()["ids"]
            if not all_ids:
                raise ValueError(" Not found documents in collection")
            documents = vectordb.get_by_ids(all_ids)
            bm25_retriever = BM25Retriever.from_documents(documents)
            # tạo retrievel kết hợp BM25 và similarity
            ensemble_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
            docs_retriever = ensemble_retriever.invoke(query)

            return docs_retriever
        hybrid_search.name ="hybridsearch"
        hybrid_search.description = "Uses the query to search through a list of documents and return the most relevant documents."
        class SearchInput(BaseModel):
            query: str = Field(description="The users query")
        hybrid_search.args_schema = SearchInput

        self.tools = [hybrid_search]
    

    def build_agent(self):
        system = """You are an expert at AI. Your name is AlphaAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        agent = create_tool_calling_agent(llm=self.chat_model, tools=self.tools, prompt=prompt)
        self.agent_excutor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run_agent(self, query, history):
        respone = self.agent_excutor.invoke(
            {
                "input": query,
                "chat_history": history
            }
        )

        return respone
    
def main():
    pdf1 = r"data\1506.02640v5.pdf"
    pdf2 = r"data\2312.16862v3.pdf"
    pdfs = [pdf1,pdf2]
    vectorstore = LocalVectorStore(is_local=False, embedding_model="text-embedding-3-small")
    vectorstore.set_vectorstore(pdfs)
    agent1 = Agent(is_local=False, chat_model="gpt-4o-mini", embedding_model="text-embedding-3-small", vector_store=vectorstore)
    chat_history = []
    while (user_input := input("user input: ")):
        if user_input.lower() == "quit":
            break
        respone = agent1.run_agent(user_input,chat_history)
        # chat_history.append(HumanMessage(content=user_input))
        output = respone["output"]
        # chat_history.append(AIMessage(content=output))
        print(f"AI: {respone}")

if __name__ == "__main__":
    main()