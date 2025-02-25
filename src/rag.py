from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from database_vector import connect_Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def retriever(embedd_model: str):
    # kết nối tới vectordb
    vectorDB = connect_Chroma(embed_model=embedd_model)
    # tạo retrival chroma
    chroma_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs ={"k":4})
    # tạo retriver BM25

    # tạo retrievel kết hợp BM25 và chroma
    
    return chroma_retriever

def create_agent1(chat_model: str, embedd_model: str) -> AgentExecutor:

    # khởi tạo model chat
    if chat_model == "llama3.1":
        llm = ChatOllama(
            model = 'llama3.1',
            temperature=0.1,
            streaming=True
        )
    else:
        llm = ChatOpenAI(
            model = 'gpt-4o-mini',
            temperature=0.1,
            streaming=True
        )

    
    #tạo tool retriever
    retrieve_tool =create_retriever_tool(retriever= retriever(embedd_model=embedd_model), 
    name="file_search", 
    description="Search for information within the uploaded files. For any question related to YOLO, you must use this tool to get the answer!")
    tools = [retrieve_tool]

    # tạo prompt cho agent
    prompt = ChatPromptTemplate.from_messages(
        [("system",
            "You are a helpful assistant named AlphaAI. "
            "When a user's query is about the content of uploaded files, "
            "you must always use the 'file_search' tool to retrieve the information instead of relying solely on your pre-trained knowledge."
),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')]
    )

    agent = create_openai_tools_agent(llm=llm, tools= tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return agent_executor



@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # kết nối tới vectordb
    vector_store = connect_Chroma()
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langgraph.checkpoint.memory import MemorySaver

def create_agent2(chat_model: str):
    # khởi tạo model chat
    if chat_model == "llama3.1":
        llm = ChatOllama(
            model = 'llama3.1',
            temperature=0.1,
            streaming=True
        )
    else:
        llm = ChatOpenAI(
            model = 'gpt-4o-mini',
            temperature=0.1,
            streaming=True
        )
    system_message = "You are a helpful assistant. Respond only in Vietnamese."
    agent_executor = create_react_agent(llm, [retrieve], state_modifier = system_message)

    return agent_executor

def create_session_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
        
        }
    )
    
    return response


def main():

    # test retriever
    # retriever = retrieve(embedd_model="nomic-embed-text")
    # respone = retriever.invoke(input="what is yolo")
    # for doc in respone:
    #     print(doc)
    #     print("###########")
    # test agent1
    # chat_history = []
    # agent_executor = create_agent1(chat_model="gpt-4o-mini",embedd_model="nomic-embed-text")
    # while (user_input := input("user input: ")):
    #     if user_input.lower() == "quit":
    #         break
    #     respone = create_session_chat(agent_executor, user_input, chat_history)
    #     chat_history.append(HumanMessage(content=user_input))
    #     output = respone["output"]
    #     chat_history.append(AIMessage(content=output))
    #     print(f"AI: {respone}")
    # test agent 2
    agent_executor = create_agent2(chat_model="gpt-4o-mini")
    while (user_input := input("user input: ")):
        if user_input.lower() == "quit":
            break
        respone = agent_executor.invoke({"messages": [("human",user_input)]})
        print(f"AI: {respone}")


if __name__ =="__main__":
    main()