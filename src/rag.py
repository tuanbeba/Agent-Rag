from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools.retriever import create_retriever_tool
from database_vector import connect_Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from langgraph.graph import MessagesState, StateGraph
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def retrieve(embedd_model: str):
    """Retrieve information related to a query."""
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

    llm = ChatOpenAI(
            model = 'gpt-4o-mini',
            temperature=0.1,
            streaming=True
    )
    #tạo tool retriever
    retrieve_tool =create_retriever_tool(retriever= retrieve(embedd_model=embedd_model), 
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

def create_session_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
        
        }
    )
    
    return response

def main():
    # test function
    chat_history = []
    agent_executor = create_agent1(chat_model="gpt-4o-mini",embedd_model="nomic-embed-text")
    while (user_input := input("user input: ")):
        if user_input.lower() == "quit":
            break
        respone = create_session_chat(agent_executor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        output = respone["output"]
        # chat_history.append(AIMessage(content=output))
        print(f"AI: {respone}")

if __name__ =="__main__":
    main()