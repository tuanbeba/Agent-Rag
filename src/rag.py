from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from database_vector import connect_Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from langgraph.graph import MessagesState, StateGraph
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@tool
def retrieve():
    """Retrieve information related to a query."""
    # kết nối tới vectordb
    vectorDB = connect_Chroma()
    # tạo retrival chroma
    chroma_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs ={"k":4})
    # tạo retriver BM25

    # tạo retrievel kết hợp BM25 và chroma
    
    return chroma_retriever

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def create_chain(chat_model: str, embedd_model: str):

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


    llm_with_tools = llm.bind_tools([retrieve(embedd_model)])

    return llm_with_tools
    # # tạo prompt template
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "Answer the user's questions base on the context: {context}"),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human","{input}")

    # ])

    # # tạo chain thiết lập với prompt
    # chain = create_stuff_documents_chain(llm =llm, prompt=prompt)
    # #tạo retrieval (truy xuất từ DB)
    # retrieval = create_retrieval()
    
    # retrieval_prompt = ChatPromptTemplate.from_messages([
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}"),
    #     ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    # ])
    # #tạo retrieval truy xuất với chat_history
    # history_retrieval = create_history_aware_retriever(
    #     llm=llm,
    #     retriever=retrieval,
    #     prompt= retrieval_prompt)
    
    # # Tạo retrieval chain
    # retrievel_chain = create_retrieval_chain(retriever=history_retrieval,
    #                                          combine_docs_chain=chain)

    # return retrievel_chain

def create_session_chat(chain, user_input, chat_history):
    response = chain.stream({
        "input": user_input,
        "chat_history": chat_history
        
    })
    
    return response

def main():
    
    retrie = retrieve()
    respone = retrie.invoke("what is yolo")
    for text in respone:
        print(text)
        print("#####################")

if __name__ =="__main__":
    main()