from langchain_ollama import ChatOllama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from database_vector import connect_Chroma
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_retrieval(collection_name: str="collection_1", top_k: int=4):

    # kết nối tới vectordb
    vectorDB = connect_Chroma(collection_name= collection_name)
    # tạo retrival
    retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs ={"k":top_k})
    
    return retriever

def create_chain():

    # khởi tạo model chat
    llm = ChatOllama(
        model = 'llama3.1',  # hoặc model khác tùy chọn
        temperature=0.3,
        streaming=True
    )

    # tạo prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions base on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")

    ])

    # chain = prompt | llm
    # tạo chain thiết lập với prompt
    chain = create_stuff_documents_chain(llm =llm, prompt=prompt)
    #tạo retrieval (truy xuất từ DB)
    retrieval = create_retrieval()
    
    retrieval_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    #tạo retrieval truy xuất với chat_history
    history_retrieval = create_history_aware_retriever(
        llm=llm,
        retriever=retrieval,
        prompt= retrieval_prompt)
    
    # Tạo retrieval chain
    retrievel_chain = create_retrieval_chain(retriever=history_retrieval,
                                             combine_docs_chain=chain)

    return retrievel_chain

def create_session_chat(chain, user_input, chat_history):
    result = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
        
    })
    
    return result

def main():
    
    chain = create_chain()
    chat_history = []
    
    while True:

        user_input = input("Human: ")
        if user_input.lower() == "exit":
            break
        result = create_session_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["answer"]))

        print(f"Assistant:", result["answer"])

if __name__ =="__main__":
    main()