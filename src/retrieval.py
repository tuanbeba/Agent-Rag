from langchain_ollama import ChatOllama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from database_vector import connect_Chroma
from dotenv import load_dotenv
import os

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
    prompt = ChatPromptTemplate.from_template(
        """
        answer the user's question:
        Context:{context}
        Question:{input}
        """
    )

    # chain = prompt | llm
    # tạo chain thiết lập với prompt
    chain = create_stuff_documents_chain(llm =llm, prompt=prompt)
    #tạo retrieval
    retrieval = create_retrieval()

    retrievel_chain = create_retrieval_chain(retriever=retrieval,
                                             combine_docs_chain=chain)

    return retrievel_chain


def main():
    
    chain = create_chain()
    
    while True:

        user_input = input("Human: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({
        "input": user_input
        })

        print(f"Assistant: ", result["answer"])

if __name__ =="__main__":
    main()