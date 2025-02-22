import streamlit as st
from database_vector import create_ChromaDB
from chain_retrieval import create_chain
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

def load_api_keys():
    load_dotenv()

def setup_page_config():
    st.set_page_config(
        page_title="ChatBot_PDF",  
        page_icon="💬",
        
    )

def handle_file(embedd_model):
    st.header("Tải tệp PDF ")
    pdf_file = st.file_uploader(label="Upload pdf here", type= ["pdf"])

    if "bool_db" not in st.session_state:
        st.session_state.bool_db = False
    
    if pdf_file is not None:
        #lấy tên file hiện tại
        current_filename = pdf_file.name     
        # kiểm tra xem có thay đổi file không
        if "upload_file" in st.session_state:
            if current_filename != st.session_state.upload_file:
                st.session_state.bool_db = False
        else :
            st.session_state.upload_file = current_filename

        if st.session_state.bool_db == False:
            with st.spinner("Vui lòng chờ trong giây lát"):
                try:

                    vector_store = create_ChromaDB(pdf_file=pdf_file, embed_model=embedd_model)
                    st.session_state.bool_db = True
                    st.success("Tạo vectorDB thành công.")
                except Exception as e:
                    st.error(f" lỗi {str(e)}")
        # pass
        
    else:
        st.session_state.bool_db = False
        st.warning("Bạn cần tải ít nhất một file PDF trước khi chat.")
        st.stop()  # Lệnh này sẽ dừng các phần code bên dưới, ẩn luôn giao diện chat
    


def setup_sidebar():
    with st.sidebar:
        st.title("Chatbot Setting")
        
        # chọn model embed
        embedd_model = st.sidebar.selectbox(label="Lựa chọn model embedding", 
                                            options=["nomic-embed-text", "text-embedding-3-large"])
        if embedd_model == "llama3.1":
            print("Model chat: ", embedd_model)
        else: # text-embedding-3-large
            print("model embedding: ", embedd_model)
        # xóa lịch sử chat
        if st.button(label="Clear history"):
            st.session_state.message = []
            print("Delete history")
           
        # chọn model chat
        chat_model = st.sidebar.selectbox(label="Lựa chọn model chat", options=["llama3.1", "gpt-4o-mini"])
        if chat_model == "llama3.1":
            chain = create_chain(chat_model=chat_model, embedd_model=embedd_model)
            print("Model chat: ",chat_model)
        else: # gpt4o-mini
            chain = create_chain(chat_model=chat_model, embedd_model=embedd_model)
            print("model chat: ", chat_model)

    return embedd_model,chain


def display_chat_history():
    st.title("Asisstant ChatBot")
    # khởi tạo chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # hiển thị tin nhắn trò chuyện từ lịch sử khi chạy lại ứng dụng
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_input(llm_with_tools):

    # đầu vào người dùng
    if prompt:=st.chat_input(placeholder="Nhập câu hỏi của bạn ở đây?"):
        # thêm tin nhắn người dùng vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        # hiển thị tin nhắn người dùng trong chat message container
        with st.chat_message("user"):
            st.write(prompt)
        
        # hiển thị phản hồi của ai trong chat message container
        with st.chat_message("assistant"):
            # lấy lịch sử chat lưu trong session_state
            chat_history = []
            for message in st.session_state.messages[:-1]:
                chat_history.append({"role": message["role"], "content": message["content"]})
            # lấy phản hồi từ chain
            # response = chain.invoke({
            #     "input": prompt,
            #     "chat_history": chat_history    
            #     }
            # )
            reponse = llm_with_tools.invoke([HumanMessage(content=prompt)])
            print("chat history: \n", chat_history)
            st.write(reponse)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})
        


def main():

    load_api_keys()
    setup_page_config()
    embedd_model, llm_with_tool = setup_sidebar()
    handle_file(embedd_model)
    display_chat_history()
    handle_user_input(llm_with_tool)
    print("#######################")


if __name__ == "__main__":
    main()

