import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from options_model import LocalChat,LocalEmbedding,OnlineChat,OnlineEmbedding
from vectorstore import LocalVectorStore
import tempfile
import os

def load_api_keys():
    load_dotenv()

def initialize_ss_state():
    if "is_local" not in st.session_state:
        st.session_state.is_local = None
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = None
    if "file_list" not in st.session_state:
        st.session_state.file_list = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def setup_page_config():
    st.set_page_config(
        page_title="ChatBot_PDF",  
        page_icon="💬",
        
    )


def setup_sidebar():
    with st.sidebar:
        st.title("Chatbot Setting")
        
        # chọn model embed
        environment = st.selectbox(label="Choosing type", 
                                options=("Online", "Local"),
                                label_visibility="collapsed")
        st.session_state.is_local = False if environment == "Online" else True
        chat_models = OnlineChat.keys() if environment == "Online" else LocalChat.keys()
        embedding_models = OnlineEmbedding.keys() if environment == "Online" else LocalEmbedding.keys()


        # Chọn mô hình chat và mô hình embedding
        selected_chat_model = st.selectbox("Choose Chat Model", options=chat_models)
        st.session_state.chat_model = selected_chat_model
        selected_embedding_model = st.selectbox("Choose Embedding Model", options=embedding_models)
        st.session_state.embedding_model = selected_embedding_model

        # xóa lịch sử chat
        if st.button(label="Clear history"):
            st.session_state.chat_history = []
            print("Delete history")
        uploaded_files=st.file_uploader(label="Upload Pdf documents",
                         type=["pdf"],
                         accept_multiple_files=True
                         )
        st.session_state.file_list = uploaded_files
        if st.button(label="Upload"):
            handle_files_input(uploaded_files)

def handle_files_input(uploaded_files):
    if len(uploaded_files) == 0:
        st.warning("Bạn cần tải ít nhất một file PDF trước khi chat.")
        st.stop()  # Lệnh này sẽ dừng các phần code bên dưới, ẩn luôn giao diện chat

    temp_paths = []  # Danh sách đường dẫn tạm thời của các file
    try:
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())  # Ghi dữ liệu vào file
                temp_paths.append(tmp_file.name)  # Lưu đường dẫn file tạm vào danh sách

        with st.spinner("Vui lòng chờ trong giây lát"):
            vector_store = LocalVectorStore(st.session_state.is_local, st.session_state.embedding_model)
            vector_store.set_vectorstore(input_files=temp_paths)
            st.session_state.vector_store = vector_store

    except Exception as e:
                st.error(f"❌ lỗi {str(e)}")
    finally:
            # Xóa từng file tạm
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception as e:
                st.warning(f"Không thể xóa file tạm {path}: {e}")
        


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

    initialize_ss_state()
    load_api_keys()
    setup_page_config()
    setup_sidebar()
    # display_chat_history()
    # handle_user_input(agent_rag)
    print("#######################")


if __name__ == "__main__":
    main()

