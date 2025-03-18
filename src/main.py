import streamlit as st
from dotenv import load_dotenv
from options_model import LocalChat,LocalEmbedding,OnlineChat,OnlineEmbedding
from vectorstore import LocalVectorStore
from agent import Agent
import tempfile
import os
import asyncio

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
    if "agent" not in st.session_state:
        st.session_state.agent = None

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
            # print("Delete history")
        st.subheader(body="Upload Document")
        uploaded_files=st.file_uploader(label="Upload Pdf documents",
                         type=["pdf"],
                         label_visibility="collapsed",
                         accept_multiple_files=True
                         )
        upload = st.button(label="Upload File")
        if uploaded_files != st.session_state.file_list and not upload:
            st.warning("Đã thay đổi tệp")
        elif uploaded_files != st.session_state.file_list and upload:
            st.session_state.file_list = uploaded_files  # Cập nhật danh sách file
            handle_files_input(uploaded_files)  # Gọi hàm xử lý file
            st.success("Files uploaded successfully!")
            st.session_state.chat_history = []

def handle_files_input(uploaded_files):

    temp_paths = []  # Danh sách đường dẫn tạm thời của các file
    try:
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())  # Ghi dữ liệu vào file
                temp_paths.append(tmp_file.name)  # Lưu đường dẫn file tạm vào danh sách

        with st.spinner("Vui lòng chờ trong giây lát"):
            vector_store = LocalVectorStore(st.session_state.is_local, st.session_state.embedding_model)
            vector_store.set_vectorstore(input_files=temp_paths)
            st.session_state.agent = Agent(st.session_state.is_local, st.session_state.chat_model, vector_store)

    except Exception as e:
                st.error(f"❌ lỗi {str(e)}")
    finally:
            # Xóa từng file tạm
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception as e:
                st.warning(f"Không thể xóa file tạm {path}: {e}")
        
async def handle_user_input():

    st.title("Asisstant ChatBot")
    # Kiểm tra nếu không có file nào thì làm mờ hộp chat
    is_disabled = not st.session_state.get("file_list", [])  # True nếu file_list rỗng

    # hiển thị tin nhắn trò chuyện từ lịch sử khi chạy lại ứng dụng
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Làm mờ nếu không có file
    if is_disabled:
        st.warning("Bạn cần tải ít nhất lên một tệp PDF trước khi chat!")
        st.stop()  # Dừng luồng thực thi để ngăn nhập liệu
    
    # đầu vào người dùng
    if prompt:=st.chat_input(placeholder="Nhập câu hỏi của bạn ở đây?"):
        # thêm tin nhắn người dùng vào lịch sử chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # hiển thị tin nhắn người dùng trong chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # hiển thị phản hồi của ai trong chat message container
        with st.chat_message("assistant"):
            #giữ chỗ cho AI messgage
            response_container = st.empty()
            response_text = ""
            # lấy phản hồi từ agent
            async for chunk in st.session_state.agent.run_agent(prompt, st.session_state.chat_history):
                response_text += chunk
                response_container.markdown(response_text)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            


async def main():

    initialize_ss_state()
    load_api_keys()
    setup_page_config()
    setup_sidebar()
    await handle_user_input()
    print("#######################")


if __name__ == "__main__":
    asyncio.run(main())

