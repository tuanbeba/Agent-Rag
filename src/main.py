import streamlit as st
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

def load_api_keys():

    load_dotenv()

def setup_page_config():
    st.set_page_config(
        page_title="ChatBot",  
        page_icon="💬", 
    )

def setup_sidebar():
    with st.sidebar:
        st.title("Chatbot Setting")
        select_model = st.sidebar.selectbox(label="Lựa chọn model", options=["llama3.1-7B", "gpt-4o-mini"])
        if select_model == "llama3.1-7B":
            pass
        else:
            pass
        if st.button(label="Clear history"):
            st.session_state.message = []
        st.file_uploader(label="Upload a file")

def display_chat_history():
    st.title("Asisstant ChatBot")
    # khởi tạo chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # hiển thị tin nhắn trò chuyện từ lịch sử khi chạy lại ứng dụng
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def handle_user_input(llm):
    # đầu vào người dùng
    if prompt:=st.chat_input(placeholder="Nhập tin nhắn của bạn ở đây?"):
        # thêm tin nhắn người dùng vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        # hiển thị tin nhắn người dùng trong chat message container
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = llm.stream(prompt)
            output = st.write_stream(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})


def main():
    llm = ChatOllama(
        model="llama3.1",  # hoặc model khác tùy chọn
        temperature=0,
        streaming=True
    )
    load_api_keys()
    setup_page_config()
    setup_sidebar()
    display_chat_history()
    handle_user_input(llm)


if __name__ == "__main__":
    main()

