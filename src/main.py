import streamlit as st
from langchain_ollama import ChatOllama


def setup_sidebar():
    with st.sidebar:
        st.radio("Choose a shipping method",  ("Standard (5-15 days)", "Express (2-5 days)"))

def setup_chat(llm):

    st.title("echo bot")
    # khởi tạo chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # hiển thị tin nhắn trò chuyện từ lịch sử khi chạy lại ứng dụng
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # đầu vào người dùng
    if prompt:=st.chat_input(placeholder="Nhập tin nhắn người dùng?"):
        # hiển thị tin nhắn người dùng trong chat message container
        with st.chat_message("user"):
            st.write(prompt)
        # thêm tin nhắn người dùng vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            response = llm.stream(prompt)
            output = st.write_stream(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})


def main():
    setup_sidebar()

    llm = ChatOllama(
        model="llama3.1",  # hoặc model khác tùy chọn
        temperature=0,
        streaming=True
    )

    setup_chat(llm=llm)


if __name__ == "__main__":
    main()

