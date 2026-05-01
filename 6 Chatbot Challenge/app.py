import streamlit as st

from llm import get_ai_message, service_init

st.set_page_config(page_title="Docidian", page_icon="📄")

if "initialized" not in st.session_state:
    with st.spinner("시스템을 초기화하고 있습니다..."):
        service_init()
        st.session_state["initialized"] = True

st.title("📄 Docidian")
st.caption("모아둔 문서에서 지식을 찾아보세요!")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="문서에 있을 것 같은 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
