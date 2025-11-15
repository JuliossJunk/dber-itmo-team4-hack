import streamlit as st
from orchestrator import graph
from langchain_core.messages import HumanMessage
from uuid import uuid4

st.set_page_config(page_title="TEAM4_Chat", layout="wide")

gradient_css = """
<style>
    .stApp {
        background: linear-gradient(to bottom left, #e9480d, #FFFFFF);
    }
</style>
"""

st.markdown(gradient_css, unsafe_allow_html=True)

title_html = "<h1 style='color:black;'>TEAM4_Chat</h1>"
st.markdown(title_html, unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "state" not in st.session_state:
    st.session_state.state = {
        "query": "",
        "sub_queries": [],
        "data": {},
        "verified_facts": {},
        "messages": [],
        "final_answer": ""
    }

left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("Chain of Thought (real-time)")
    chain_placeholder = st.empty()


def render_chain_of_thought():
    """Рисуем CoT из реального состояния графа"""
    messages = st.session_state.state.get("messages", [])
    with chain_placeholder.container():
        for i, msg in enumerate(messages):
            st.markdown(f"### Шаг {i+1} • {msg.__class__.__name__}")
            st.code(msg.content)


with left_col:

    user_query = st.chat_input("Введите запрос...", key="chat_input")

    if user_query and st.session_state.get("last_handled") != user_query:
        st.session_state.last_handled = user_query
        st.session_state.state["messages"].append(HumanMessage(content=user_query))
        st.session_state.state["query"] = user_query

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        for update in graph.stream(st.session_state.state, config=config):

            if isinstance(update, dict) and "state" in update:
                st.session_state.state = update["state"]
            render_chain_of_thought()

    st.subheader("Диалог")

    for msg in st.session_state.state.get("messages", []):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if st.session_state.state.get("final_answer", "Что еще хотите узнать?"):
        st.subheader("Финальный ответ")
        st.markdown(st.session_state.state["final_answer"])
