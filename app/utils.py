import streamlit as st
from functools import wraps


def trace_messages(func):
    """
    Декоратор: после вызова функции отображает состояние state["messages"]
    как "цепочку рассуждений".
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        state = kwargs.get("state") or (args[0] if args else None)
        if state and "messages" in state:
            st.subheader("Chain of Thought")
            for i, msg in enumerate(state["messages"]):
                st.markdown(f"**Шаг {i + 1}:** `{msg['role']}`")
                st.code(msg["content"], language="text")
                st.markdown("---")
        return result

    return wrapper
