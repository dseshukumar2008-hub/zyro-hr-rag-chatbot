import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Help Desk")

question = st.chat_input("Ask an HR question...")

if question:
    st.chat_message("user").write(question)

    st.chat_message("assistant").write(
        "Streamlit deployment is working successfully."
    )
