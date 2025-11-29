import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load API Keys

@st.cache_resource(show_spinner=False)
def load_chain():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the question asked"),
            ("user", "Question:{question}")
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain



# Streamlit UI

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Interactive AI Chatbot (Groq + Llama 3.1)")
st.write("Ask anything and get instant answers!")

# Load chain only once
try:
    chain = load_chain()
except Exception as e:
    st.error(f"Error while loading model: {e}")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"question": user_input})
            except Exception as e:
                response = f"❌ Error: {e}"
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
