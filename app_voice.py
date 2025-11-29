import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
import base64
from io import BytesIO

load_dotenv()

def speak(text):
    audio = gTTS(text=text, lang="hi")
    mp3 = BytesIO()
    audio.write_to_fp(mp3)
    mp3.seek(0)
    b64 = base64.b64encode(mp3.read()).decode()
    st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")

def transcribe(audio_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="hi-IN")
    except:
        return None

@st.cache_resource
def load_chain():
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Continue the conversation with memory and reply in the same language the user speaks."),
            ("human", "{question}")
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    session_histories = {}
    def get_history(session_id):
        if session_id not in session_histories:
            session_histories[session_id] = ChatMessageHistory()
        return session_histories[session_id]

    return RunnableWithMessageHistory(
        chain, get_history,
        input_messages_key="question",
        history_messages_key="history"
    )

st.set_page_config(page_title="Custom GPT chatbot", page_icon="🎤")
st.title("🎤 Custom GPT chatbot")

chain = load_chain()
if "session_id" not in st.session_state:
    st.session_state.session_id = "user123"

user_input = st.chat_input("Type your message...")

# 🎤 MIC ICON INPUT
st.write("### 🎤 Click mic & speak")
audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#cccccc", recording_color="#ff4d4d")

if audio_bytes:
    with st.spinner("⏳ Converting speech to text..."):
        text = transcribe(audio_bytes)
    if text:
        user_input = text
        st.success(f"🗣 You said: {text}")
    else:
        st.error("⚠ Voice samajh nahi aayi. Please try again.")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            st.write(response)
            speak(response)
