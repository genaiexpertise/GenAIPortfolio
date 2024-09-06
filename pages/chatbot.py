import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain_core.messages import HumanMessage

from dependencies import check_password

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = st.secrets.APIKEY.OPENAI_API_KEY
openai_api_key = os.environ["OPENAI_API_KEY"]

chatbot = ChatOpenAI(model="gpt-3.5-turbo")

chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history
)

st.set_page_config(page_title="AI Chatbot")

if not check_password():
    st.stop()

st.title("Chatbot")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "hello, how can I help you today?"}    
    ]

def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)  # Add response to message history


for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input(
    "Your question",
):  # Prompt for user input and save to chat history
    # TODO: hacky
    if "has_rerun" in st.session_state.keys() and st.session_state.has_rerun:
        # if this is true, skip the user input
        st.session_state.has_rerun = False
    else:
        add_to_message_history("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    session_id =  st.session_state.keys()
                    response = chatbot_with_message_history.invoke(
                        [HumanMessage(content=prompt)],
                        config={"configurable": {"session_id": "001"}},
                    )
                    st.write(response.content)
                    add_to_message_history("assistant", str(response.content))

        else:
            pass

  
        st.session_state.has_rerun = True
        st.rerun()

else:
    # TODO: set has_rerun to False
    st.session_state.has_rerun = False

