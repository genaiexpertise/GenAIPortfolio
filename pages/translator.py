from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st

from dependencies import check_password

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = st.secrets.APIKEY.OPENAI_API_KEY

openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()


st.set_page_config(page_title="AI Translator")

st.logo(
    'assets/logo.png',
    link="https://genaiexpertise.com",
    icon_image="assets/icon.png",
)

if not check_password():
    st.stop()

st.title("Language Translator")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Translate text into another language with this app.")
    
with col2:
    st.write("Contact with [GenAIExpertise](https://genaiexpertise.com) to build your AI Projects")


col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area("Enter text to translate")

with col2:
    st.write("Select the language you want to translate to.")
    target_lang = st.selectbox("Language", ["Yoruba","Hausa", "Igbo"])


if st.button("Translate"):
    if text_input:
        system_template = "Translate the following into {target_lang}:"
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('user', '{text}')
        ])
        chain = prompt_template | llm | parser
        response = chain.invoke({
                "target_lang": target_lang,
                "text": text_input
                }
            )
        st.warning(response)
  

