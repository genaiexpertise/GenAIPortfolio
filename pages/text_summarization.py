import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv, find_dotenv
import os


from dependencies import check_password


_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = st.secrets.APIKEY.OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")


def generate_response(txt):
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.run(docs)


st.set_page_config(
    page_title = "Writing Text Summarization"
)

st.logo(
    'assets/logo.png',
    link="https://genaiexpertise.com",
    icon_image="assets/icon.png",
)

if not check_password():
    st.stop()

st.title("Writing Text Summarization")

txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input)
        result.append(response)

if len(result):
    st.info(response)