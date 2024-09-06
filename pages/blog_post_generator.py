import streamlit as st
from langchain_openai import OpenAI
from langchain import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import os

from dependencies import check_password

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = st.secrets.APIKEY.OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(
    page_title = "Blog Post Generator"
)

st.logo(
    'assets/logo.png',
    link="https://genaiexpertise.com",
    icon_image="assets/icon.png",
)
if not check_password():
    st.stop()

st.title("Blog Post Generator")



def generate_response(topic):
    llm = OpenAI(openai_api_key=openai_api_key)
    template = """
    As experienced startup and generative AI Engineers,
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """
    prompt = PromptTemplate(
        input_variables = ["topic"],
        template = template
    )
    query = prompt.format(topic=topic)
    response = llm(query, max_tokens=2048)
    return st.write(response)

topic_text = st.text_input("Enter topic: ")

if not openai_api_key.startswith("sk-"):
    st.warning("openai_api_key not found. Please add it to your environment variables.")
if openai_api_key.startswith("sk-") and topic_text:
    generate_response(topic_text)
        