import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model_name, temperature, max_tokens):
    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer


# Apply a custom CSS style to the app
st.markdown("""
    <style>
        .main {
            background-color: #f0f0f5;
            padding: 20px;
            border-radius: 10px;
            max-width: 800px;
            margin: 0 auto;
        }
        .stTextInput input {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        h1 {
            color: #333333;
            text-align: center;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        h2 {
            color: #4CAF50;
            text-align: center;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar for model selection and parameters
with st.sidebar:
    st.header("Model and Parameters")
    llm = st.selectbox("Select Open Source model", ["mistral"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")
