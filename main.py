import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
from financial_model_api import *

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=api_key)

# Streamlit UI
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #041317; 
        height: 100%;
        margin: 0;
        padding: 0;
    }
    .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .tagline {
        font-size: 20px;
        font-weight: bold;
        color: #a4ffff;
        text-align: center;
        margin-top: -10px;
    }
    .subtagline {
        font-size: 14px;
        color: #fcfcfc;
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    .header {
        font-size: 39px;
        font-weight: bold;
        color: #65daff;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 5]) 

with col1:
    st.image("youtiva-logo.png", width=100)

with col2:
    st.title("Youtiva")

st.markdown('<div class="tagline">Stand Out & Excel with Your Unique AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtagline">Empowering businesses with tailored AI solutions to streamline operations, boost efficiency, and sustain competitive advantage</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Financial Statements</div>', unsafe_allow_html=True)



# Initialize session state if not already set
if "financials_loaded" not in st.session_state:
    st.session_state.financials_loaded = False
    st.session_state.income_df = None
    st.session_state.balance_df = None
    st.session_state.cash_df = None
    st.session_state.all_files_uploaded = False
    st.session_state.financials_text = ""

# File uploaders for CSV files
st.subheader("Upload Financial Statements")
income_file = st.file_uploader("Upload Income Statement (CSV)", type=["csv"], key="income")
balance_file = st.file_uploader("Upload Balance Sheet (CSV)", type=["csv"], key="balance")
cash_file = st.file_uploader("Upload Cash Flow Statement (CSV)", type=["csv"], key="cash")

# Check if all files are uploaded
if income_file and balance_file and cash_file:
    st.session_state.financials_loaded = True
    st.session_state.income_df = pd.read_csv(income_file)
    st.session_state.balance_df = pd.read_csv(balance_file)
    st.session_state.cash_df = pd.read_csv(cash_file)
    st.session_state.all_files_uploaded = True
    
    # Convert financial data to text for AI analysis
    st.session_state.financials_text = (
        f"Income Statement:\n{st.session_state.income_df.to_string()}\n\n"
        f"Balance Sheet:\n{st.session_state.balance_df.to_string()}\n\n"
        f"Cash Flow Statement:\n{st.session_state.cash_df.to_string()}"
    )

# Display uploaded data if all files are present
if st.session_state.financials_loaded:
    st.subheader("Income Statement")
    st.dataframe(st.session_state.income_df, height=300)

    st.subheader("Balance Sheet")
    st.dataframe(st.session_state.balance_df, height=300)

    st.subheader("Cash Flow Statement")
    st.dataframe(st.session_state.cash_df, height=300)

# Model selection dropdown
model_name = st.selectbox(
    "Choose a model:",
    [
        "gpt-4o", 
        "Meta_Llama_3_8B_Instruct", 
        "Meta_Llama_3dot3_70B_Instruct_Turbo", 
        "Meta_Llama_3dot3_70B_Instruct", 
        "Mistral_Small_24B_Instruct_2501"
    ],
    index=0  # Default to gpt-4o
)
# User input for questions
if st.session_state.all_files_uploaded:
    user_question = st.text_area("Ask a question about this financial data:")
    
    # Button to analyze data
    if st.button("Analyze Data"):
        if user_question:
            answer = ask_question(user_question, st.session_state.financials_text, model_name)
            st.subheader("Analysis")
            st.write(answer)
        else:
            st.warning("Please enter a question before clicking 'Analyze Data'.")
else:
    st.info("Please upload all three financial statements before analyzing the data.")
