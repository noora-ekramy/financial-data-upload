import streamlit as st
import pandas as pd
import os
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
from financial_model_api import ask_question

# Load environment and set up OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Page styling and header
st.markdown("""
    <style>
    html, body, .stApp {background-color: #041317; margin: 0; padding: 0;}
    .stApp {display: flex; flex-direction: column; min-height: 100vh;}
    .tagline {font-size: 20px; font-weight: bold; color: #a4ffff; text-align: center; margin-top: -10px;}
    .subtagline {font-size: 14px; color: #fcfcfc; text-align: center; padding: 20px; margin-bottom: 20px;}
    .header {font-size: 39px; font-weight: bold; color: #65daff; text-align: center; margin: -10px 0 10px 0;}
    </style>
    """, unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    st.image("youtiva-logo.png", width=100)
with col2:
    st.title("Youtiva")
st.markdown('<div class="tagline">Stand Out & Excel with Your Unique AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtagline">Tailored AI for your biz needs</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Financial Statements</div>', unsafe_allow_html=True)

# Ask how many years
num_years = st.number_input("How many years? (Enter a number)", min_value=1, step=1, value=1)

yearly_data = []
all_uploaded = True
missing_msg = ""
st.markdown("---")
for i in range(1, num_years + 1):
    st.subheader(f"Year {i}")
    # Year label input
    year_label = st.text_input(f"Enter label for Year {i} (e.g., 2021)", value=f"Year {i}", key=f"year_label_{i}")
    
    # Core file uploads
    income_file = st.file_uploader(f"Income Statement (CSV or PDF) for {year_label}", type=["csv", "pdf"], key=f"income_{i}")
    balance_file = st.file_uploader(f"Balance Sheet (CSV or PDF) for {year_label}", type=["csv", "pdf"], key=f"balance_{i}")
    cash_file = st.file_uploader(f"Cash Flow Statement (CSV or PDF) for {year_label}", type=["csv", "pdf"], key=f"cash_{i}")

    missing = []
    if not income_file:
        missing.append("Income Statement")
    if not balance_file:
        missing.append("Balance Sheet")
    if not cash_file:
        missing.append("Cash Flow Statement")
    
    # Extra files section
    extra_files = []
    num_extra = st.number_input(f"Number of extra files for {year_label}", min_value=0, value=0, step=1, key=f"num_extra_{i}")
    for j in range(1, num_extra + 1):
        extra_name = st.text_input(f"Extra File {j} Name", key=f"extra_name_{i}_{j}")
        extra_file = st.file_uploader(f"Upload file for '{extra_name}'", type=["txt", "pdf", "csv"], key=f"extra_{i}_{j}")
        if extra_file:
            # Process extra file based on type
            if extra_file.name.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(extra_file)
                extra_text = ""
                for page in pdf_reader.pages:
                    extra_text += page.extract_text() + "\n"
            elif extra_file.name.endswith(".csv"):
                df_extra = pd.read_csv(extra_file)
                extra_text = df_extra.to_string()
            else:
                extra_text = extra_file.read().decode("utf-8")
            extra_files.append({"name": extra_name, "text": extra_text})
    
    if missing:
        all_uploaded = False
        missing_msg += f"{year_label} missing: {', '.join(missing)}\n"
    else:
        if income_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(income_file)
            income_df = pd.DataFrame()  # Placeholder for PDF data processing
            for page in pdf_reader.pages:
                income_df = pd.concat([income_df, pd.read_csv(page.extract_text())], ignore_index=True)  # Example processing
        else:
            income_df = pd.read_csv(income_file)

        if balance_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(balance_file)
            balance_df = pd.DataFrame()  # Placeholder for PDF data processing
            for page in pdf_reader.pages:
                balance_df = pd.concat([balance_df, pd.read_csv(page.extract_text())], ignore_index=True)  # Example processing
        else:
            balance_df = pd.read_csv(balance_file)

        if cash_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(cash_file)
            cash_df = pd.DataFrame()  # Placeholder for PDF data processing
            for page in pdf_reader.pages:
                cash_df = pd.concat([cash_df, pd.read_csv(page.extract_text())], ignore_index=True)  # Example processing
        else:
            cash_df = pd.read_csv(cash_file)

        yearly_data.append({
            "label": year_label,
            "income": income_df,
            "balance": balance_df,
            "cash": cash_df,
            "extra": extra_files
        })
    
    st.markdown("---")  # Horizontal rule after each year

# If everything is in place, combine all text data
if all_uploaded and yearly_data:
    combined_text = ""
    for data in yearly_data:
        combined_text += f"{data['label']}:\n"
        combined_text += f"Income Statement:\n{data['income'].to_string()}\n"
        combined_text += f"Balance Sheet:\n{data['balance'].to_string()}\n"
        combined_text += f"Cash Flow Statement:\n{data['cash'].to_string()}\n"
        if data["extra"]:
            for extra in data["extra"]:
                combined_text += f"Extra File - {extra['name']}:\n{extra['text']}\n"
        combined_text += "\n"
        
        st.subheader(f"{data['label']} - Income Statement")
        st.dataframe(data['income'], height=300)
        st.subheader(f"{data['label']} - Balance Sheet")
        st.dataframe(data['balance'], height=300)
        st.subheader(f"{data['label']} - Cash Flow Statement")
        st.dataframe(data['cash'], height=300)
    
    model_name = st.selectbox(
        "Choose a model:",
        ["gpt-4o", "Meta_Llama_3_8B_Instruct", "Meta_Llama_3dot3_70B_Instruct_Turbo",
         "Meta_Llama_3dot3_70B_Instruct", "Mistral_Small_24B_Instruct_2501"],
        index=0
    )
    user_question = st.text_area("Ask a question about these financials (all years):")
    if st.button("Analyze Data"):
        if user_question:
            answer = ask_question(user_question, combined_text, model_name)
            st.subheader("Analysis")
            st.write(answer)
        else:
            st.warning("Please enter a question before clicking 'Analyze Data'.")
else:
    if missing_msg:
        st.info(f"Please upload all three core financial statements before analyzing the data:\n{missing_msg}")
