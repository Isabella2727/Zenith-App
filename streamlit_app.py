import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd

# Configure Gemini AI using the secret from secrets.toml
gemini_api_key = st.secrets["Gemini_API"]
genai.configure(api_key=gemini_api_key)

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to read PDF files
def read_pdf_files(folder_name):
    folder_path = os.path.join(script_dir, folder_name)
    pdf_contents = []
    if not os.path.exists(folder_path):
        st.error(f"The '{folder_name}' folder does not exist in {script_dir}. Please create it and add your PDF catalogs.")
        st.stop()
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in the '{folder_name}' folder. Please add your PDF catalogs.")
        return pdf_contents

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            pdf_contents.append({"filename": filename, "content": text})
    return pdf_contents

# Read PDF files from the 'catalogs' folder
catalog_folder = 'catalogs'
pdf_contents = read_pdf_files(catalog_folder)

# Function to get product recommendations using Gemini
def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    model = genai.GenerativeModel('gemini-pro')
    context = "\n".join([f"Catalog {i+1}: {content['content'][:500]}..." for i, content in enumerate(pdf_contents)])
    prompt = f"Based on the following catalogs:\n{context}\n\nRecommend a product for this request: {user_input}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.title('Smart Product Selection App')

if not pdf_contents:
    st.info("To use this app, please follow these steps:")
    st.markdown(f"""
    1. Ensure there's a folder named 'catalogs' in this directory: {script_dir}
    2. Add your PDF catalog files to the 'catalogs' folder.
    3. Restart the Streamlit app.
    """)
else:
    # User input
    user_input = st.text_input("Describe what you're looking for:")

    if user_input:
        # Get product recommendation
        recommendation = get_product_recommendation(user_input, pdf_contents)
        st.subheader("Recommended Product:")
        st.write(recommendation)

    # Display catalog information
    st.subheader("Available Catalogs")
    for pdf in pdf_contents:
        st.write(f"- {pdf['filename']}")

    # Option to view catalog contents
    if pdf_contents:
        selected_catalog = st.selectbox("Select a catalog to view:", [pdf['filename'] for pdf in pdf_contents])
        if selected_catalog:
            selected_content = next(pdf['content'] for pdf in pdf_contents if pdf['filename'] == selected_catalog)
            st.text_area("Catalog Content (first 1000 characters):", selected_content[:1000], height=200)

# Display the current working directory and script directory for debugging
st.sidebar.write(f"Current Working Directory: {os.getcwd()}")
st.sidebar.write(f"Script Directory: {script_dir}")