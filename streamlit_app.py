import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import re

# Configure Gemini AI using the secret from secrets.toml
gemini_api_key = st.secrets["Gemini_API"]
genai.configure(api_key=gemini_api_key)

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
                text += page.extract_text() + "\n"
            cleaned_text = clean_text(text)
            pdf_contents.append({"filename": filename, "content": cleaned_text})
    return pdf_contents

# Read PDF files from the 'catalogs' folder
catalog_folder = 'catalogs'
pdf_contents = read_pdf_files(catalog_folder)

# Function to get product recommendations using Gemini
def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    model = genai.GenerativeModel('gemini-pro')
    context = "\n".join([f"Catalog {i+1} ({content['filename']}): {content['content'][:1000]}..." for i, content in enumerate(pdf_contents)])
    prompt = f"""Based on the following catalog excerpts, recommend a product for this request: "{user_input}"
    If you can't find a specific product, suggest the most relevant category or type of product.
    Always provide a recommendation, even if it's not an exact match.
    
    Catalog excerpts:
    {context}
    
    Remember, these are just excerpts. The full catalogs likely contain more products.
    """
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
    st.write(f"Loaded {len(pdf_contents)} catalog(s) successfully.")
    
    # User input
    user_input = st.text_input("Describe what you're looking for:")

    if user_input:
        # Get product recommendation
        recommendation = get_product_recommendation(user_input, pdf_contents)
        st.subheader("Recommended Product:")
        st.write(recommendation)

    # Debug information
    if st.checkbox("Show debugging information"):
        st.subheader("Catalog Contents (First 500 characters of each):")
        for pdf in pdf_contents:
            st.write(f"**{pdf['filename']}**")
            st.write(pdf['content'][:500] + "...")