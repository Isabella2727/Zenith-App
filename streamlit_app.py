import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import re

# Configure Gemini AI using the secret from secrets.toml
gemini_api_key = st.secrets["Gemini_API"]
genai.configure(api_key=gemini_api_key)

# ... [Previous code for read_pdf_files and other functions remains the same] ...

# Function to get product recommendations using Gemini
def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    model = genai.GenerativeModel('gemini-pro')
    
    # Combine all catalog contents
    all_content = "\n\n".join([f"Catalog: {content['filename']}\n{content['content']}" for content in pdf_contents])
    
    prompt = f"""You are a knowledgeable sales assistant with access to product catalogs. 
    Based on the following catalog contents, recommend a product for this customer request: "{user_input}"
    
    If you can't find a specific product, suggest the most relevant category or type of product.
    Always provide a recommendation, even if it's not an exact match.
    If you find multiple suitable products, list up to three options.
    
    Catalog contents:
    {all_content}
    
    Please format your response as follows:
    1. Recommended Product(s): [List the product(s) here]
    2. Reason for Recommendation: [Explain why you recommended this product]
    3. Additional Information: [Provide any relevant details about the product(s)]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating recommendations: {str(e)}"

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
        with st.spinner("Generating recommendation..."):
            recommendation = get_product_recommendation(user_input, pdf_contents)
        st.subheader("Recommended Product:")
        st.write(recommendation)

    # Debug information
    if st.checkbox("Show debugging information"):
        st.subheader("Catalog Contents:")
        for pdf in pdf_contents:
            st.write(f"**{pdf['filename']}**")
            st.write(f"Total characters: {len(pdf['content'])}")
            st.text_area(f"Preview of {pdf['filename']}", pdf['content'][:1000] + "...", height=200)

# Display any errors that occurred during PDF processing
if 'errors' in st.session_state and st.session_state.errors:
    st.error("Errors occurred while processing PDFs:")
    for error in st.session_state.errors:
        st.write(error)