import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert_model()

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
        return pdf_contents
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in the '{folder_name}' folder. Please add your PDF catalogs.")
        return pdf_contents

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                cleaned_text = clean_text(text)
                pdf_contents.append({"filename": filename, "content": cleaned_text})
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    return pdf_contents

# Function to get product recommendations using BERT
def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    
    # Combine all catalog contents
    all_content = "\n\n".join([f"Catalog: {content['filename']}\n{content['content']}" for content in pdf_contents])
    
    # Prepare input for BERT
    input_text = f"Customer request: {user_input}\n\nCatalog contents: {all_content[:5000]}"  # Limit to 5000 chars for performance
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Generate mask tokens for recommendation
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    # If no mask token, add one at the end
    if len(mask_token_index) == 0:
        inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[tokenizer.mask_token_id]])], dim=-1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.tensor([[1]])], dim=-1)
        mask_token_index = torch.tensor([inputs["input_ids"].shape[1] - 1])
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    mask_word_prob = softmax[0, mask_token_index, :]
    top_5_words = torch.topk(mask_word_prob, 5, dim=-1).indices[0].tolist()
    
    recommended_words = tokenizer.decode(top_5_words)
    
    return f"""Based on the customer request and catalog contents, here are some recommended product keywords:
    {recommended_words}
    
    Please note that these are general suggestions based on the available information. For more specific recommendations, please consult the full product catalogs."""

# Streamlit app
st.title('Smart Product Selection App')

# Read PDF files from the 'catalogs' folder
catalog_folder = 'catalogs'
pdf_contents = read_pdf_files(catalog_folder)

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
        st.subheader("Recommended Products:")
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