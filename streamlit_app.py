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
    input_text = f"Customer request: {user_input}\n\nCatalog contents: {all_content}"
    
    # Tokenize and truncate
    tokens = tokenizer.tokenize(input_text)
    if len(tokens) > 510:  # Reserve 2 spots: 1 for [CLS] and 1 for [MASK]
        tokens = tokens[:510]
    tokens.append('[MASK]')
    
    # Convert tokens to ids and create attention mask
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens)
    attention_mask = [1] * len(input_ids)
    
    # Pad sequences to max length
    padding_length = 512 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    
    # Convert to tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    return f"""Based on the customer request and catalog contents, a recommended product or category might be:
    {predicted_token}
    
    Please note that this is a general suggestion based on the available information. For more specific recommendations, please consult the full product catalogs."""

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
        st.subheader("Recommended Product or Category:")
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