import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_products(text):
    # This is a simple pattern matching. You might need to adjust based on your catalog format
    pattern = r'([\w\s]+)\s*\$([\d,]+\.?\d*)\s*(.+?)\s*(?=\$|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    products = [{'name': clean_text(m[0]), 'price': m[1], 'description': clean_text(m[2])} for m in matches]
    return products

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
                products = extract_products(cleaned_text)
                pdf_contents.append({"filename": filename, "products": products})
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    return pdf_contents

def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    
    # Prepare TF-IDF vectorizer
    all_products = [p for pdf in pdf_contents for p in pdf['products']]
    product_texts = [f"{p['name']} {p['description']}" for p in all_products]
    vectorizer = TfidfVectorizer(stop_words='english')
    product_vectors = vectorizer.fit_transform(product_texts)
    
    # Vectorize user input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate similarity
    similarities = cosine_similarity(user_vector, product_vectors).flatten()
    
    # Get top 5 matches
    top_indices = similarities.argsort()[-5:][::-1]
    
    recommendations = []
    for index in top_indices:
        product = all_products[index]
        recommendations.append(f"Product: {product['name']}")
        recommendations.append(f"Price: ${product['price']}")
        recommendations.append(f"Description: {product['description']}")
        recommendations.append("")
    
    return "\n".join(recommendations)

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
        st.subheader("Extracted Products:")
        for pdf in pdf_contents:
            st.write(f"**{pdf['filename']}**")
            st.write(f"Total products extracted: {len(pdf['products'])}")
            if pdf['products']:
                st.write("Sample product:")
                st.json(pdf['products'][0])

# Display any errors that occurred during PDF processing
if 'errors' in st.session_state and st.session_state.errors:
    st.error("Errors occurred while processing PDFs:")
    for error in st.session_state.errors:
        st.write(error)