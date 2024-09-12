import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def extract_keywords(text, n=5):
    # Simple keyword extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    scores = dense.tolist()[0]
    scores_with_features = [(score, feature) for score, feature in zip(scores, feature_names)]
    sorted_scores = sorted(scores_with_features, reverse=True)
    return [feature for score, feature in sorted_scores[:n]]

def get_product_recommendation(user_input, pdf_contents):
    if not pdf_contents:
        return "No catalog data available for recommendations."
    
    # Extract keywords from user input
    user_keywords = extract_keywords(user_input)
    
    # Extract keywords from each catalog
    catalog_keywords = []
    for pdf in pdf_contents:
        keywords = extract_keywords(pdf['content'], n=20)
        catalog_keywords.append({'filename': pdf['filename'], 'keywords': keywords})
    
    # Find best matching catalogs
    best_matches = []
    for catalog in catalog_keywords:
        common_keywords = set(user_keywords) & set(catalog['keywords'])
        if common_keywords:
            best_matches.append((catalog['filename'], len(common_keywords), common_keywords))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    
    if not best_matches:
        return "No specific product recommendations found. Please try a different search term."
    
    recommendations = []
    for match in best_matches[:2]:  # Get top 2 matches
        filename, _, keywords = match
        recommendations.append(f"Catalog: {filename}")
        recommendations.append(f"Relevant keywords: {', '.join(keywords)}")
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
        st.subheader("Recommended Products or Categories:")
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