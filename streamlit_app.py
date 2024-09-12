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
    return re.sub(r'\s+', ' ', text).strip()

def extract_products(text, catalog_name):
    # Improved pattern matching for product extraction
    pattern = r'(\S+(?:\s+\S+){0,5})\s*\$\s*([\d,]+(?:\.\d{2})?)\s*(.+?)(?=\$|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    products = []
    for m in matches:
        name = clean_text(m[0])
        price = m[1]
        description = clean_text(m[2])
        if name and price and description:
            products.append({
                'name': name,
                'price': price,
                'catalog': catalog_name,
                'description': description
            })
    return products

def validate_product(product):
    errors = []
    
    # Validate Name
    if not isinstance(product['name'], str) or len(product['name']) < 2:
        errors.append("Invalid name")
    
    # Validate Price
    try:
        price = float(product['price'].replace(',', ''))
        if price <= 0:
            errors.append("Invalid price")
    except ValueError:
        errors.append("Invalid price format")
    
    # Validate Catalog
    if not isinstance(product['catalog'], str) or not product['catalog'].endswith('.pdf'):
        errors.append("Invalid catalog name")
    
    # Validate Description
    if not isinstance(product['description'], str) or len(product['description']) < 5:
        errors.append("Invalid description")
    
    return errors

def process_catalogs(folder_name):
    folder_path = os.path.join(script_dir, folder_name)
    all_products = []
    invalid_products = []
    
    if not os.path.exists(folder_path):
        st.error(f"The '{folder_name}' folder does not exist in {script_dir}. Please create it and add your PDF catalogs.")
        return all_products, invalid_products
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in the '{folder_name}' folder. Please add your PDF catalogs.")
        return all_products, invalid_products

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                cleaned_text = clean_text(text)
                products = extract_products(cleaned_text, filename)
                
                for product in products:
                    errors = validate_product(product)
                    if errors:
                        invalid_products.append((product, errors))
                    else:
                        all_products.append(product)
                        
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    
    return all_products, invalid_products

def match_products_to_brief(project_brief, products, top_n=5):
    if not products:
        return []
    
    product_texts = [f"{p['name']} {p['description']}" for p in products]
    vectorizer = TfidfVectorizer(stop_words='english')
    product_vectors = vectorizer.fit_transform(product_texts)
    
    brief_vector = vectorizer.transform([project_brief])
    similarities = cosine_similarity(brief_vector, product_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return [products[i] for i in top_indices]

# Streamlit app
st.title('Smart Product Selection App')

# Process catalogs and build product library
catalog_folder = 'catalogs'
products, invalid_products = process_catalogs(catalog_folder)

if not products:
    st.info("To use this app, please follow these steps:")
    st.markdown(f"""
    1. Ensure there's a folder named 'catalogs' in this directory: {script_dir}
    2. Add your PDF catalog files to the 'catalogs' folder.
    3. Restart the Streamlit app.
    """)
else:
    st.write(f"Loaded {len(products)} valid products from catalogs successfully.")
    
    if invalid_products:
        st.warning(f"Found {len(invalid_products)} invalid products. Check debug information for details.")
    
    # User input for project brief
    project_brief = st.text_area("Enter your project brief:")

    if project_brief:
        # Match products to project brief
        with st.spinner("Matching products to your project brief..."):
            matched_products = match_products_to_brief(project_brief, products)
        
        st.subheader("Matching Products:")
        for i, product in enumerate(matched_products, 1):
            st.markdown(f"""
            **Product {i}:**
            - **Name:** {product['name']}
            - **Price:** ${product['price']}
            - **Catalog:** {product['catalog']}
            - **Description:** {product['description']}
            """)
            st.markdown("---")

    # Debug information
    if st.checkbox("Show debugging information"):
        st.subheader("Product Library Overview:")
        st.write(f"Total valid products: {len(products)}")
        st.write("Sample valid products:")
        for product in products[:5]:
            st.json(product)
        
        if invalid_products:
            st.subheader("Invalid Products:")
            for product, errors in invalid_products:
                st.json(product)
                st.write(f"Errors: {', '.join(errors)}")
                st.markdown("---")

# Display any errors that occurred during processing
if 'errors' in st.session_state and st.session_state.errors:
    st.error("Errors occurred during processing:")
    for error in st.session_state.errors:
        st.write(error)