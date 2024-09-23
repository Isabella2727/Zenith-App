import os
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdfs(input_dir, output_file):
    data = []
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            text = pdf_to_text(pdf_path)
            data.append({
                'filename': pdf_file,
                'text': text
            })
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Processed {len(data)} PDF files. Output saved to {output_file}")

if __name__ == "__main__":
    input_dir = "data/raw_pdfs"
    output_file = "data/processed_text/pdf_contents.csv"
    process_pdfs(input_dir, output_file)