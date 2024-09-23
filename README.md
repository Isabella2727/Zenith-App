# PDF Catalog Processor

This project processes PDF catalogs for machine learning-based information extraction.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Unix or MacOS: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Place your PDF catalogs in the `data/raw_pdfs` directory
2. Run the processing script: `python scripts/process_pdfs.py`
3. Processed data will be saved in `data/processed_text/pdf_contents.csv`

## Directory Structure

- `data/`
  - `raw_pdfs/`: Place your PDF catalogs here
  - `processed_text/`: Contains processed text data
- `scripts/`: Python scripts for data processing
- `venv/`: Virtual environment (not tracked by git)