import streamlit as st
import os
import json
import pdfplumber
import pandas as pd
import re
from transformers import pipeline

# Initialize the question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

def extract_product_info(text):
    # This function tries to extract product information from a chunk of text
    product_info = {
        "product_name": "",
        "variation": "",
        "price": "",
        "specifications": ""
    }
    
    # Try to identify product name (assumes it's the first line or a line with larger font)
    lines = text.split('\n')
    product_info["product_name"] = lines[0].strip()
    
    # Try to identify price (assumes it's a number with currency symbol)
    price_match = re.search(r'[\$£€]?\d+(?:[.,]\d{2})?', text)
    if price_match:
        product_info["price"] = price_match.group()
    
    # Assume everything else is either variation or specifications
    other_info = ' '.join(lines[1:]).replace(product_info["price"], "").strip()
    if len(other_info) > 50:  # If there's substantial text, treat it as specifications
        product_info["specifications"] = other_info
    else:
        product_info["variation"] = other_info
    
    return product_info

def extract_structured_data_from_pdf(pdf_path):
    structured_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try to extract tables
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                for _, row in df.iterrows():
                    product_data = {
                        "product_name": str(row.get(0, '')),
                        "variation": str(row.get(1, '')),
                        "price": str(row.get(2, '')),
                        "specifications": str(row.get(3, ''))
                    }
                    structured_data.append(product_data)
            
            # If no tables, try to extract text and parse it
            if not tables:
                text = page.extract_text()
                chunks = text.split('\n\n')  # Assume double newline separates products
                for chunk in chunks:
                    if len(chunk.strip()) > 0:
                        product_data = extract_product_info(chunk)
                        structured_data.append(product_data)
    
    return structured_data

# Function to load and process catalogs
@st.cache_data
def load_catalogs():
    catalogs = {}
    catalog_dir = "catalogs"
    for filename in os.listdir(catalog_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(catalog_dir, filename)
            catalogs[filename] = extract_structured_data_from_pdf(file_path)
    return catalogs

# Load catalogs
catalogs = load_catalogs()

# Function to generate recommendations
def recommend_furniture(query, catalogs):
    all_products = []
    for catalog_data in catalogs.values():
        all_products.extend(catalog_data)
    
    context = json.dumps(all_products)
    result = qa_model(question=query, context=context)
    return result['answer']

# Streamlit UI
st.title("Furniture Recommendation App")

# Display available catalogs
st.subheader("Available Catalogs:")
for catalog_name in catalogs.keys():
    st.write(f"- {catalog_name}")

# Text input for project brief
project_brief = st.text_area("Enter your project brief", height=200)

# Text input for additional project details
additional_details = st.text_area("Additional project details or specific requirements")

if st.button("Generate Recommendations"):
    if project_brief:
        query = project_brief + " " + additional_details
        
        recommendations = recommend_furniture(query, catalogs)
        
        st.subheader("Recommendations:")
        st.write(recommendations)
    else:
        st.error("Please enter a project brief.")

# Option to view catalog contents
if st.checkbox("Show catalog contents"):
    selected_catalog = st.selectbox("Select a catalog to view:", list(catalogs.keys()))
    st.table(pd.DataFrame(catalogs[selected_catalog]))