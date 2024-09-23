import streamlit as st
import os
import json
from tabula import read_pdf as tabula_read_pdf
import pandas as pd
from transformers import pipeline

# Initialize the question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

# Function to extract structured data from PDF using Tabula
def extract_structured_data_from_pdf(pdf_path):
    tables = tabula_read_pdf(pdf_path, pages='all', multiple_tables=True)
    structured_data = []

    for table in tables:
        for _, row in table.iterrows():
            try:
                product_name = row[0]
                variation = row[1]
                price = row[2]
                specs = row[3]
                
                structured_data.append({
                    "product_name": product_name,
                    "variation": variation,
                    "price": price,
                    "specifications": specs
                })
            except:
                # Skip rows that don't match the expected format
                continue

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

# File uploader for project brief
uploaded_brief = st.file_uploader("Upload project brief (PDF)", type="pdf")

# Text input for additional project details
additional_details = st.text_area("Additional project details or specific requirements")

if st.button("Generate Recommendations"):
    if uploaded_brief is not None:
        # Extract text from the uploaded brief
        brief_tables = tabula_read_pdf(uploaded_brief, pages='all', multiple_tables=True)
        brief_text = " ".join([table.to_string() for table in brief_tables])
        
        query = brief_text + " " + additional_details
        
        recommendations = recommend_furniture(query, catalogs)
        
        st.subheader("Recommendations:")
        st.write(recommendations)
    else:
        st.error("Please upload a project brief.")

# Option to view catalog contents
if st.checkbox("Show catalog contents"):
    selected_catalog = st.selectbox("Select a catalog to view:", list(catalogs.keys()))
    st.table(pd.DataFrame(catalogs[selected_catalog]))