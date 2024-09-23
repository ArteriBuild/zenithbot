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
    product_info = {
        "product_name": "",
        "variation": "",
        "price": "",
        "specifications": ""
    }
    
    lines = text.split('\n')
    product_info["product_name"] = lines[0].strip()
    
    price_match = re.search(r'[\$£€]?\d+(?:[.,]\d{2})?', text)
    if price_match:
        product_info["price"] = price_match.group()
    
    dimensions_match = re.search(r'\d+(?:\.\d+)?[xX]\d+(?:\.\d+)?(?:[xX]\d+(?:\.\d+)?)?(?:\s*mm|\s*cm|\s*m)?', text)
    if dimensions_match:
        product_info["specifications"] = dimensions_match.group()
    
    other_info = ' '.join(lines[1:]).replace(product_info["price"], "").replace(product_info["specifications"], "").strip()
    product_info["variation"] = other_info
    
    return product_info

def extract_structured_data_from_pdf(pdf_path):
    structured_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
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
            
            if not tables:
                text = page.extract_text()
                chunks = text.split('\n\n')
                for chunk in chunks:
                    if len(chunk.strip()) > 0:
                        product_data = extract_product_info(chunk)
                        structured_data.append(product_data)
    
    return structured_data

@st.cache_data
def load_catalogs():
    catalogs = {}
    catalog_dir = "catalogs"
    for filename in os.listdir(catalog_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(catalog_dir, filename)
            catalogs[filename] = extract_structured_data_from_pdf(file_path)
    return catalogs

catalogs = load_catalogs()

def recommend_furniture(query, catalogs):
    all_products = []
    for catalog_data in catalogs.values():
        all_products.extend(catalog_data)
    
    context = json.dumps(all_products)
    
    # Generate multiple recommendations
    recommendations = []
    for _ in range(5):  # Try to get 5 recommendations
        result = qa_model(question=query, context=context)
        if result['answer'] not in recommendations:
            recommendations.append(result['answer'])
    
    # Match recommendations with product details
    detailed_recommendations = []
    for rec in recommendations:
        for product in all_products:
            if rec.lower() in product['product_name'].lower():
                detailed_recommendations.append(product)
                break
    
    return detailed_recommendations

st.title("Furniture Recommendation App")

st.subheader("Available Catalogs:")
for catalog_name in catalogs.keys():
    st.write(f"- {catalog_name}")

project_brief = st.text_area("Enter your project brief and requirements", height=200)

if st.button("Generate Recommendations"):
    if project_brief:
        recommendations = recommend_furniture(project_brief, catalogs)
        
        st.subheader("Recommendations:")
        if recommendations:
            for rec in recommendations:
                st.write(f"**Product:** {rec['product_name']}")
                st.write(f"**Price:** {rec['price']}")
                st.write(f"**Variation:** {rec['variation']}")
                st.write(f"**Specifications:** {rec['specifications']}")
                st.write("---")
        else:
            st.write("No specific recommendations found. Please try adjusting your project brief.")
    else:
        st.error("Please enter a project brief.")

if st.checkbox("Show catalog contents"):
    selected_catalog = st.selectbox("Select a catalog to view:", list(catalogs.keys()))
    st.table(pd.DataFrame(catalogs[selected_catalog]))