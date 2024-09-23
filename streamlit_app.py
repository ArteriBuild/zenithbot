import streamlit as st
import pandas as pd
import PyPDF2
import re

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to parse the extracted text into a structured format
def parse_catalog_text(text):
    products = []
    lines = text.split('\n')
    current_product = {}
    
    for line in lines:
        if re.match(r'^[A-Z\s]+$', line):  # Category
            if current_product:
                products.append(current_product)
            current_product = {'Category': line.strip()}
        elif 'Code' in line and 'P/Coat' in line:  # Table header
            continue
        elif re.match(r'^\d+', line):  # Product line
            parts = line.split()
            if len(parts) >= 4:
                current_product = {
                    'Size': ' '.join(parts[:-3]),
                    'Code': parts[-3],
                    'P/Coat': parts[-2],
                    'S/Steel': parts[-1] if len(parts) > 4 else 'N/A'
                }
                products.append(current_product)
    
    return pd.DataFrame(products)

# Load and process the PDF
@st.cache_resource
def load_catalog():
    pdf_path = "catalogs/KM Tubular Commercial Furniture Product Catalog.pdf"
    text = extract_text_from_pdf(pdf_path)
    df = parse_catalog_text(text)
    return df

# Function to search products based on user query
def search_products(df, query):
    query = query.lower()
    return df[df.apply(lambda row: query in ' '.join(row.astype(str).values).lower(), axis=1)]

# Load the catalog
df = load_catalog()

# Streamlit app
st.title("KM Tubular Commercial Furniture Product Catalog Search")

# User input
user_query = st.text_input("Enter your product search query:")

if user_query:
    # Search for products
    results = search_products(df, user_query)
    
    # Display results
    st.subheader("Matching Products:")
    if not results.empty:
        for _, product in results.iterrows():
            st.write("---")
            for col in df.columns:
                if pd.notna(product[col]):
                    st.write(f"**{col}:** {product[col]}")
    else:
        st.write("No matching products found.")

# Option to view all data
if st.checkbox("Show all catalog data"):
    st.dataframe(df)

# Display column names
st.subheader("Available Product Information:")
st.write(", ".join(df.columns))