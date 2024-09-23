import streamlit as st
import pandas as pd
import PyPDF2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
    current_category = ""
    for line in lines:
        if re.match(r'^[A-Z\s]+$', line):  # Category
            current_category = line.strip()
        elif 'Code' in line and 'P/Coat' in line:  # Table header
            continue
        elif re.match(r'^\d+', line):  # Product line
            parts = line.split()
            if len(parts) >= 4:
                product = {
                    'Category': current_category,
                    'Size': ' '.join(parts[:-3]),
                    'Code': parts[-3],
                    'P/Coat': parts[-2],
                    'S/Steel': parts[-1] if len(parts) > 4 else 'N/A'
                }
                products.append(product)
    return pd.DataFrame(products)

# Load and process the PDF
@st.cache_resource
def load_catalog():
    pdf_path = "catalogs/KM Tubular Commercial Furniture Product Catalog.pdf"
    text = extract_text_from_pdf(pdf_path)
    df = parse_catalog_text(text)
    return df

# Function to process query and search products
def process_query_and_search(df, query):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    query_tokens = [word.lower() for word in word_tokenize(query) if word.isalnum() and word.lower() not in stop_words]
    
    # Search for products
    results = df[df.apply(lambda row: any(token in ' '.join(row.astype(str).values).lower() for token in query_tokens), axis=1)]
    
    return results, query_tokens

# Function to generate response
def generate_response(results, query_tokens):
    if results.empty:
        return "I'm sorry, but I couldn't find any products matching your query. Could you please try rephrasing or provide more details?"
    
    response = f"Based on your query, I found {len(results)} matching products. Here are some options that might interest you:\n\n"
    
    for _, product in results.iterrows():
        response += f"- A {product['Category']} with size {product['Size']}. "
        response += f"It's available in powder coat finish for ${product['P/Coat']}. "
        if product['S/Steel'] != 'N/A':
            response += f"There's also a stainless steel version available for ${product['S/Steel']}. "
        response += f"The product code is {product['Code']}.\n\n"
    
    response += "Would you like more details on any specific product or have any other questions?"
    return response

# Load the catalog
df = load_catalog()

# Streamlit app
st.title("KM Tubular Commercial Furniture Product Assistant")

# User input
user_query = st.text_input("How can I help you with furniture today?")

if user_query:
    # Process query and search for products
    results, query_tokens = process_query_and_search(df, user_query)
    
    # Generate and display response
    response = generate_response(results, query_tokens)
    st.write(response)

# Option to view all data
if st.checkbox("Show full product catalog"):
    st.dataframe(df)