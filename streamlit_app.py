import streamlit as st
import pdfplumber
import os
from transformers import pipeline
import io

# Initialize the question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages)

# Function to load pre-stored catalogs
@st.cache_data
def load_catalogs():
    catalogs = {}
    catalog_dir = "catalogs"  # Directory where your PDF catalogs are stored
    for filename in os.listdir(catalog_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(catalog_dir, filename)
            catalogs[filename] = extract_text_from_pdf(file_path)
    return catalogs

# Load pre-stored catalogs
catalogs = load_catalogs()

# Function to generate recommendations
def recommend_furniture(query, catalogs):
    context = " ".join(catalogs.values())
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
        brief_text = extract_text_from_pdf(uploaded_brief)
        query = brief_text + " " + additional_details
        
        recommendations = recommend_furniture(query, catalogs)
        
        st.subheader("Recommendations:")
        st.write(recommendations)
    else:
        st.error("Please upload a project brief.")

# Option to view catalog contents (for demonstration purposes)
if st.checkbox("Show catalog contents"):
    selected_catalog = st.selectbox("Select a catalog to view:", list(catalogs.keys()))
    st.text(catalogs[selected_catalog][:1000] + "...")  # Display first 1000 characters