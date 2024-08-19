import streamlit as st
import pdfplumber
from transformers import pipeline
import io
import re
from collections import defaultdict

# Initialize the question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
       text = ""
       with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
           for page in pdf.pages:
               text += page.extract_text() + "\n"
       return text

# Function to process catalogs
def process_catalogs(catalogs):
    catalog_data = {}
    for catalog in catalogs:
        text = extract_text_from_pdf(catalog)
        catalog_data[catalog.name] = text
    return catalog_data

# Function to extract furniture items from text
def extract_furniture_items(text):
    # This is a simplified extraction. In a real scenario, you'd want a more robust method.
    items = re.findall(r'\b(\w+(?:\s+\w+){0,3}(?:chair|table|desk|sofa|cabinet|shelf))\b', text.lower())
    return list(set(items))  # Remove duplicates

# Function to generate recommendations
def recommend_furniture(brief_text, additional_details, catalog_data):
    context = " ".join(catalog_data.values())
    query = brief_text + " " + additional_details
    
    requested_items = extract_furniture_items(query)
    
    recommendations = defaultdict(list)
    missing_items = []
    
    for item in requested_items:
        result = qa_model(question=f"What {item} is available?", context=context)
        if result['score'] > 0.1:  # Adjust this threshold as needed
            recommendations[item].append(result['answer'])
        else:
            missing_items.append(item)
    
    return dict(recommendations), missing_items

# Streamlit UI
st.title("Furniture Recommendation App")

# File uploaders
uploaded_catalogs = st.file_uploader("Upload product catalogs (PDFs)", type="pdf", accept_multiple_files=True)
uploaded_brief = st.file_uploader("Upload project brief (PDF)", type="pdf")

# Text input for additional details
additional_details = st.text_area("Additional project details")

if st.button("Generate Recommendations"):
    if uploaded_catalogs and uploaded_brief:
        with st.spinner("Processing..."):
            # Process catalogs
            catalog_data = process_catalogs(uploaded_catalogs)
            
            # Process brief
            brief_text = extract_text_from_pdf(uploaded_brief)
            
            # Generate recommendations
            recommendations, missing_items = recommend_furniture(brief_text, additional_details, catalog_data)
            
            # Display results
            st.subheader("Recommendations:")
            for item, recs in recommendations.items():
                st.write(f"For {item}:")
                for rec in recs:
                    st.write(f"- {rec}")
                st.write("")
            
            st.subheader("Items without good matches:")
            for item in missing_items:
                st.write(f"- {item}")
    else:
        st.error("Please upload both catalogs and a project brief.")

# Add a section to display the contents of the uploaded files
if st.checkbox("Show uploaded file contents"):
    if uploaded_catalogs:
        st.subheader("Catalog Contents:")
        for catalog in uploaded_catalogs:
            st.write(f"Contents of {catalog.name}:")
            st.text(extract_text_from_pdf(catalog)[:1000] + "...")  # Display first 1000 characters
    
    if uploaded_brief:
        st.subheader("Project Brief Contents:")
        st.text(extract_text_from_pdf(uploaded_brief)[:1000] + "...")  # Display first 1000 characters