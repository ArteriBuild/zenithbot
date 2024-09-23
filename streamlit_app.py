import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # Adjust the file path as needed
    df = pd.read_excel('catalogs/furniture_catalog.xlsx')
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Rename columns if they don't have names
    df.columns = ['Document ID', 'Description'] if len(df.columns) == 2 else df.columns
    return df

# Load the data
df = load_data()

def recommend_furniture(query):
    # Combine Document ID and Description for vectorization
    df['combined_text'] = df['Document ID'] + ' ' + df['Description'].fillna('')
    
    # Create a list of product descriptions
    descriptions = df['combined_text'].tolist()
    
    # Add the query to the list of descriptions
    descriptions.append(query)
    
    # Vectorize the descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute similarity between query and all products
    query_vec = tfidf_matrix[-1]
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix[:-1]).flatten()
    
    # Get top 5 most similar products
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    return df.iloc[top_indices]

st.title("Furniture Recommendation App")

st.subheader("Available Catalog:")
st.write(f"Furniture catalog with {len(df)} items")

project_brief = st.text_area("Enter your project brief and requirements", height=200)

if st.button("Generate Recommendations"):
    if project_brief:
        recommendations = recommend_furniture(project_brief)
        
        st.subheader("Recommendations:")
        if not recommendations.empty:
            for _, rec in recommendations.iterrows():
                st.write("**Product Details:**")
                st.write(f"- Document ID: {rec['Document ID']}")
                st.write(f"- Description: {rec['Description']}")
                st.write("---")
        else:
            st.write("No specific recommendations found. Please try adjusting your project brief.")
    else:
        st.error("Please enter a project brief.")

if st.checkbox("Show catalog contents"):
    st.dataframe(df)

# Display column names for debugging
st.subheader("Column Names in Excel File:")
st.write(df.columns.tolist())