import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Function to load data from the URL
@st.cache_data
def load_data(url):
    response = requests.get(url)
    if 'text/csv' in response.headers.get('Content-Type', ''):
        data = StringIO(response.text)
        df = pd.read_csv(data, sep='\t')
        return df
    else:
        return None

# Function to search products based on user query
def search_products(df, query):
    query = query.lower()
    matches = df[df.apply(lambda row: query in ' '.join(row.astype(str).values).lower(), axis=1)]
    return matches

# Load the data
url = "https://claude.site/artifacts/468bee8b-00e4-42bc-85d8-7164ad07eaf1"
df = load_data(url)

# Streamlit app
st.title("Product Catalog Search")

if df is not None:
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
        st.write(df)

    # Display column names
    st.subheader("Available Product Information:")
    st.write(", ".join(df.columns))

else:
    st.error("Unable to retrieve product catalog data. The data source may be unavailable or in an unexpected format. Please try again later or contact support.")