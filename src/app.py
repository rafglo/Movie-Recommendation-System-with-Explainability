# app.py
import streamlit as st
import pandas as pd
import sys
import os

# Ensure the src folder is in the system path so we can import our engine
sys.path.append(os.path.abspath('./src'))
from content_engine import get_content_recommendations

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie DNA Engine", page_icon="🍿", layout="centered")

# --- UI HEADER ---
st.title("🍿 Movie DNA Recommender")
st.markdown("""
Welcome to the Cold-Start routing engine! Type in a movie you love, and the 
system will analyze the **Tag Genome** (1,128 latent cinematic traits) to find 
movies with the exact same DNA.
""")
st.divider()

# --- USER INPUTS ---
col1, col2 = st.columns([3, 1])
with col1:
    user_movie = st.text_input("Enter a Movie Title:", placeholder="e.g., Matrix, The or Toy Story")
with col2:
    top_n = st.number_input("How many?", min_value=1, max_value=20, value=5)

# --- INFERENCE BUTTON ---
if st.button("🔮 Analyze DNA & Recommend", type="primary"):
    if user_movie:
        with st.spinner('Spinning up the Content Engine and calculating vectors...'):
            # Call our compiled backend!
            results = get_content_recommendations(user_movie, top_n=top_n)
            
            # Error handling if movie isn't found
            if isinstance(results, str):
                st.error(results)
            else:
                st.success(f"Top {top_n} matches found!")
                
                # Display results as a beautiful interactive dataframe
                st.dataframe(
                    results,
                    column_config={
                        "title": "Movie Title",
                        "genres": "Genres",
                        "similarity_score": st.column_config.ProgressColumn(
                            "DNA Match Score",
                            help="Cosine similarity based on Tag Genome",
                            format="%.3f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.warning("Please enter a movie title first!")