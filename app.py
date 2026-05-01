import streamlit as st
import pandas as pd

# The clean, Pythonic way to import from a subfolder
from src.content_engine import get_content_recommendations

# --- PAGE CONFIG ---
st.set_page_config(page_title="NLP Movie Engine", page_icon="🍿", layout="centered")

st.title("🍿 NLP Movie Recommender")
st.markdown("""
Welcome to the Cold-Start routing engine! Type in a movie you love, and the 
system will use **TF-IDF & Cosine Similarity** on user tags and genres to find 
movies with the exact same DNA.
""")
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    user_movie = st.text_input("Enter a Movie Title:", placeholder="e.g., Matrix, The or Toy Story")
with col2:
    top_n = st.number_input("How many?", min_value=1, max_value=20, value=5)

if st.button("🔮 Analyze NLP & Recommend", type="primary"):
    if user_movie:
        with st.spinner('Calculating TF-IDF vectors...'):
            results = get_content_recommendations(user_movie, top_n=top_n)
            
            if isinstance(results, str):
                st.error(results)
            else:
                st.success(f"Top {top_n} matches found!")
                st.dataframe(
                    results,
                    column_config={
                        "title": "Movie Title",
                        "genres": "Genres",
                        "similarity_score": st.column_config.ProgressColumn(
                            "NLP Match Score",
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