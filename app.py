import streamlit as st
import pandas as pd

# Import the Hybrid Engine
from src.hybrid_engine import HybridRecommender

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Movie Engine", page_icon="🍿", layout="wide")

# Cache the engine so it only boots up the PyTorch model once
@st.cache_resource
def load_engine():
    return HybridRecommender()

engine = load_engine()

st.title("🍿 Deep Learning Movie Recommender")
st.markdown("""
Welcome to the **Hybrid RecSys**. 
1. **Pipeline B (TF-IDF)** finds 50 movies with similar DNA to your seed movie.
2. **Pipeline A (Neural CF)** predicts exactly how much you will like them.
""")
st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Tuning Parameters")
user_id = st.sidebar.number_input("Who is watching? (User ID)", min_value=1, max_value=610, value=1)
user_movie = st.sidebar.text_input("Seed Movie Title:", value="Matrix, The")
top_n = st.sidebar.slider("How many recommendations?", min_value=1, max_value=10, value=5)

# --- MAIN APP ---
if st.button("🔮 Generate Personalized Picks", type="primary"):
    if user_movie:
        with st.spinner(f'Consulting the Neural Network for User #{user_id}...'):
            results = engine.recommend(user_id=user_id, movie_title=user_movie, top_n=top_n)
            
            if isinstance(results, str):
                st.error(results)
            else:
                st.success("Analysis Complete!")
                
                # --- DISPLAY RESULTS ---
                for index, row in results.iterrows():
                    with st.container():
                        st.subheader(f"🎬 {row['title']}")
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            rating = float(row['predicted_rating'])
                            st.metric(label="Predicted Rating", value=f"⭐ {rating:.2f} / 5.0")
                            
                        with col2:
                            sim = float(row['similarity_score'])
                            st.metric(label="TF-IDF DNA Match", value=f"{sim:.3f}")
                            
                        with col3:
                            st.markdown("**Core Genres:**")
                            genres_formatted = " ".join([f"`{g}`" for g in row['genres'].split(' ')])
                            st.markdown(genres_formatted)
                            
                        st.divider()
    else:
        st.warning("Please enter a movie title first!")