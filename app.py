import streamlit as st

from src.hybrid_ranker import HybridRanker


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_engine():
    return HybridRanker()


engine = load_engine()

st.title("Movie Recommendation System")

st.sidebar.header("Controls")
mode = st.sidebar.radio("Recommendation mode", ["For a user", "Similar movies"])
top_n = st.sidebar.slider("Number of results", min_value=5, max_value=20, value=10)
diversity_weight = st.sidebar.slider(
    "Diversity",
    min_value=0.0,
    max_value=0.3,
    value=0.0,
    step=0.05,
)


def render_recommendations(results):
    if results.empty:
        st.warning("No recommendations found for this selection.")
        return

    for _, row in results.iterrows():
        st.subheader(row["title"])
        col1, col2, col3 = st.columns([1, 2, 3])

        with col1:
            st.metric("Relevance", f"{float(row['relevance_score']):.3f}")
            if "item_knn_score" in row:
                st.caption(
                    f"KNN {row['item_knn_score']:.2f} | "
                    f"Pop {row['popularity_score']:.2f} | "
                    f"Genre {row['genre_score']:.2f}"
                )

        with col2:
            genre_tags = " ".join(f"`{genre.strip()}`" for genre in row["genres"].split("|"))
            st.markdown(genre_tags)

        with col3:
            if "explanation_summary" in row:
                st.markdown(row["explanation_summary"])

            explanations = row["explanations"]
            if explanations:
                st.markdown("Similar to:")
                for item in explanations:
                    st.markdown(f"- {item['title']} ({item['similarity']:.3f})")
            else:
                st.markdown("No strong similarity explanation available.")

            if "component_contributions" in row:
                with st.expander("Contribution details"):
                    contributions = row["component_contributions"]["contributions"]
                    for label, value in contributions.items():
                        st.markdown(f"`{label}`: {value:.3f}")

                    if "item_knn_shap" in row:
                        st.markdown("SHAP values")
                        st.markdown(f"`item_knn`: {row['item_knn_shap']:.3f}")
                        st.markdown(f"`popularity`: {row['popularity_shap']:.3f}")
                        st.markdown(f"`genre`: {row['genre_shap']:.3f}")

        st.divider()


if mode == "For a user":
    user_ids = engine.available_user_ids()
    user_id = st.sidebar.selectbox("User ID", user_ids, index=0)

    liked = engine.user_liked_movies(user_id, limit=8)
    if not liked.empty:
        st.caption(f"Recent high-rated movies for user {user_id}")
        st.dataframe(liked[["title", "rating"]], width="stretch", hide_index=True)

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Finding movies similar to this user's taste..."):
            recommendations = engine.recommend_for_user(
                user_id=user_id,
                top_n=top_n,
                diversity_weight=diversity_weight,
            )
        render_recommendations(recommendations)

else:
    seed_title = st.sidebar.text_input("Movie title", value="Matrix")

    if st.button("Find Similar Movies", type="primary"):
        with st.spinner("Computing item-item similarities..."):
            recommendations = engine.item_engine.recommend_similar_movies(seed_title, top_n=top_n)
        render_recommendations(recommendations)
