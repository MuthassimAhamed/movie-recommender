import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests as req
from ratings_store import save_rating, load_ratings

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]# <-- paste your real Groq API key here

st.set_page_config(page_title="Movie & Web Series Recommender", page_icon="🎬", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #12131f 0%, #1a1830 100%);
    color: #F5F1E8;
}

.marquee-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 3rem;
    text-align: center;
    color: #E8B34C;
    letter-spacing: 0.02em;
    margin-bottom: 0.1em;
    text-shadow: 0 0 20px rgba(232,179,76,0.25);
}

.marquee-sub {
    text-align: center;
    color: #A79C8E;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.film-divider {
    height: 12px;
    background-image: repeating-linear-gradient(90deg, #8C2F39 0px, #8C2F39 10px, transparent 10px, transparent 20px);
    opacity: 0.45;
    margin: 1.6rem 0;
    border-radius: 4px;
}

div[data-baseweb="input"] > div {
    background-color: #1e1b2e !important;
    border: 1px solid #8C2F39 !important;
    border-radius: 8px !important;
}
input {
    color: #F5F1E8 !important;
}

.stButton>button {
    background-color: #8C2F39;
    color: #F5F1E8;
    border: none;
    border-radius: 6px;
    padding: 0.6em 1.8em;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #a53a46;
    color: #ffffff;
}

.ai-panel {
    background: linear-gradient(135deg, #2a1f3d 0%, #1e1b2e 100%);
    border: 1px solid #E8B34C;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1.4rem 0;
}
.ai-panel-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    color: #E8B34C;
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
}
.ai-panel-body {
    color: #F5F1E8;
    line-height: 1.6;
    font-size: 0.98rem;
}

.result-card {
    background-color: #1e1b2e;
    border-left: 4px solid #E8B34C;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F5F1E8;
    margin-bottom: 0.2rem;
}
.result-meta {
    color: #E8B34C;
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}
.result-overview {
    color: #cfc7b8;
    font-size: 0.92rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------- Data & model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("master_dataset.csv")
    embeddings = np.load("embeddings.npy")
    return df, embeddings

model = load_model()
df, embeddings = load_data()

def search(query, top_n=5, content_type="both"):
    query_embedding = model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = similarities / norms

    df_scored = df.copy()
    df_scored["similarity"] = similarities

    if content_type == "movie":
        df_scored = df_scored[df_scored["type"] == "movie"]
    elif content_type == "series":
        df_scored = df_scored[df_scored["type"] == "series"]

    top_results = df_scored.sort_values("similarity", ascending=False).head(top_n)
    return top_results

def apply_rating_boost(results, user_id="local_user"):
    ratings = load_ratings()
    user_ratings = ratings[ratings["user_id"] == user_id]

    if user_ratings.empty:
        return results

    results = results.copy()
    rating_map = dict(zip(user_ratings["title"], user_ratings["rating"]))

    def boost(row):
        if row["title"] in rating_map:
            # Rating of 5 -> +0.10 boost, rating of 1 -> -0.10 penalty, 3 -> neutral
            return (rating_map[row["title"]] - 3) * 0.05
        return 0

    results["adjustment"] = results.apply(boost, axis=1)
    results["similarity"] = results["similarity"] + results["adjustment"]
    return results.sort_values("similarity", ascending=False)

def ai_explain(query, results):
    candidates_text = "\n".join([
        f"- {row['title']} ({row['type']}): {row['overview'][:150]}"
        for _, row in results.iterrows()
    ])

    prompt = f"""You are a recommendation assistant. A user wants: "{query}"

Below is a numbered list of candidates. You must ONLY recommend from this exact list. Do not invent, add, or mention ANY title that is not in this list, even if you know of a similar one.

Candidates:
{candidates_text}

Pick the best 2-3 matches ONLY from the list above. Use the titles exactly as written. Explain briefly why each fits the user's request. Be conversational and friendly."""

    response = req.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# ---------- UI ----------
st.markdown('<div class="marquee-title">🎬 NOW SHOWING</div>', unsafe_allow_html=True)
st.markdown('<div class="marquee-sub">Tell us what you\'re in the mood for — movies and web series, matched by meaning.</div>', unsafe_allow_html=True)
st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

query = st.text_input("What are you in the mood for?", placeholder="e.g. a mysterious thriller with a slow burn plot")
content_type = st.radio("Show me:", ["both", "movie", "series"], horizontal=True)
top_n = st.slider("How many results?", 1, 10, 5)

if st.button("Find recommendations") and query:
    results = search(query, top_n=top_n, content_type=content_type)
    results = apply_rating_boost(results)

    with st.spinner("Thinking about the best picks..."):
        explanation = ai_explain(query, results)

    # Save to session state so rating interactions don't wipe the results
    st.session_state["results"] = results
    st.session_state["explanation"] = explanation

if "results" in st.session_state:
    results = st.session_state["results"]
    explanation = st.session_state["explanation"]

    st.markdown(f"""
    <div class="ai-panel">
        <div class="ai-panel-title">🤖 The Projectionist Recommends</div>
        <div class="ai-panel-body">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="marquee-sub" style="text-align:left; font-weight:600; color:#F5F1E8;">All matches found</div>', unsafe_allow_html=True)

    for _, row in results.iterrows():
        year = str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "N/A"
        rating = row["vote_average"] if pd.notna(row["vote_average"]) else "N/A"

        poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}" if pd.notna(row.get("poster_path")) else None

        col1, col2 = st.columns([1, 3])
        with col1:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown('<div style="background:#2a1f3d; border-radius:8px; height:220px; display:flex; align-items:center; justify-content:center; color:#A79C8E;">No poster</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card" style="margin-bottom:0;">
                <div class="result-title">{row['title']} <span style="color:#A79C8E; font-weight:400; font-size:1rem;">({row['type']}, {year})</span></div>
                <div class="result-meta">⭐ {rating} &nbsp;|&nbsp; 🎭 {row['genre_names']}</div>
                <div class="result-overview">{row['overview']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form(key=f"form_{row['title']}"):
                user_rating = st.select_slider(
                    "Your rating",
                    options=[0, 1, 2, 3, 4, 5],
                    value=0,
                    key=f"rate_{row['title']}"
                )
                user_review = st.text_area(
                    "Why? (optional)",
                    placeholder="e.g. loved the slow build-up, but the ending felt rushed",
                    key=f"review_{row['title']}",
                    height=80
                )
                submitted = st.form_submit_button("Save rating")

                if submitted and user_rating > 0:
                    save_rating(user_id="local_user", title=row["title"], rating=user_rating, review=user_review)
                    st.caption(f"Saved: {user_rating} ⭐ — thanks for the feedback!")
                elif submitted and user_rating == 0:
                    st.caption("Pick a star rating before saving.")

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)