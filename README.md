# 🎬 Movie & Web Series Recommender

An AI-powered recommendation app that suggests movies and web series based on natural language descriptions of what you're in the mood for — combining semantic search over real movie/series data with AI-generated explanations.

🔗 **Live app:** https://muthassimahamed-movie-recommender-app-bedxeh.streamlit.app

## Features

- 🔍 **Semantic search** — describe what you want in plain English (e.g. "a mysterious thriller with a slow burn plot") and get relevant matches, powered by sentence embeddings
- 🎭 **Movies + web series** — combined dataset from TMDb covering both categories
- 🤖 **AI-generated explanations** — an LLM (via Groq) explains why each recommendation fits your request
- ⭐ **Ratings & reviews** — rate titles and leave notes; your ratings subtly boost similar future recommendations
- 🖼️ **Posters** — real poster images pulled from TMDb
- 🎨 **Custom cinema-themed UI** built with Streamlit

## Tech stack

- **Frontend:** Streamlit
- **Data:** [TMDb API](https://www.themoviedb.org/documentation/api), [MovieLens](https://grouplens.org/datasets/movielens/)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **AI reasoning:** Groq API (`openai/gpt-oss-20b`)
- **Core libraries:** pandas, numpy, scikit-learn

## Running it locally

1. Clone the repo and create a virtual environment
2. Install dependencies: pip install -r requirements.txt
3. Add your API keys to `.streamlit/secrets.toml`:
```toml
   GROQ_API_KEY = "your_groq_key"
```
4. Run the data pipeline scripts (`fetch_movies.py`, `fetch_series.py`, `combine_data.py`, `add_genre_names.py`, `build_embeddings.py`) to generate the dataset
5. Launch the app: streamlit run app.py

## How it works

1. User describes what they're in the mood for
2. The app converts the query into an embedding and finds the closest matching titles by cosine similarity
3. Past ratings subtly re-rank the results
4. An LLM explains why the top picks fit the request, grounded strictly in the retrieved candidates
5. Results are displayed with posters, ratings, and genres — with an option to rate and review each title

## Roadmap ideas

- Real collaborative filtering using MovieLens + accumulated user ratings
- Search by actor/director
- User accounts for multi-person personalization

