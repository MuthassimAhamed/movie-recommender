import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data and embeddings
df = pd.read_csv("master_dataset.csv")
embeddings = np.load("embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

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
    return top_results[["title", "type", "genre_names", "release_date", "vote_average", "overview", "similarity"]]

def print_results(results):
    for _, row in results.iterrows():
        year = str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "N/A"
        rating = row["vote_average"] if pd.notna(row["vote_average"]) else "N/A"
        print(f"{row['title']} ({row['type']}, {year}) - Rating: {rating} - Similarity: {row['similarity']:.3f}")
        print(f"  Genres: {row['genre_names']}")
        print(f"  {row['overview'][:100]}...")
        print()

query = "a mysterious thriller with a slow burn plot"

print("=== Both movies and series ===")
print_results(search(query, top_n=3, content_type="both"))

print("=== Movies only ===")
print_results(search(query, top_n=3, content_type="movie"))

print("=== Series only ===")
print_results(search(query, top_n=3, content_type="series"))