import requests
import pandas as pd
import ast

API_KEY = "c316bde1b402a5a9dac22bc9a67753fa"

# Fetch genre mappings for both movies and TV
movie_genres_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}"
tv_genres_url = f"https://api.themoviedb.org/3/genre/tv/list?api_key={API_KEY}"

movie_response = requests.get(movie_genres_url)
tv_response = requests.get(tv_genres_url)

print("Movie genres status:", movie_response.status_code)
print("TV genres status:", tv_response.status_code)

movie_genres = movie_response.json()["genres"]
tv_genres = tv_response.json()["genres"]

# Combine into one lookup dictionary: {id: name}
genre_lookup = {g["id"]: g["name"] for g in movie_genres}
genre_lookup.update({g["id"]: g["name"] for g in tv_genres})

# Load your dataset
df = pd.read_csv("master_dataset.csv")

def convert_genre_ids(genre_ids_str):
    try:
        ids = ast.literal_eval(genre_ids_str)
        names = [genre_lookup.get(i, "") for i in ids]
        return ", ".join([n for n in names if n])
    except:
        return ""

df["genre_names"] = df["genre_ids"].apply(convert_genre_ids)

df.to_csv("master_dataset.csv", index=False)
print("Done! Sample:")
print(df[["title", "type", "genre_names"]].head(10))