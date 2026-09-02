import requests
import pandas as pd
import time

API_KEY = "c316bde1b402a5a9dac22bc9a67753fa"

all_movies = []

# Fetch 50 pages (~1000 movies) of popular movies
for page in range(1, 51):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=popularity.desc&page={page}"
    response = requests.get(url)
    data = response.json()

    for movie in data.get("results", []):
        all_movies.append({
            "title": movie.get("title"),
            "type": "movie",
            "genre_ids": movie.get("genre_ids"),
            "overview": movie.get("overview"),
            "release_date": movie.get("release_date"),
            "popularity": movie.get("popularity"),
            "vote_average": movie.get("vote_average"),
            "poster_path": movie.get("poster_path")
        })

    time.sleep(0.25)

df = pd.DataFrame(all_movies)
df.to_csv("movies_tmdb.csv", index=False)
print(f"Saved {len(df)} movies to movies_tmdb.csv")
print(df.head())