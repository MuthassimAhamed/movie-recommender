import requests
import pandas as pd
import time

API_KEY = "c316bde1b402a5a9dac22bc9a67753fa"

all_shows = []

# Fetch 50 pages (~1000 shows) of popular TV/web series
for page in range(1, 51):
    url = f"https://api.themoviedb.org/3/discover/tv?api_key={API_KEY}&sort_by=popularity.desc&page={page}"
    response = requests.get(url)
    data = response.json()

    for show in data.get("results", []):
        all_shows.append({
            "title": show.get("name"),
            "type": "series",
            "genre_ids": show.get("genre_ids"),
            "overview": show.get("overview"),
            "release_date": show.get("first_air_date"),
            "popularity": show.get("popularity"),
            "vote_average": show.get("vote_average"),
            "poster_path": show.get("poster_path")
        })

    time.sleep(0.25)

df = pd.DataFrame(all_shows)
df.to_csv("web_series.csv", index=False)
print(f"Saved {len(df)} web series to web_series.csv")
print(df.head())