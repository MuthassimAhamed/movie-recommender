import requests

API_KEY = "c316bde1b402a5a9dac22bc9a67753fa"

# Search for a popular TV/web series
url = f"https://api.themoviedb.org/3/discover/tv?api_key={API_KEY}&sort_by=popularity.desc"
response = requests.get(url)
data = response.json()

for show in data["results"][:5]:
    print(show["name"], "-", show.get("first_air_date"), "-", show.get("overview")[:60])